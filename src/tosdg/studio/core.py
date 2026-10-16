"""Studio core: scene evaluation and drag-IK on top of the tosdg skeleton/camera modules.

Everything the studio shows (FOV, 2D keypoints, capsule silhouettes) is computed with the
same FK and `Camera.project` the generator uses for ground truth, so what you see is what
`tosdg generate` would label. No HTTP here - see server.py.

Contents:
  constants: DEFAULT_CAMERA, DEFAULT_OFFICER
  officer_transform
  make_camera — Orbit parametrisation identical to camera.sample_camera…
  camera_summary
  dof_axes — World-space unit vector of the rotation axis of every DOF…
  evaluate — Project the posed skeleton through the camera
  match_gestures — Every gesture variant whose constraints the current pose…
  set_joint_angle — Turn one DOF of one joint to `value` degrees, as far as…
  solve_drag — Move point (joint, offset) towards `target_char`…
  camera_plane_target — World point under pixel `uv` at the same camera depth as…
  ground_target — Intersection of the pixel ray with the horizontal plane at…
  to_char_frame
  (+ 4 private helpers)
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .. import baton as batonmod
from ..camera import Camera, look_at_rotator
from ..lens import Lens
from ..gestures import approach_side, constraint_failures
from ..scene import DEFAULT_VALIDATION
from ..skeleton import Pose, Skeleton, axis_rotation

MIN_DISTANCE_CM, MAX_DISTANCE_CM = 50.0, 3000.0     # camera-to-officer; farther than this the officer is a few pixels and the Studio breaks
# default = dashcam 4 m away, looking straight at the officer, level, at neck height (neck_01 is 150 cm in proxy_v1)
DEFAULT_CAMERA = {
    "distance_cm": 400.0, "azimuth_deg": 0.0, "height_cm": 150.0, "aim_height_cm": 150.0,
    "yaw_offset_deg": 0.0, "pitch_offset_deg": 0.0, "roll_deg": 0.0,
    "focal_length_mm": 3.6, "sensor_width_mm": 6.17, "width": 1280, "height": 720,
    "lens_k1": 0.0, "lens_k2": 0.0, "lens_circle": 0.0,      # radial distortion (k1 < 0 barrel) and lit image circle (0 = whole frame), see lens.py
    "lens_border_auto": 0.0,                                # 1: the black border follows k1, k2, focal length and size (lens_circle is then ignored)
}
DEFAULT_OFFICER = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "z": 0.0}   # z: height of the standing surface above the ground (the podium); the camera is measured from the ground

# IK cost per DOF: higher = the solver prefers to move other joints first.
_BASE_COST = (("spine", 40.0), ("neck", 12.0), ("head", 8.0), ("clavicle", 4.0))


# --------------------------------------------------------------------------- scene math
def officer_transform(officer: dict) -> tuple[np.ndarray, np.ndarray]:
    return axis_rotation("z", float(officer["yaw_deg"])), np.array([float(officer["x"]), float(officer["y"]), float(officer.get("z", 0.0))])


def make_camera(cam: dict, officer: dict) -> Camera:
    """Orbit parametrisation identical to camera.sample_camera (azimuth in the officer frame)."""
    c = {**DEFAULT_CAMERA, **cam}
    o = np.array([float(officer["x"]), float(officer["y"]), 0.0])
    world_az = math.radians(float(officer["yaw_deg"]) + float(c["azimuth_deg"]))
    dist = float(c["distance_cm"])
    eye = o + np.array([dist * math.cos(world_az), dist * math.sin(world_az), 0.0])
    eye[2] = float(c["height_cm"])
    pitch, yaw = look_at_rotator(eye, o + np.array([0.0, 0.0, float(c["aim_height_cm"])]))
    k1, k2 = float(c["lens_k1"]), float(c["lens_k2"])
    f_px = float(c["focal_length_mm"]) * int(c["width"]) / float(c["sensor_width_mm"])
    circle = Lens.auto_circle(k1, k2, int(c["width"]), int(c["height"]), f_px) if float(c["lens_border_auto"]) >= 0.5 else float(c["lens_circle"])
    return Camera("studio", tuple(float(x) for x in eye), pitch + float(c["pitch_offset_deg"]),
                  yaw + float(c["yaw_offset_deg"]), float(c["roll_deg"]), float(c["focal_length_mm"]),
                  float(c["sensor_width_mm"]), int(c["width"]), int(c["height"]),
                  lens=Lens(k1, k2, circle) if (k1 or k2 or circle) else None)


def _px_scale(cam: Camera, pts: np.ndarray) -> np.ndarray:
    """Magnification of the lens at the pinhole image of these world points (1 without distortion): circles of radius r px become r * scale."""
    if cam.lens is None or not cam.lens.distorts:
        return np.ones(len(pts))
    uv0, _ = cam.project_pinhole(pts)
    return cam.lens.local_scale(uv0, cam.cx, cam.cy, cam.fx)


def camera_summary(cam: Camera) -> dict:
    diag_mm = math.hypot(cam.sensor_width_mm, cam.sensor_height_mm)
    return {
        "position": [float(x) for x in cam.position], "pitch": cam.pitch, "yaw": cam.yaw, "roll": cam.roll,
        "hfov_deg": cam.hfov_deg, "vfov_deg": cam.vfov_deg,
        "dfov_deg": float(np.degrees(2 * np.arctan(diag_mm / (2 * cam.focal_length_mm)))),
        "equiv_35mm_focal": cam.focal_length_mm * 36.0 / cam.sensor_width_mm,
        "fx": cam.fx, "cx": cam.cx, "cy": cam.cy, "width": cam.width, "height": cam.height,
        "R": cam.R.tolist(),
        "horizon_v": float(cam.horizon_v),
        "lens": {"k1": float(cam.lens.k1), "k2": float(cam.lens.k2), "circle": float(cam.lens.image_circle),
                 "r_limit": float(cam.lens.r_limit()) if cam.lens.distorts and np.isfinite(cam.lens.r_limit()) else None,    # None = no fold (JSON has no Infinity)
                 "r_reach": float(cam.lens.r_reach()) if cam.lens.distorts and np.isfinite(cam.lens.r_reach()) else None} if cam.lens else None,
    }


def _uv(p) -> list[float] | None:
    return [float(p[0]), float(p[1])] if np.all(np.isfinite(p)) else None


def dof_axes(sk: Skeleton, pose: Pose, fk) -> dict:
    """World-space unit vector of the rotation axis of every DOF of every joint, for the Alt+click rotation gizmo: {joint: {dof: [x, y, z]}}.
    A joint's DOFs are applied in the listed order, each about its own axis of the frame left by the previous ones, so later axes
    follow the earlier rotations. Turning the DOF by +v degrees rotates the part by sign*v degrees, right-handed, about this vector."""
    out = {}
    for name, j in sk.joints.items():
        if not j.dofs:
            continue
        Rp = fk.rotations[j.parent] if j.parent else fk.rotations[name] @ np.eye(3)
        R = np.eye(3)
        axes = {}
        for d in j.dofs:
            e = np.zeros(3)
            e[{"x": 0, "y": 1, "z": 2}[d["axis"]]] = 1.0
            axes[d["name"]] = (Rp @ R @ e).tolist()
            R = R @ axis_rotation(d["axis"], d["sign"] * float(pose.get(name, {}).get(d["name"], 0.0)))
        out[name] = axes
    return out


def evaluate(sk: Skeleton, pose: Pose, officer: dict, cam_params: dict, validation: dict | None = None,
             baton: dict | None = None) -> dict:
    """Project the posed skeleton through the camera. Returns plain JSON-able data. `baton` = a normalised baton spec plus "mode"
    (the pose already carries the grip, see baton.apply_grip): the baton is projected and counts towards the bbox."""
    R, t = officer_transform(officer)
    fk = sk.fk(pose).transformed(R, t)
    cam = make_camera(cam_params, officer)

    names = list(sk.joints)
    pts = np.array([fk.positions[n] for n in names])
    uv, depth = cam.project(pts)
    joints = {n: {"w": pts[i].tolist(), "uv": _uv(uv[i]), "depth": float(depth[i])} for i, n in enumerate(names)}

    keypoints = {}
    for set_name in sk.keypoint_sets:
        kn, kp = sk.keypoints(fk, set_name)
        kuv, kd = cam.project(kp)
        keypoints[set_name] = [
            {"name": kn[i], "joint": sk.keypoint_sets[set_name][i]["joint"],
             "offset": list(sk.keypoint_sets[set_name][i]["offset"]),
             "w": kp[i].tolist(), "uv": _uv(kuv[i]), "depth": float(kd[i])} for i in range(len(kn))]

    segments, extent = [], []
    for seg in sk.segments.values():
        p = np.stack([fk.positions[seg["from"]], fk.positions[seg["to"]]])
        suv, sd = cam.project(p)
        if np.any(sd <= 1.0) or not np.all(np.isfinite(suv)):
            continue
        r = seg["radius"] * cam.fx / sd * _px_scale(cam, p)
        segments.append({"name": seg["name"], "region": seg["region"], "a": suv[0].tolist(), "b": suv[1].tolist(),
                         "ra": float(r[0]), "rb": float(r[1]), "depth": float(sd.mean())})
        for k in range(2):
            extent.append([suv[k, 0] - r[k], suv[k, 1] - r[k], suv[k, 0] + r[k], suv[k, 1] + r[k]])
    baton_out = None
    if baton:
        a, b = batonmod.segment(sk, fk, baton, baton["mode"])
        buv, bd = cam.project(np.stack([a, b]))
        baton_out = {"hand": baton["hand"], "mode": baton["mode"], "radius_cm": baton["radius_cm"], "uv": [_uv(buv[0]), _uv(buv[1])],
                     "depth": [float(bd[0]), float(bd[1])], "closure": baton.get("closure"), "thumb_closure": baton.get("thumb_closure"),
                     "collisions": batonmod.body_collisions(sk, fk, baton, baton["mode"])}
        if np.all(bd > 1.0) and np.all(np.isfinite(buv)):
            r = baton["radius_cm"] * cam.fx / bd * _px_scale(cam, np.stack([a, b]))
            for k in range(2):
                extent.append([buv[k, 0] - r[k], buv[k, 1] - r[k], buv[k, 0] + r[k], buv[k, 1] + r[k]])
    bbox = None
    if extent:
        e = np.array(extent)
        bbox = [float(e[:, 0].min()), float(e[:, 1].min()), float(e[:, 2].max()), float(e[:, 3].max())]

    coco = keypoints["coco17"]
    in_frame = [bool(k["uv"] and k["depth"] > 0 and cam.in_image(np.array(k["uv"]))) for k in coco]
    errors = sk.pose_errors(pose)
    collisions = sk.collisions(sk.fk(pose))
    v = {**DEFAULT_VALIDATION, **(validation or {})}     # same rules as the generator (the bbox may leave the frame)
    vis = bbox and max(0.0, min(bbox[2], cam.width) - max(bbox[0], 0)) > 0 and max(0.0, min(bbox[3], cam.height) - max(bbox[1], 0)) > 0
    checks = {
        "subject_in_front": bool(segments),
        "enough_keypoints_in_frame": bool(in_frame) and sum(in_frame) / len(in_frame) >= v["min_keypoints_in_frame_frac"] and bool(vis),
        "bbox_height_ok": bool(bbox and bbox[3] - bbox[1] >= v["min_bbox_height_px"]),
        "bbox_edge_ok": bool(bbox and (v["max_bbox_edge_touch"] or (bbox[0] > 0 and bbox[1] > 0 and bbox[2] < cam.width and bbox[3] < cam.height))),
    }
    az = float(cam_params.get("azimuth_deg", 0.0))
    return {
        "officer_z": float(officer.get("z", 0.0)), "camera": camera_summary(cam), "joints": joints, "dof_axes": dof_axes(sk, pose, fk), "keypoints": keypoints, "segments": segments,
        "bbox": bbox, "bbox_height_px": (bbox[3] - bbox[1]) if bbox else 0.0, "kp_in_frame": in_frame,
        "px_per_cm_at_officer": float(cam.fx / max(cam.to_camera(pts[names.index("pelvis")])[0, 0], 1e-3)),
        "limit_errors": errors, "collisions": collisions, "checks": checks, "baton": baton_out,
        "generator_would_accept_view": all(checks.values()), "approach_side": approach_side(az),
    }


# --------------------------------------------------------------------------- gestures
def match_gestures(sk: Skeleton, gestures: dict[str, dict], pose: Pose, az_deg: float) -> list[dict]:
    """Every gesture variant whose constraints the current pose satisfies (physical validity checked separately)."""
    fk = sk.fk(pose)
    side = approach_side(az_deg)
    out = []
    for gid, g in gestures.items():
        best = None
        for v in g["pose_variants"]:
            fails = constraint_failures(g.get("constraints", []) + v.get("constraints", []), sk, pose, fk)
            if best is None or len(fails) < len(best[1]):
                best = (v["id"], fails)
        ok = best is not None and not best[1]
        out.append({"gesture": gid, "label": g["label"], "variant": best[0] if best else None, "ok": ok,
                    "failures": best[1] if best else [], "signalling": g.get("signalling", True),
                    "instruction_for_side": g["instructions"][side] if ok else None})
    out.sort(key=lambda m: (not m["ok"], len(m["failures"])))
    return out


# --------------------------------------------------------------------------- drag IK
def _dof_cost(joint_name: str, dragging_finger: bool) -> float:
    cost = next((c for prefix, c in _BASE_COST if joint_name.startswith(prefix)), 1.0)
    if dragging_finger and not any(f in joint_name for f in ("thumb", "index", "middle", "ring", "pinky")):
        cost *= 6.0  # moving a fingertip should curl the finger, not swing the arm
    return cost


def _chain(sk: Skeleton, joint: str) -> list[str]:
    out, j = [], joint
    while j is not None:
        if sk.joints[j].dofs:
            out.append(j)
        j = sk.joints[j].parent
    return out[::-1]  # root -> tip


# --------------------------------------------------------------------------- rotating one joint by hand (the Alt+click lever)
def _wrist_limit(j, dof: str, other_value: float) -> tuple[float, float]:
    """The wrist moves in an ellipse, not a rectangle: the more it is flexed, the less it can deviate (and the other way round).
    Returns the allowed [lo, hi] of `dof` given the current value of the other wrist DOF."""
    d, o = j.dof(dof), next(x for x in j.dofs if x["name"] != dof)
    ov = other_value / (o["max"] if other_value >= 0 else o["min"]) if (o["max"] if other_value >= 0 else o["min"]) else 0.0
    k = math.sqrt(max(0.0, 1.0 - min(1.0, abs(ov)) ** 2))
    return d["min"] * k, d["max"] * k


def set_joint_angle(sk: Skeleton, pose: Pose, joint: str, dof: str, value: float) -> tuple[Pose, float, str | None]:
    """Turn one DOF of one joint to `value` degrees, as far as physics allows. Rules, in order:
      1. the joint's own limits (skeleton.json),
      2. the wrist's elliptical range (flexion and deviation limit each other),
      3. no NEW self-collision: the body parts may not run into each other (collisions already present are tolerated, so a pose that
         is already tangled can still be untangled). The angle stops at the last collision-free value on the way to the target.
    Returns (new pose, the angle actually used, why it was limited: None / "joint limit" / "wrist range" / "collision a~b")."""
    j = sk.joints.get(joint)
    d = j.dof(dof) if j else None
    if d is None:
        raise ValueError(f"unknown joint/dof {joint}.{dof}")
    cur = float(pose.get(joint, {}).get(dof, 0.0))
    value = float(value)
    lo, hi, why = d["min"], d["max"], "joint limit"
    if joint.startswith("hand_") and len(j.dofs) == 2:
        other = next(x for x in j.dofs if x["name"] != dof)["name"]
        lo, hi = _wrist_limit(j, dof, float(pose.get(joint, {}).get(other, 0.0)))
        why = "wrist range"
    target = min(max(value, lo), hi)
    limited = why if abs(target - value) > 1e-9 else None
    base = set(sk.collisions(sk.fk(pose)))

    def with_value(v: float) -> Pose:
        out = {k: dict(x) for k, x in pose.items()}
        out.setdefault(joint, {})[dof] = float(v)
        return out

    def new_hits(v: float) -> list[str]:
        return [h for h in sk.collisions(sk.fk(with_value(v))) if h not in base and h.split(" dist")[0] not in {b.split(" dist")[0] for b in base}]

    hits = new_hits(target)
    if hits:
        a, b = min(max(cur, lo), hi), target                # a is collision-free (it is where the joint already is)
        for _ in range(24):
            mid = 0.5 * (a + b)
            if new_hits(mid):
                b = mid
            else:
                a = mid
        target, limited = a, "collision " + hits[0].split(" dist")[0]
    return with_value(target), float(target), limited


LOCK_WEIGHT = 8.0


def solve_drag(sk: Skeleton, pose: Pose, joint: str, offset, target_char, iters: int = 60,
               tol_cm: float = 0.05, lock_axis: int | None = None) -> tuple[Pose, float]:
    """Move point (joint, offset) towards `target_char` (character frame, cm) by changing the DOFs of
    its ancestors. Weighted damped-least-squares with joint limits (active-set). `lock_axis` (0/1/2 = character x/y/z, Studio's
    Ctrl+drag): an error along that axis weighs LOCK_WEIGHT times more, so when the target cannot be reached exactly the point gives
    way within the plane, not towards / away from it. Returns (pose, residual cm)."""
    pose = copy.deepcopy(pose)
    offset = np.asarray(offset, float)
    target = np.asarray(target_char, float)
    chain = _chain(sk, joint)
    if not chain:
        return pose, float(np.linalg.norm(target - sk.fk(pose).point(joint, offset)))
    finger = any(f in joint for f in ("thumb", "index", "middle", "ring", "pinky"))
    dofs = [(jn, d) for jn in chain for d in sk.joints[jn].dofs]
    cost = np.array([_dof_cost(jn, finger) for jn, _ in dofs])
    lo = np.array([d["min"] for _, d in dofs])
    hi = np.array([d["max"] for _, d in dofs])
    lam = 0.6  # cm of damping; J columns are ~0.3-1 cm per degree

    for _ in range(iters):
        fk = sk.fk(pose)
        p = fk.point(joint, offset)
        err = target - p
        dist = float(np.linalg.norm(err))
        if dist < tol_cm:
            break
        err = err * min(1.0, 12.0 / dist)  # cap the step so the linearisation stays valid
        M = np.eye(3)
        if lock_axis is not None:
            M[lock_axis, lock_axis] = LOCK_WEIGHT
        cols = []
        for jn in chain:
            J = sk.joints[jn]
            Rp = fk.rotations[J.parent] if J.parent else np.eye(3)
            prefix = np.eye(3)
            for d in J.dofs:
                e = np.zeros(3)
                e[{"x": 0, "y": 1, "z": 2}[d["axis"]]] = 1.0
                axis_w = d["sign"] * (Rp @ prefix @ e)
                cols.append(np.cross(axis_w, p - fk.positions[jn]) * (math.pi / 180.0))  # cm per degree
                prefix = prefix @ axis_rotation(d["axis"], d["sign"] * pose.get(jn, {}).get(d["name"], 0.0))
        Jac = M @ np.stack(cols, axis=1)  # 3 x n (rows weighted: the locked axis counts LOCK_WEIGHT times)
        werr = M @ err

        cur = np.array([pose.get(jn, {}).get(d["name"], 0.0) for jn, d in dofs])
        active = np.ones(len(dofs), bool)
        step = np.zeros(len(dofs))
        for _pass in range(3):  # drop DOFs pinned at a limit that would be pushed further out
            Ja, Winv = Jac[:, active], 1.0 / cost[active]
            A = (Ja * Winv) @ Ja.T + (lam ** 2) * np.eye(3)
            step = np.zeros(len(dofs))
            step[active] = Winv * (Ja.T @ np.linalg.solve(A, werr))
            pinned = ((cur <= lo + 1e-9) & (step < 0)) | ((cur >= hi - 1e-9) & (step > 0))
            if not (pinned & active).any():
                break
            active &= ~pinned
        step = np.clip(step, -15.0, 15.0)
        new = np.clip(cur + step, lo, hi)
        if np.max(np.abs(new - cur)) < 1e-4:
            break
        for (jn, d), v in zip(dofs, new):
            pose.setdefault(jn, {})[d["name"]] = float(v)
    final = float(np.linalg.norm(target - sk.fk(pose).point(joint, offset)))
    return pose, final


def camera_plane_target(cam: Camera, world_point, uv) -> np.ndarray:
    """World point under pixel `uv` at the same camera depth as `world_point` (drag in the screen plane)."""
    depth = float(cam.to_camera(np.asarray(world_point, float))[0, 0])
    uv = cam.undistort_uv(np.asarray(uv, float))                 # the pointer is in the distorted image
    d = np.array([1.0, (uv[0] - cam.cx) / cam.fx, -(uv[1] - cam.cy) / cam.fy]) * depth
    return np.asarray(cam.position) + cam.R @ d


def axis_plane_target(cam: Camera, uv, point, normal) -> np.ndarray | None:
    """World point under pixel `uv` on the plane through `point` with normal `normal` (drag that keeps one axis coordinate fixed);
    None when the ray runs along the plane or the plane is behind the camera."""
    uv = cam.undistort_uv(np.asarray(uv, float))
    d = cam.R @ np.array([1.0, (uv[0] - cam.cx) / cam.fx, -(uv[1] - cam.cy) / cam.fy])
    n = np.asarray(normal, float)
    den = float(d @ n)
    if abs(den) < 1e-3:
        return None
    s = float((np.asarray(point, float) - np.asarray(cam.position)) @ n) / den
    if s <= 0:
        return None
    return np.asarray(cam.position) + s * d


def ground_target(cam: Camera, uv, z: float = 0.0) -> np.ndarray | None:
    """Intersection of the pixel ray with the horizontal plane at height z (default: the ground; None if the ray does not go down to it)."""
    uv = cam.undistort_uv(np.asarray(uv, float))
    d = cam.R @ np.array([1.0, (uv[0] - cam.cx) / cam.fx, -(uv[1] - cam.cy) / cam.fy])
    if abs(d[2]) < 1e-6:
        return None
    s = (z - cam.position[2]) / d[2]
    if s <= 0:
        return None
    return np.asarray(cam.position) + s * d


def to_char_frame(officer: dict, p_world) -> np.ndarray:
    R, t = officer_transform(officer)
    return R.T @ (np.asarray(p_world, float) - t)
