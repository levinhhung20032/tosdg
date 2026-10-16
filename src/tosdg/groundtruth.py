"""Ground truth derived from simulator state (never estimated).

- 2D keypoints: 3D joint/socket positions -> camera projection; "occluded" when the line of sight from the camera to the
  point passes through one of the officer's body capsules (or the held baton) first (self_occlusion).
- Bounding box: tight extent of the rendered instance mask.
- Traffic-control semantics: the officer's instruction (gestures.json "instructions") for the side of the officer the
  camera is on, when the camera is a road user's.

Contents:
  constants: OCCLUSION_EPS_CM, HEAD_REGIONS
  keypoints
  self_occlusion — Which keypoints the officer's own body hides from the camera
  bbox_from_mask
  bbox_full — Unclipped extent of the officer's body capsules projected…
  baton_capsules — [(a, b, radius_cm, colour region)] of the held baton in…
  baton_info — Metadata of the baton prop
  traffic_control
"""
from __future__ import annotations

import numpy as np

from .gestures import approach_side, side_boundary_distance
from .skeleton import segment_distance
from .scene import Scene

OCCLUSION_EPS_CM = 1.0       # the line of sight stops this short of the point (a point on a surface is not hidden by it)
HEAD_REGIONS = ("head", "headwear")


def keypoints(scene: Scene, skeleton) -> dict:
    cam = scene.camera
    out2d, out3d = {}, {}
    for set_name in sorted(skeleton.keypoint_sets):
        names, pts = skeleton.keypoints(scene.fk_world, set_name)
        uv, depth = cam.project(pts)
        in_frame = (depth > 0) & cam.in_image(uv)                  # inside the frame and not in the lens' black border
        hidden = self_occlusion(scene, skeleton, set_name, pts)
        out2d[set_name] = {"names": names,
                           "uv": [[round(float(a), 3), round(float(b), 3)] if np.isfinite(a) else None for a, b in uv],
                           "depth_cm": [round(float(d), 3) for d in depth],
                           "in_frame": [bool(x) for x in in_frame],
                           "occluded": [bool(h and f) for h, f in zip(hidden, in_frame)]}
        out3d[set_name] = [[round(float(c), 4) for c in p] for p in pts]
    return {"keypoints_2d": out2d, "keypoints_3d_world": out3d,
            "visibility_model": "in_frame + self_occlusion (body collision capsules and held baton; other objects in the photo not modelled)"}


def self_occlusion(scene: Scene, skeleton, set_name: str, pts: np.ndarray) -> list[bool]:
    """For each keypoint of a set: does the officer's own body hide it from the camera? The line of sight from the camera
    centre to the point (stopping OCCLUSION_EPS_CM short) is tested against every body capsule and the held baton. A
    keypoint at a joint ignores the capsules that start or end at that joint, and any capsule it lies inside (a hip in the
    pelvis); a point on the surface (nose, eyes, ears: non-zero offset) is hidden by its own body part (for the head: head
    and kepi) when its outward normal turns away from the camera, and by the other capsules as above. From the
    scene state, the same for every renderer; the uniform's cloth is a little wider than the capsules."""
    fk = scene.fk_world
    c = np.asarray(scene.camera.position, dtype=float)
    caps = [(fk.positions[g["from"]], fk.positions[g["to"]], float(g["radius"]), g["from"], g["to"], g.get("region"))
            for g in skeleton.segments.values()]
    caps += [(a, b, float(r), None, None, "baton") for a, b, r, _ in baton_capsules(skeleton, scene)]
    out = []
    for k, p in zip(skeleton.keypoint_sets[set_name], pts):
        d = p - c
        L = float(np.linalg.norm(d))
        if L <= OCCLUSION_EPS_CM:
            out.append(False)
            continue
        end = p - d / L * OCCLUSION_EPS_CM
        on_joint = not any(k.get("offset", (0, 0, 0)))
        hit = False
        for a, b, r, ja, jb, region in caps:
            own = k["joint"] in (ja, jb) or (k["joint"] == "head" and region in HEAD_REGIONS)   # the kepi is part of the head
            if own and on_joint:
                continue
            if own:
                # a point on the surface of its own body part (nose, eyes, ears): seen only from its side of that part -
                # its outward normal (from the capsule axis to the point) must face the camera
                ab = b - a
                t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0.0, 1.0))
                if (p - (a + t * ab)) @ (c - p) <= 0.0:
                    hit = True
                    break
                continue
            if segment_distance(p, p, a, b) < r:
                continue                       # the point is inside this capsule (a hip in the pelvis): it cannot hide it
            if segment_distance(c, end, a, b) < r:
                hit = True
                break
        out.append(hit)
    return out


def bbox_from_mask(mask: np.ndarray) -> dict | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    H, W = mask.shape
    return {"xyxy": [x0, y0, x1, y1], "xywh": [x0, y0, x1 - x0, y1 - y0], "source": "instance_mask",
            "truncated": bool(x0 == 0 or y0 == 0 or x1 == W or y1 == H), "mask_area_px": int(len(xs))}


def bbox_full(scene: Scene, skeleton) -> list[float] | None:
    """Unclipped extent of the officer's body capsules projected through the camera (may lie partly outside the
    image). Complements the mask bbox, which only covers what is visible in the frame."""
    cam = scene.camera
    ext = []
    for seg in skeleton.segments.values():
        p = np.stack([scene.fk_world.positions[seg["from"]], scene.fk_world.positions[seg["to"]]])
        uv0, d = cam.project_pinhole(p)
        if np.any(d <= 1.0) or not np.all(np.isfinite(uv0)):
            continue
        r = seg["radius"] * cam.fx / d
        uv = uv0
        if cam.lens is not None and cam.lens.distorts:             # the lens moves the centres and magnifies the circles around them
            r = r * cam.lens.local_scale(uv0, cam.cx, cam.cy, cam.fx)
            uv = cam.lens.distort(uv0, cam.cx, cam.cy, cam.fx)
            if not np.all(np.isfinite(uv)):
                continue
        ext += [[uv[k, 0] - r[k], uv[k, 1] - r[k], uv[k, 0] + r[k], uv[k, 1] + r[k]] for k in range(2)]
    if not ext:
        return None
    e = np.array(ext)
    return [round(float(e[:, 0].min()), 2), round(float(e[:, 1].min()), 2), round(float(e[:, 2].max()), 2), round(float(e[:, 3].max()), 2)]


def baton_capsules(skeleton, scene: Scene) -> list:
    """[(a, b, radius_cm, colour region)] of the held baton in world coordinates (empty when no baton)."""
    if not scene.baton:
        return []
    from .baton import segment
    a, b = segment(skeleton, scene.fk_world, scene.baton, scene.baton["mode"])
    return [(a, b, scene.baton["radius_cm"], "leather")]


def baton_info(scene: Scene, skeleton) -> dict:
    """Metadata of the baton prop: its end points (3D world, and projected through the camera), always derived from the scene state."""
    if not scene.baton:
        return {"enabled": False}
    (a, b, r, _), = baton_capsules(skeleton, scene)
    uv, depth = scene.camera.project(np.stack([a, b]))
    cam = scene.camera
    inside = [bool(d > 0 and cam.in_image(p)) for p, d in zip(uv, depth)]
    sp = scene.baton
    return {"enabled": True, "hand": sp["hand"], "mode": sp["mode"], "length_cm": sp["length_cm"], "radius_cm": sp["radius_cm"],
            "handle_cm": sp["handle_cm"], "grip_closure": sp["closure"], "thumb_closure": sp["thumb_closure"],
            "handle_world": [round(float(x), 3) for x in a], "tip_world": [round(float(x), 3) for x in b],
            "handle_uv": [round(float(x), 3) for x in uv[0]], "tip_uv": [round(float(x), 3) for x in uv[1]], "in_frame": inside}


def traffic_control(scene: Scene, margin_deg: float) -> dict:
    """What the officer's gesture tells the camera: the side of the officer the camera is on, and for a road user's camera
    the instruction of the gesture for that side (None for an infrastructure camera)."""
    g = scene.gesture
    role = scene.camera_profile["observer_role"]
    az = scene.camera_placement["azimuth_deg"]
    side = approach_side(az)
    return {
        "observer_role": role,
        "observer_azimuth_deg": round(float(az), 4),
        "approach_side": side,
        "approach_side_ambiguous": bool(side_boundary_distance(az) < margin_deg),
        "instruction_for_observer": g["instructions"][side] if role == "road_user" else None,
    }
