"""The baton prop: held in one hand (scene option `baton`) or, when it is not held, hanging from the belt.

The baton is NOT part of the person model (the model must not carry one): the renderers add it to the model's mesh. Held, the hand takes
a grip pose around it and the baton is rigidly attached to the hand joint. Not held, it hangs from the belt, rigidly attached to the
pelvis, at the place the model's fit gives (`belt_baton` in profiles/<CC>/assets/<model>.fit.json); a model without that key shows no
baton. Either way it is one prop, so there is never more than one baton in a picture.

Grip modes (`pose variant["baton_axis"]`):
  "across" (default)  the baton crosses the fist, perpendicular to the fingers' run: arm forward + thumb up -> baton points straight up.
  "along"             the baton continues the forearm (arm raised straight up -> baton continues the arm). A fist cannot hold a baton that is
                      parallel to the fingers' run, so the baton crosses the palm diagonally (60 deg from "across") and the wrist deviates
                      just enough to line it up with the forearm.

The grip (all modes): wrist straight, a tight overhand fist (each finger as closed as the baton allows), the thumb resting on the radial
side of the fist near the base of the index finger.

Everything is derived from the pose (forward kinematics); nothing here uses randomness, so the same pose always gives the same baton.

Contents:
  constants: DEFAULT_BATON, FINGERS, MODES
  sign
  prop_path — Where a profile keeps its 3D baton model (optional)
  normalise — Fill defaults
  segment — End points (a = handle end, b = tip), in the frame of `fk`
  apply_grip — The pose with the baton hand gripping the baton
  body_collisions — Baton against every part of the body except the hand and…
  local_segment — The baton's end points in the hand joint's own frame…
  baton_mesh_parts — A capped cylinder in the hand frame
  mesh_with_baton — A copy of a skinned UniformMesh with the baton added,…
  belt_config / mesh_with_stowed_baton — The baton hanging from the belt (the model's fit says where)
  mesh_with_officer_baton — Held baton when the scene holds one, else the one on the belt
  load_prop — Read a baton GLB (a straight rod, optionally with a wrist…
  (+ 9 private helpers)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .skeleton import FKResult, Pose, Skeleton, segment_distance

DEFAULT_BATON = {"enabled": False, "hand": "r", "length_cm": 50.0, "radius_cm": 1.5, "handle_cm": 12.0}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MODES = ("across", "along")
# hand-local geometry (cm). Hand frame: fingers run along -z, +x is the thumb side, the palm faces -y (right) / +y (left).
_RING_CENTRES = {"across": (2.0, 1.5, -6.0), "along": (4.0, 1.5, -5.0)}   # where the baton passes through the fist: (x, palm-side depth, z)

_MODE_ANGLE = {"across": 0.0, "along": 60.0}     # tilt of the baton from "across the fist" towards the forearm direction, in the palm plane
# ----------------------------------------------------------------------------- the grip: a tight overhand fist
# Wrist straight (hand in line with the forearm; for "along" it deviates just enough to bring the baton in line with the forearm),
# the four fingers wrapped as tightly as the baton allows, the thumb resting on the radial side of the fist near the base of the index.
_FINGER_NAMES = ("index", "middle", "ring", "pinky")
_GRIP_CACHE: dict = {}
# ----------------------------------------------------------------------------- the 3D baton model (profiles/<CC>/props/baton.glb)
_PROP_TEXTURE_PX = 512          # the renderer samples the colour texture only; 4k source images are far more than a 320..1920 px view needs
_PROPS: dict = {}




def sign(side: str) -> float:
    return 1.0 if side == "r" else -1.0


def prop_path(profile_dir) -> Path:
    """Where a profile keeps its 3D baton model (optional)."""
    return Path(profile_dir) / "props" / "baton.glb"


def normalise(spec: dict | None) -> dict | None:
    """Fill defaults; None when no baton is held."""
    if not spec or not spec.get("enabled"):
        return None
    out = {**DEFAULT_BATON, **spec}
    if out["hand"] not in ("l", "r"):
        raise ValueError(f"baton hand must be 'l' or 'r', got {out['hand']!r}")
    for k in ("length_cm", "radius_cm", "handle_cm"):
        if not isinstance(out[k], (int, float)) or out[k] <= 0:
            raise ValueError(f"baton {k} must be a positive number")
    if out["handle_cm"] >= out["length_cm"]:
        raise ValueError("baton handle_cm must be shorter than length_cm")
    return out


def _axis_local(mode: str) -> np.ndarray:
    """Baton axis in the hand frame, from the handle end towards the tip: +x (thumb side) tilted towards the fingers' direction (-z)."""
    if mode not in MODES:
        raise ValueError(f"unknown baton_axis {mode!r}")
    phi = np.radians(_MODE_ANGLE[mode])
    return np.array([np.cos(phi), 0.0, -np.sin(phi)])


def segment(sk: Skeleton | None, fk: FKResult, spec: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """End points (a = handle end, b = tip), in the frame of `fk`."""
    side = spec["hand"]
    s = sign(side)
    R, h = fk.rotations[f"hand_{side}"], fk.positions[f"hand_{side}"]
    ax = _axis_local(mode)
    cx, cy, cz = _RING_CENTRES[mode]
    c = np.array([cx, -s * cy, cz])
    a = c - ax * spec["handle_cm"]
    b = c + ax * (spec["length_cm"] - spec["handle_cm"])
    return h + R @ a, h + R @ b


def _finger_joints(sk: Skeleton, side: str, f: str) -> list[str]:
    return list(sk.fingers[side][f])


def _penetration(sk: Skeleton, fk: FKResult, a, b, radius: float, side: str, fingers: tuple, tol: float) -> float:
    """Worst (limit - distance) between the baton and the segments of the given fingers; <= 0 = none touches it."""
    worst = -1e9
    names = {sk.fingers[side][f][i] for f in fingers for i in range(3)}
    for seg in sk.segments.values():
        if seg["region"] == "finger" and seg["from"] in names:
            d = segment_distance(a, b, fk.positions[seg["from"]], fk.positions[seg["to"]])
            worst = max(worst, tol * (radius + seg["radius"]) - d)
    return worst


def _hand_local(fk: FKResult, side: str, name: str) -> np.ndarray:
    return fk.rotations[f"hand_{side}"].T @ (fk.positions[name] - fk.positions[f"hand_{side}"])


def _solve_grip(sk: Skeleton, spec: dict, mode: str, fist: Pose) -> tuple:
    side = spec["hand"]
    radius = spec["radius_cm"]
    # 1. wrist: straight; "along": the deviation that brings the baton closest to the forearm direction
    dev = 0.0
    if mode == "along":
        best = None
        d0 = sk.joints[f"hand_{side}"].dof("deviation")
        for d in np.arange(d0["min"], d0["max"] + 0.5, 1.0):
            fk = sk.fk({f"hand_{side}": {"deviation": float(d)}})
            a, b = segment(sk, fk, spec, mode)
            fore = fk.positions[f"hand_{side}"] - fk.positions[f"lowerarm_{side}"]
            cosang = float((b - a) @ fore / np.linalg.norm(b - a) / np.linalg.norm(fore))
            if best is None or cosang > best[0]:
                best = (cosang, float(d))
        dev = best[1]
    pose: Pose = {f"hand_{side}": {"flexion": 0.0, "deviation": dev}}
    # 2. four fingers: each as closed as it can be (towards the fist) without entering the baton: tight contact
    cs = {}
    for f in _FINGER_NAMES:
        chosen = 0.0
        for c in np.linspace(1.0, 0.0, 41):
            cand = {**pose, **{j: {"flexion": fist[j]["flexion"] * c} for j in _finger_joints(sk, side, f) if j in fist}}
            fk = sk.fk(cand)
            a, b = segment(sk, fk, spec, mode)
            if _penetration(sk, fk, a, b, radius, side, (f,), 0.95) <= 0:
                chosen = float(c)
                break
        cs[f] = chosen
        pose.update({j: {"flexion": round(fist[j]["flexion"] * chosen, 4)} for j in _finger_joints(sk, side, f) if j in fist})
    # 3. thumb: lies on the radial (thumb) side of the fist next to the index finger, near its base, clear of the baton and the fingers
    pose.update(_solve_thumb(sk, spec, mode, pose))
    return pose, {"closure": round(float(np.mean(list(cs.values()))), 3), "thumb_closure": 1.0, "finger_closure": {k: round(v, 3) for k, v in cs.items()}}


def _solve_thumb(sk: Skeleton, spec: dict, mode: str, pose: Pose) -> Pose:
    import itertools
    side = spec["hand"]
    s = sign(side)
    tj = [(f"thumb_01_{side}", "flexion"), (f"thumb_01_{side}", "abduction"), (f"thumb_02_{side}", "flexion"), (f"thumb_03_{side}", "flexion")]
    lim = [(sk.joints[j].dof(d)["min"], sk.joints[j].dof(d)["max"]) for j, d in tj]
    fk0 = sk.fk(pose)
    target = _hand_local(fk0, side, f"index_02_{side}") + np.array([2.2, -s * 0.4, 0.6])
    a, b = segment(sk, fk0, spec, mode)
    thumb_segs = [g for g in sk.segments.values() if g["region"] == "finger" and g["name"].startswith("thumb") and g["from"].endswith(f"_{side}")]
    other = [g for g in sk.segments.values() if g["region"] == "finger" and not g["name"].startswith("thumb") and g["from"].endswith(f"_{side}")]

    def build(x):
        p = dict(pose)
        for (j, d), v in zip(tj, x):
            p[j] = {**p.get(j, {}), d: float(v)}
        return p

    def cost(x):
        fk = sk.fk(build(x))
        tip = _hand_local(fk, side, f"thumb_tip_{side}")
        c = float(np.sum((tip - target) ** 2))
        for g in thumb_segs:
            d = segment_distance(a, b, fk.positions[g["from"]], fk.positions[g["to"]])
            if g["name"].startswith("thumb_1"):              # the thumb's metacarpal sits inside the palm's thenar mass: the baton may cross it
                d = 1e9
            c += 600.0 * max(0.0, 0.95 * (spec["radius_cm"] + g["radius"]) - d) ** 2
            for o in other:
                d2 = segment_distance(fk.positions[g["from"]], fk.positions[g["to"]], fk.positions[o["from"]], fk.positions[o["to"]])
                c += 300.0 * max(0.0, 0.9 * (g["radius"] + o["radius"]) - d2) ** 2
        return c

    grid = [np.array(v, float) for v in itertools.product(*[np.linspace(lo, hi, 4) for lo, hi in lim])]
    x = min(grid, key=cost)
    cx, step = cost(x), np.array([8.0, 8.0, 8.0, 8.0])
    for _ in range(200):
        improved = False
        for k in range(4):
            for sg in (-1.0, 1.0):
                y = x.copy()
                y[k] = np.clip(y[k] + sg * step[k], *lim[k])
                cy = cost(y)
                if cy < cx - 1e-9:
                    x, cx, improved = y, cy, True
        if not improved:
            step *= 0.5
            if step.max() < 0.1:
                break
    out: Pose = {}
    for (j, d), v in zip(tj, x):
        out.setdefault(j, {})[d] = round(float(v), 3)
    return out


def apply_grip(sk: Skeleton, pose: Pose, spec: dict, mode: str, fist: Pose, keep_wrist: bool = False) -> tuple:
    """The pose with the baton hand gripping the baton: wrist straight, fingers wrapped around it, thumb on the radial side of the fist.
    The grip depends only on the hand (its own frame), never on the body pose; it is solved once and cached. `fist` = the library
    hand pose `fist` already named for this side. `keep_wrist` (Studio): the wrist keeps the pose's own flexion and deviation (the
    'along' grip's deviation added), so a wrist turned by hand stays turned; the generator straightens it. Returns (pose, info)."""
    side = spec["hand"]
    key = (id(sk), side, mode, spec["radius_cm"], spec["length_cm"], spec["handle_cm"],
           tuple(sorted((j, d["flexion"]) for j, d in fist.items() if "flexion" in d)))
    if key not in _GRIP_CACHE:
        _GRIP_CACHE[key] = _solve_grip(sk, spec, mode, fist)
    grip, info = _GRIP_CACHE[key]
    out = {j: dict(d) for j, d in pose.items()}
    own = dict(pose.get(f"hand_{side}", {}))
    for j, d in grip.items():
        out.setdefault(j, {}).update(d)
    if keep_wrist:
        hj = sk.joints[f"hand_{side}"]
        w = out[f"hand_{side}"]
        for dof, extra in (("flexion", 0.0), ("deviation", float(grip[f"hand_{side}"].get("deviation", 0.0)))):
            d = hj.dof(dof)
            w[dof] = float(min(max(float(own.get(dof, 0.0)) + extra, d["min"]), d["max"]))
    return {j: {k: round(v, 4) for k, v in d.items()} for j, d in out.items()}, {"hand": side, "mode": mode, **info}


def body_collisions(sk: Skeleton, fk: FKResult, spec: dict, mode: str) -> list[str]:
    """Baton against every part of the body except the hand and forearm that hold it."""
    side = spec["hand"]
    a, b = segment(sk, fk, spec, mode)
    hits = []
    for seg in sk.segments.values():
        own = seg["name"].endswith(f"_{side}") and seg["region"] in ("finger", "hand", "forearm")
        if own:
            continue
        d = segment_distance(a, b, fk.positions[seg["from"]], fk.positions[seg["to"]])
        limit = sk.collision_tolerance * (spec["radius_cm"] + seg["radius"])
        if d < limit:
            hits.append(f"baton~{seg['name']} dist={d:.1f}cm < {limit:.1f}cm")
    return hits


def local_segment(spec: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """The baton's end points in the hand joint's own frame (rigid with the hand, whatever the pose)."""
    eye = FKResult({f"hand_{spec['hand']}": np.zeros(3)}, {f"hand_{spec['hand']}": np.eye(3)})
    return segment(None, eye, spec, mode)


def baton_mesh_parts(spec: dict, mode: str, sides: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A capped cylinder in the hand frame: (positions (V,3), normals (V,3), triangles (T,3), counter-clockwise from outside)."""
    a, b = local_segment(spec, mode)
    return cylinder_parts(a, b, spec["radius_cm"], sides)


def cylinder_parts(a: np.ndarray, b: np.ndarray, r: float, sides: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A capped cylinder from a to b of radius r: (positions (V,3), normals (V,3), triangles (T,3), counter-clockwise from outside)."""
    ax = (b - a) / np.linalg.norm(b - a)
    u = np.cross(ax, [0.0, 1.0, 0.0] if abs(ax[1]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v          # (n, 3) outward directions
    pos = np.vstack([a + r * ring, b + r * ring, a[None], b[None]])
    nrm = np.vstack([ring, ring, -ax[None], ax[None]])
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris += [[i, j, sides + j], [i, sides + j, sides + i]]            # side wall
        tris += [[2 * sides, j, i], [2 * sides + 1, sides + i, sides + j]]   # end caps
    tris = np.array(tris, np.int32)
    # make every triangle face outwards (cheap and robust): flip those whose normal points to the axis
    c = (pos[tris[:, 0]] + pos[tris[:, 1]] + pos[tris[:, 2]]) / 3
    n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    mid = (a + b) / 2
    flip = ((c - mid) * n).sum(axis=1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pos, nrm, tris


def mesh_with_baton(mesh, spec: dict, mode: str, region: str = "leather", prop: dict | None = None):
    """A copy of a skinned UniformMesh with the baton added, rigid on the hand joint (the baton keeps its place in the hand whatever
    the pose). With a `prop` (load_prop) the baton is that textured 3D model; without, a plain cylinder whose colour comes from the
    uniform's `region` colour."""
    from dataclasses import replace
    if prop is not None:
        return _mesh_with_prop(mesh, spec, mode, prop)
    pos_l, nrm_l, tris = baton_mesh_parts(spec, mode)
    hj = mesh.joint_names.index(f"hand_{spec['hand']}")
    Rb, pb = mesh.bind_rotations[hj], mesh.bind_positions[hj]
    pos = pb + pos_l @ Rb.T                       # the mesh's vertices live in the bind pose
    nrm = nrm_l @ Rb.T
    V0, n = len(mesh.positions), len(pos)
    joints = np.zeros((n, 4), np.int32)
    joints[:, 0] = hj
    weights = np.zeros((n, 4), np.float32)
    weights[:, 0] = 1.0
    mat = len(mesh.materials)
    return replace(
        mesh,
        positions=np.vstack([mesh.positions, pos]), normals=np.vstack([mesh.normals, nrm]),
        joints=np.vstack([mesh.joints, joints]), weights=np.vstack([mesh.weights, weights]),
        triangles=np.vstack([mesh.triangles, tris + V0]).astype(np.int32),
        tri_material=np.concatenate([mesh.tri_material, np.full(len(tris), mat, np.int32)]),
        materials=[*mesh.materials, {"name": "baton", "region": region, "jitter": False}],
        uv=None if mesh.uv is None else np.vstack([mesh.uv, np.zeros((n, 2), np.float32)]),
        vertex_jitter=None if mesh.vertex_jitter is None else np.concatenate([mesh.vertex_jitter, np.zeros(n, np.float32)]),
        cache={})


# ----------------------------------------------------------------------------- the baton on the belt
DEFAULT_BELT = {"joint": "pelvis", "direction": [0.0, 0.0, -1.0], "length_cm": 50.0, "radius_cm": 1.5}


def belt_config(mesh) -> dict | None:
    """The model's `belt_baton` fit settings with defaults, or None when the model's fit has none (no baton on the belt).
    handle_cm = where the handle end (the strap loop) hangs from the belt, in the character frame at the bind pose (cm: x forward,
    y right, z up); direction = from the handle towards the tip (hanging: straight down); length_cm, radius_cm; joint it is rigid with."""
    cfg = (mesh.info.get("fit") or {}).get("belt_baton")
    if not cfg:
        return None
    out = {**DEFAULT_BELT, **cfg}
    if "handle_cm" not in out or len(out["handle_cm"]) != 3:
        raise ValueError("fit belt_baton needs handle_cm: [x, y, z]")
    if out["joint"] not in mesh.joint_names:
        raise ValueError(f"fit belt_baton joint '{out['joint']}' is not a joint of the model")
    return out


def mesh_with_stowed_baton(mesh, prop: dict | None = None):
    """A copy of `mesh` with the baton hanging from the belt (rigid with the belt's joint, so it moves with the body but does not
    swing); the mesh itself when the model's fit has no `belt_baton`. With a `prop` it is the textured 3D baton, else a plain cylinder."""
    cfg = belt_config(mesh)
    if cfg is None:
        return mesh
    H = np.asarray(cfg["handle_cm"], float)
    f1 = _unit(np.asarray(cfg["direction"], float))
    if prop is None:
        pos, nrm, tris = cylinder_parts(H, H + f1 * cfg["length_cm"], cfg["radius_cm"])
        return _append_triangles(mesh, pos, nrm, tris, cfg["joint"], None, {"name": "baton", "region": "leather", "jitter": False})
    toward_body = np.array([-H[0], -H[1], 0.0])                  # the strap loop turned towards the body
    f2 = toward_body - (toward_body @ f1) * f1
    f2 = _unit(f2) if np.linalg.norm(f2) > 1e-6 else _unit(np.cross(f1, [0.0, 1.0, 0.0]))
    pos, nrm = _place_prop(prop, H, f1, f2, cfg["length_cm"])
    return _append_prop(mesh, pos, nrm, prop, cfg["joint"], H, f1)


def mesh_with_officer_baton(mesh, spec: dict | None, mode: str, prop: dict | None = None):
    """The baton as the scene has it: in the hand when `spec` (a normalised baton spec) is given, otherwise hanging from the belt."""
    if spec:
        return mesh_with_baton(mesh, spec, mode, prop=prop)
    return mesh_with_stowed_baton(mesh, prop)


def load_prop(path) -> dict:
    """Read a baton GLB (a straight rod, optionally with a wrist strap) and describe it in its own axis frame.
    The rod axis comes from the vertices (principal axis); the handle end is the thicker end (grip, strap) and the strap, if any, is the
    lateral direction of the vertices farthest from the axis. Returns positions relative to the handle end, unit = the file's own."""
    from pathlib import Path as _P
    from PIL import Image
    from .import_mesh import read_static_glb
    path = _P(path)
    key = (str(path), path.stat().st_mtime_ns)
    if key in _PROPS:
        return _PROPS[key]
    g = read_static_glb(path)
    if g["texture"] is None:
        raise ValueError(f"{path.name}: the baton model needs a base-colour texture")
    p = g["pos"].astype(float)
    c = p.mean(axis=0)
    ax = np.linalg.svd(p - c, full_matrices=False)[2][0]
    t = (p - c) @ ax
    r = np.linalg.norm((p - c) - np.outer(t, ax), axis=1)
    n = max(1, int(0.15 * len(p)))
    order = np.argsort(t)
    if r[order[-n:]].max() > r[order[:n]].max():       # the thicker end is the handle: make it the low-t end (axis = handle -> tip)
        ax = -ax
    t = (p - c) @ ax
    rod_len = float(t.max() - t.min())
    handle = c + ax * t.min()
    far = np.argmax(r)
    lat = (p[far] - c) - ((p[far] - c) @ ax) * ax
    lat = lat / max(np.linalg.norm(lat), 1e-9)         # towards the strap (or any lateral direction for a plain rod)
    tris = g["tris"].astype(np.int32)
    tex = np.asarray(Image.fromarray(g["texture"]).resize((_PROP_TEXTURE_PX, _PROP_TEXTURE_PX), Image.LANCZOS), np.uint8)
    out = {"pos": p - handle, "uv": g["uv"], "tris": tris, "texture": tex, "axis": ax, "lateral": lat, "length": rod_len,
           "radius": float(np.median(r)), "source": path.name}
    _PROPS[key] = out
    return out


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _place_prop(prop: dict, handle: np.ndarray, f1: np.ndarray, f2: np.ndarray, length_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """The prop's vertices and normals with its handle end at `handle`, its rod along f1 (scaled to `length_cm`) and its strap (lateral
    direction) towards f2, all in one common frame."""
    e1, e2 = prop["axis"], prop["lateral"]
    e2 = _unit(e2 - (e2 @ e1) * e1)
    E = np.stack([e1, e2, np.cross(e1, e2)], axis=1)
    F = np.stack([f1, f2, np.cross(f1, f2)], axis=1)
    M = F @ E.T                                          # a proper rotation: the prop's frame onto the target frame
    pos = handle + (length_cm / prop["length"]) * (prop["pos"] @ M.T)
    tris = prop["tris"]
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, tris[:, k], fn)
    return pos, nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)


def _prop_in_hand(spec: dict, mode: str, prop: dict) -> tuple[np.ndarray, np.ndarray]:
    """The prop's vertices and normals in the hand frame: handle end at the baton's handle point, tip at its tip, scaled so the rod is
    `length_cm` long; the strap (lateral direction) turned towards the wrist."""
    a, b = local_segment(spec, mode)
    f1 = _unit(b - a)
    toward_wrist = -a - ((-a) @ f1) * f1                 # from the handle end to the hand origin, perpendicular to the baton
    if np.linalg.norm(toward_wrist) < 1e-6:
        toward_wrist = np.array([0.0, 0.0, 1.0]) - f1[2] * f1
    return _place_prop(prop, a, f1, _unit(toward_wrist), spec["length_cm"])


def _append_triangles(mesh, pos_bind, nrm_bind, tris, joint: str, tex_and_uv, material: dict):
    """`mesh` plus extra triangles (positions/normals already in the mesh's bind-pose space), rigid on `joint`; `tex_and_uv` =
    (texture, uv) for a textured material, else None."""
    from dataclasses import replace
    hj = mesh.joint_names.index(joint)
    V0, nv = len(mesh.positions), len(pos_bind)
    joints = np.zeros((nv, 4), np.int32)
    joints[:, 0] = hj
    weights = np.zeros((nv, 4), np.float32)
    weights[:, 0] = 1.0
    mat = len(mesh.materials)
    if tex_and_uv is not None:
        tex, uv = tex_and_uv
        material = {**material, "texture": len(mesh.textures)}
        extra = {"textures": [*mesh.textures, tex],
                 "uv": np.vstack([mesh.uv if mesh.uv is not None else np.zeros((V0, 2), np.float32), uv]).astype(np.float32)}
    else:
        extra = {"uv": None if mesh.uv is None else np.vstack([mesh.uv, np.zeros((nv, 2), np.float32)])}
    return replace(
        mesh,
        positions=np.vstack([mesh.positions, pos_bind]), normals=np.vstack([mesh.normals, nrm_bind]),
        joints=np.vstack([mesh.joints, joints]), weights=np.vstack([mesh.weights, weights]),
        triangles=np.vstack([mesh.triangles, tris + V0]).astype(np.int32),
        tri_material=np.concatenate([mesh.tri_material, np.full(len(tris), mat, np.int32)]),
        materials=[*mesh.materials, material],
        vertex_jitter=None if mesh.vertex_jitter is None and tex_and_uv is None else np.concatenate(
            [mesh.vertex_jitter if mesh.vertex_jitter is not None else np.zeros(V0, np.float32), np.zeros(nv, np.float32)]),
        cache={}, **extra)


def _append_prop(mesh, pos_bind, nrm_bind, prop: dict, joint: str, handle: np.ndarray, f1: np.ndarray):
    """The textured prop (placed by _place_prop, in the mesh's bind-pose space) added rigid on `joint`, wound counter-clockwise seen
    from outside."""
    tris = prop["tris"]
    c = (pos_bind[tris[:, 0]] + pos_bind[tris[:, 1]] + pos_bind[tris[:, 2]]) / 3
    n = np.cross(pos_bind[tris[:, 1]] - pos_bind[tris[:, 0]], pos_bind[tris[:, 2]] - pos_bind[tris[:, 0]])
    radial = (c - handle) - np.outer((c - handle) @ f1, f1)
    if float(((radial * n).sum(axis=1) > 0).mean()) < 0.5:
        tris = tris[:, [0, 2, 1]]
        nrm_bind = -nrm_bind
    return _append_triangles(mesh, pos_bind, nrm_bind, tris, joint, (prop["texture"], prop["uv"]),
                             {"name": "baton", "region": "leather", "jitter": False})


def _mesh_with_prop(mesh, spec: dict, mode: str, prop: dict):
    pos_l, nrm_l = _prop_in_hand(spec, mode, prop)
    a, b = local_segment(spec, mode)
    hj = mesh.joint_names.index(f"hand_{spec['hand']}")
    Rb, pb = mesh.bind_rotations[hj], mesh.bind_positions[hj]
    pos = pb + pos_l @ Rb.T                                  # the mesh's vertices live in the bind pose
    nrm = nrm_l @ Rb.T
    return _append_prop(mesh, pos, nrm, prop, f"hand_{spec['hand']}", pb + Rb @ a, _unit(Rb @ (b - a)))
