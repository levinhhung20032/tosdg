"""The traffic podium (a round platform at a junction) as a static 3D prop of the mesh renderer. Studio preview only: the generator
does not use it.

The model is a textured GLB (profiles/<CC>/props/podium.glb, metres, glTF axes: Y up). It is merged into the officer mesh as extra
triangles skinned 100 % to the `root` joint, which never moves, so the podium stays put while the officer is posed. In the character frame the standing
surface is z = 0 and the body of the podium reaches below it; because the officer is placed on the podium (its transform lifts the
root joint by the podium depth), the podium's bottom ends up on the ground (world z = 0) and the officer stands on top.

Contents:
  constants: DIAMETER_CM, LEDGE_Y, RIM_TOL, DECK_Y
  prop_path
  _rebuild_rim — Replace the model's jagged rim with a clean ring
  load — The GLB in the character frame (cm, X forward, Y right, Z up
  sole_level — Height (cm, character frame) of the lowest point of the…
  lift_cm — How high the officer's root is above the ground when it…
  size — Dimensions for the Studio's simple podium drawing (plan /…
  mesh_with_podium — `mesh` plus the podium, rigidly attached to the `root`…
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

DIAMETER_CM = 84.0            # whole model scaled uniformly (30 % smaller than the first 120 cm version)
LEDGE_Y = 0.51                # height of the outer ledge in the GLB (metres): the rebuilt rim's outer wall reaches down to it
RIM_TOL = 0.5                 # cm above the standing surface: higher vertices belong to the rim (rebuilt, see _rebuild_rim)
DECK_Y = 0.555                # height of the standing surface in the GLB (metres): the flat top inside the rim (the 0.51 ledge is the outer edge)
_TEXTURE_PX = 1024
_CACHE: dict = {}


def prop_path(profile_dir) -> Path:
    return Path(profile_dir) / "props" / "podium.glb"


def load(path) -> dict:
    """The GLB in the character frame (cm, X forward, Y right, Z up; standing surface at z = 0, centred on the origin)."""
    from PIL import Image
    from .import_mesh import read_static_glb
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    g = read_static_glb(path)
    if g["texture"] is None:
        raise ValueError(f"{path.name}: the podium model needs a base-colour texture")
    p = g["pos"].astype(float)
    width = max(np.ptp(p[:, 0]), np.ptp(p[:, 2]))
    s = DIAMETER_CM / width
    cx, cz = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 2].min() + p[:, 2].max()) / 2
    pos = np.stack([-(p[:, 2] - cz), p[:, 0] - cx, p[:, 1] - DECK_Y], axis=1) * s     # glTF (x right, y up, -z forward) -> UE-like frame
    tris = g["tris"].astype(np.int32)
    centre = pos.mean(axis=0)                                                          # winding: counter-clockwise seen from outside
    n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    c = pos[tris].mean(axis=1)
    if float((((c - centre) * n).sum(axis=1) > 0).mean()) < 0.5:
        tris = tris[:, [0, 2, 1]]
        n = -n
    nrm = np.zeros_like(pos)                                                           # smooth normals: area-weighted face normals
    for k in range(3):
        np.add.at(nrm, tris[:, k], n)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    tex = np.asarray(Image.fromarray(g["texture"]).resize((_TEXTURE_PX, _TEXTURE_PX), Image.LANCZOS), np.uint8)
    pos, nrm, uv, tris = _rebuild_rim(pos, nrm, g["uv"].astype(np.float32), tris, tex, ledge_z=-(DECK_Y - LEDGE_Y) * s)
    out = {"pos": pos, "nrm": nrm, "uv": uv, "tris": tris, "texture": tex}
    _CACHE[key] = out
    return out


def _rebuild_rim(pos, nrm, uv, tris, tex, ledge_z: float, segments: int = 72):
    """The model's rim (everything above the standing surface) is a crown of jagged triangles; replace it with a clean ring of the
    same height (median of the rim vertices) and radius, coloured with the rim's own median texel. Nothing above z = RIM_TOL: the deck."""
    rim_v = pos[:, 2] > RIM_TOL
    if not rim_v.any():
        return pos, nrm, uv, tris
    h = float(np.median(pos[rim_v, 2]))
    r_all = np.hypot(pos[:, 0], pos[:, 1])
    r_in = float(r_all[rim_v].min())
    r_out = min(float(r_all[pos[:, 2] > ledge_z - 1.5].max()) + 0.3,   # wide enough to hide the old rim's base, which sticks out a little,
                DIAMETER_CM / 2 + 0.45)                               # but the podium keeps its stated diameter
    ledge_z -= 1.0
    th, tw = tex.shape[:2]                                                              # the rim's colour: the median-bright rim texel
    px = tex[np.clip((uv[rim_v, 1] * (th - 1)).astype(int), 0, th - 1), np.clip((uv[rim_v, 0] * (tw - 1)).astype(int), 0, tw - 1)]
    k = np.argsort(px.astype(float).sum(axis=1))[len(px) // 2]
    rim_uv = uv[np.nonzero(rim_v)[0][k]]
    keep = ~rim_v[tris].any(axis=1)
    tris = tris[keep]
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    c, s_ = np.cos(a), np.sin(a)
    ring = lambda rad, z: np.stack([rad * c, rad * s_, np.full(segments, z)], axis=1)
    radial = np.stack([c, s_, np.zeros(segments)], axis=1)
    up = np.tile([0.0, 0.0, 1.0], (segments, 1))
    # three bands, each with its own vertices so the edges stay sharp: inner wall, top, outer wall down to the outer ledge (it covers
    # the rest of the old rim's base, which also zig-zags)
    bands = [(ring(r_in, 0.0), ring(r_in, h), -radial), (ring(r_in, h), ring(r_out, h), up), (ring(r_out, h), ring(r_out, ledge_z), radial)]
    new_pos, new_nrm, new_tris, base = [], [], [], len(pos)
    for lo, hi, n in bands:
        new_pos += [lo, hi]
        new_nrm += [n, n]
        for i in range(segments):
            j = (i + 1) % segments
            a0, a1, b0, b1 = base + i, base + j, base + segments + i, base + segments + j
            for t in ((a0, a1, b1), (a0, b1, b0)):
                new_tris.append(t)
        base += 2 * segments
    pos2 = np.vstack([pos, *new_pos])
    nrm2 = np.vstack([nrm, *new_nrm])
    t2 = np.array(new_tris, np.int64)
    face = np.cross(pos2[t2[:, 1]] - pos2[t2[:, 0]], pos2[t2[:, 2]] - pos2[t2[:, 0]])
    flip = (face * nrm2[t2].mean(axis=1)).sum(axis=1) < 0                              # counter-clockwise seen from the side the normal points to
    t2[flip] = t2[flip][:, [0, 2, 1]]
    uv2 = np.vstack([uv, np.tile(rim_uv, (len(pos2) - len(pos), 1))]).astype(np.float32)
    tris2 = np.vstack([tris, t2])
    used = np.unique(tris2)                                                             # drop the old rim's vertices (size() reads the extent)
    remap = np.full(len(pos2), -1, np.int64)
    remap[used] = np.arange(len(used))
    return pos2[used], nrm2[used], uv2[used], remap[tris2].astype(np.int32)


def sole_level(mesh, sk) -> float:
    """Height (cm, character frame) of the lowest point of the officer mesh in the neutral standing pose. It is below 0 for a model
    whose shoes reach under the skeleton's root plane; the podium's standing surface has to be at this level or the shoes sink into it."""
    z = mesh.cache.get("sole_z")
    if z is None:
        from .render_mesh import skin_mesh
        z = mesh.cache["sole_z"] = float(skin_mesh(mesh, sk.fk({}))[0][:, 2].min())
    return z


def lift_cm(prop: dict, sole_z: float) -> float:
    """How high the officer's root is above the ground when it stands on the podium: the podium height plus the shoe depth below the root."""
    return float(-prop["pos"][:, 2].min()) - sole_z


def size(prop: dict) -> dict:
    """Dimensions for the Studio's simple podium drawing (plan / profile / camera outline): diameter, depth below the standing surface, rim height."""
    z = prop["pos"][:, 2]
    return {"diameter_cm": DIAMETER_CM, "depth_cm": round(float(-z.min()), 3), "rim_cm": round(float(max(z.max(), 0.0)), 3)}


def mesh_with_podium(mesh, prop: dict, sole_z: float = 0.0):
    """`mesh` plus the podium, rigidly attached to the `root` joint (which stays at the origin whatever the pose). The standing surface is
    put at `sole_z` (see sole_level) so the shoes rest on it."""
    rj = mesh.joint_names.index("root")
    Rb, pb = mesh.bind_rotations[rj], mesh.bind_positions[rj]
    pos = pb + (prop["pos"] + [0.0, 0.0, sole_z]) @ Rb.T
    nrm = prop["nrm"] @ Rb.T
    V0, nv = len(mesh.positions), len(pos)
    joints = np.zeros((nv, 4), np.int32)
    joints[:, 0] = rj
    weights = np.zeros((nv, 4), np.float32)
    weights[:, 0] = 1.0
    mat, tex = len(mesh.materials), len(mesh.textures)
    return replace(
        mesh,
        positions=np.vstack([mesh.positions, pos]), normals=np.vstack([mesh.normals, nrm]),
        joints=np.vstack([mesh.joints, joints]), weights=np.vstack([mesh.weights, weights]),
        triangles=np.vstack([mesh.triangles, prop["tris"] + V0]).astype(np.int32),
        instance_tris=len(mesh.triangles) if mesh.instance_tris is None else mesh.instance_tris,
        tri_material=np.concatenate([mesh.tri_material, np.full(len(prop["tris"]), mat, np.int32)]),
        materials=[*mesh.materials, {"name": "podium", "region": "leather", "jitter": False, "texture": tex}],
        uv=np.vstack([mesh.uv if mesh.uv is not None else np.zeros((V0, 2), np.float32), prop["uv"]]).astype(np.float32),
        textures=[*mesh.textures, prop["texture"]],
        vertex_jitter=np.concatenate([mesh.vertex_jitter if mesh.vertex_jitter is not None else np.zeros(V0, np.float32), np.zeros(nv, np.float32)]),
        cache={})
