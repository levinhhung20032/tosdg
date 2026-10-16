"""Software renderer for the skinned uniform mesh (backend "mesh"). numpy + Pillow only.

Pipeline: linear-blend skinning from the posed skeleton (the same FK the ground truth uses) -> pinhole projection with
`Camera.project` -> exact z-buffer rasterisation -> deferred Lambert shading with interpolated normals -> composite over the
backplate. The instance mask is the exact coverage of the mesh, so the bbox ground truth is exact.

It draws real garment geometry (sleeves, kepi, belt, strap, shoes, fingers ...), but it is NOT a physically based or photoreal
renderer and NOT the Unreal backend: lighting is the same simple sun + ambient model as the proxy backend.

Contents:
  constants: SUPERSAMPLE
  skin_mesh — Linear blend skinning
  rasterize — Exact z-buffer
  material_albedo — (n_materials, 3) albedo in 0..1 from the uniform's colours
  render — Returns (rgb uint8 HxWx3, mask bool HxW)
  render_layer — The officer alone as an RGBA uint8 image (alpha =…
  (+ 3 private helpers)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .camera import Camera
from .skeleton import FKResult
from .uniform_mesh import UniformMesh

SUPERSAMPLE = 2
AO_RADIUS_CM = 6.0      # screen-space ambient occlusion: how far (cm, world) a surface can shade its neighbour
AO_SAMPLES = 0          # 0 = off: the CPU renderer stays fast; the GPU renderer (render_gl) has its own ambient occlusion
AO_STRENGTH = 0.75
_BIG = 64          # triangles whose bounding box is larger than this (px) are rasterised one by one


# A twist (upper arm) or pronation (forearm) turns the whole bone in the skeleton, but a real limb winds up along its length: the elbow
# end of the forearm hardly turns, the wrist end turns fully (UE5 rigs have twist bones for this; the importer folds them into the base
# bone). Skinning spreads the change of the roll from the bind pose along the bone: 0 at the joint, all of it at the child joint.
TWIST_CHILD = {"upperarm_l": "lowerarm_l", "upperarm_r": "lowerarm_r", "lowerarm_l": "hand_l", "lowerarm_r": "hand_r",
               "thigh_l": "calf_l", "thigh_r": "calf_r"}


def _twist_fractions(mesh: UniformMesh) -> dict[int, np.ndarray]:
    """joint index -> (V,) share of that joint's roll each vertex follows: smoothstep of its position along the bone (bind pose)."""
    if "twist_fractions" not in mesh.cache:
        out = {}
        for jn, child in TWIST_CHILD.items():
            if jn in mesh.joint_names and child in mesh.joint_names:
                i, c = mesh.joint_names.index(jn), mesh.joint_names.index(child)
                a, b = mesh.bind_positions[i], mesh.bind_positions[c]
                t = np.clip((mesh.positions - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-9), 0.0, 1.0)
                out[i] = t * t * (3.0 - 2.0 * t)
        mesh.cache["twist_fractions"] = out
    return mesh.cache["twist_fractions"]


def _untwist(mesh: UniformMesh, fk: FKResult, local: np.ndarray) -> np.ndarray:
    """Turn the joint-frame offsets (V, 4, 3) of the influences of rolling joints back by (1 - share) of the roll's change from the
    bind pose, about the joint's own z axis (the bone): the far end of the bone follows the roll fully, the near end not at all."""
    tw = fk.twists or {}
    bp = mesh.info.get("bind_pose", {}) or {}
    fr = _twist_fractions(mesh)
    out = None
    for i, f in fr.items():
        jn = mesh.joint_names[i]
        if jn not in tw:
            continue
        dof, sign, value = tw[jn]
        delta = sign * (value - float(bp.get(jn, {}).get(dof, 0.0)))     # change of the roll from the bind pose, signed about +z
        if abs(delta) < 1e-6:
            continue
        sel = mesh.joints == i                                            # (V, 4) influences of this joint
        if not sel.any():
            continue
        if out is None:
            out = np.array(local, dtype=float, copy=True)
        vi, ki = np.nonzero(sel)
        ang = np.radians((f[vi] - 1.0) * delta)
        c, s = np.cos(ang), np.sin(ang)
        x, y = out[vi, ki, 0].copy(), out[vi, ki, 1].copy()
        out[vi, ki, 0], out[vi, ki, 1] = c * x - s * y, s * x + c * y
    return local if out is None else out


def _blend_dirs(mesh: UniformMesh, fk: FKResult, vec: np.ndarray) -> np.ndarray:
    """Skin per-vertex directions (normals, tangents): the same blend as the positions, rotation only."""
    R = np.stack([fk.rotations[n] for n in mesh.joint_names])
    j, w = mesh.joints, mesh.weights.astype(float)
    v = np.broadcast_to(vec[:, None, :], (len(vec), 4, 3))
    if not mesh.bind_is_rest:                                            # into the joint's own frame, like the positions: R_bind^T v
        v = np.einsum("vkji,vkj->vki", mesh.bind_rotations[j], v)
    v = _untwist(mesh, fk, v)
    out = (w[..., None] * np.einsum("vkij,vkj->vki", R[j], v)).sum(axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def skin_mesh(mesh: UniformMesh, fk: FKResult) -> tuple[np.ndarray, np.ndarray]:
    """Linear blend skinning. `fk` may be in any frame (character or world); bind pose = the pose in `mesh.bind_rotations`
    (the rest pose, or the model's own A-pose). Returns (positions (V, 3), unit normals (V, 3))."""
    R = np.stack([fk.rotations[n] for n in mesh.joint_names])          # (J, 3, 3)
    P = np.stack([fk.positions[n] for n in mesh.joint_names])          # (J, 3)
    j, w = mesh.joints, mesh.weights.astype(float)
    local = mesh.positions[:, None, :] - mesh.bind_positions[j]        # (V, 4, 3) offset from the joint, in the bind pose
    if not mesh.bind_is_rest:                                           # rotate it into the joint's own frame: R_bind^T (v - p_bind)
        local = np.einsum("vkji,vkj->vki", mesh.bind_rotations[j], local)
    local = _untwist(mesh, fk, local)
    pos = (w[..., None] * (np.einsum("vkij,vkj->vki", R[j], local) + P[j])).sum(axis=1)
    return pos, _blend_dirs(mesh, fk, mesh.normals)


def tangent_frame(mesh: UniformMesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex tangent (towards +u) and bitangent (towards the top of the image, glTF's +Y of a normal map) in the bind pose,
    orthonormal to the vertex normal; from the uv derivatives of the triangles, area weighted. Cached on the mesh."""
    if "tangent_frame" in mesh.cache:
        return mesh.cache["tangent_frame"]
    p, uv, t, n = mesh.positions, np.asarray(mesh.uv, float), mesh.triangles, mesh.normals
    e1, e2 = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    d1, d2 = uv[t[:, 1]] - uv[t[:, 0]], uv[t[:, 2]] - uv[t[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > 1e-14
    r = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)[:, None]
    du = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r                        # dP/du
    dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r                        # dP/dv; v runs down the image, the map's +Y runs up
    area = np.linalg.norm(np.cross(e1, e2), axis=1)[:, None]
    T, B = np.zeros_like(p), np.zeros_like(p)
    for k in range(3):
        np.add.at(T, t[:, k], du * area)
        np.add.at(B, t[:, k], -dv * area)
    T -= n * (n * T).sum(axis=1, keepdims=True)
    tl = np.linalg.norm(T, axis=1, keepdims=True)
    fallback = np.cross(n, np.where(np.abs(n[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-12)
    T = np.where(tl > 1e-9, T / np.maximum(tl, 1e-12), fallback)
    B -= n * (n * B).sum(axis=1, keepdims=True)
    B -= T * (T * B).sum(axis=1, keepdims=True)
    bl = np.linalg.norm(B, axis=1, keepdims=True)
    B = np.where(bl > 1e-9, B / np.maximum(bl, 1e-12), np.cross(n, T))
    mesh.cache["tangent_frame"] = (T, B)
    return T, B


def rasterize(sx: np.ndarray, sy: np.ndarray, iz: np.ndarray, tris: np.ndarray, W: int, H: int,
              budget: int = 1_500_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact z-buffer. sx, sy: vertex screen coordinates (pixel centres at +0.5); iz: 1 / camera depth per vertex (larger = nearer).
    Returns (tri_id (H, W) int32, -1 where empty; bary (H, W, 3) screen-space barycentric weights)."""
    X, Y, Z = sx[tris], sy[tris], iz[tris]                                         # (T, 3)
    den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
    pxmin = np.maximum(np.ceil(X.min(1) - 0.5), 0).astype(np.int64)
    pxmax = np.minimum(np.floor(X.max(1) - 0.5), W - 1).astype(np.int64)
    pymin = np.maximum(np.ceil(Y.min(1) - 0.5), 0).astype(np.int64)
    pymax = np.minimum(np.floor(Y.max(1) - 0.5), H - 1).astype(np.int64)
    alive = (np.abs(den) > 1e-9) & (pxmax >= pxmin) & (pymax >= pymin)
    size = np.where(alive, np.maximum(pxmax - pxmin, pymax - pymin) + 1, 0)

    zbuf = np.full(H * W, -np.inf)
    tri_id = np.full(H * W, -1, np.int32)
    bary = np.zeros((H * W, 3), np.float32)

    def fragments(idx: np.ndarray, k_x: int, k_y: int):
        px = pxmin[idx][:, None, None] + np.arange(k_x)[None, None, :]
        py = pymin[idx][:, None, None] + np.arange(k_y)[None, :, None]
        valid = (px <= pxmax[idx][:, None, None]) & (py <= pymax[idx][:, None, None])
        cx, cy = px + 0.5, py + 0.5
        x0, x1, x2 = (X[idx, i][:, None, None] for i in range(3))
        y0, y1, y2 = (Y[idx, i][:, None, None] for i in range(3))
        d = den[idx][:, None, None]
        l0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / d
        l1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / d
        l2 = 1.0 - l0 - l1
        inside = valid & (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        ti, yy, xx = np.nonzero(inside)
        if ti.size == 0:
            return
        tri = idx[ti]
        a, b, c = l0[ti, yy, xx], l1[ti, yy, xx], l2[ti, yy, xx]
        depth = a * Z[tri, 0] + b * Z[tri, 1] + c * Z[tri, 2]
        pix = (py[ti, yy, 0] * W + px[ti, 0, xx]).astype(np.int64)
        keep = depth > zbuf[pix]
        if not keep.any():
            return
        pix, tri, depth, a, b, c = pix[keep], tri[keep], depth[keep], a[keep], b[keep], c[keep]
        np.maximum.at(zbuf, pix, depth)
        win = depth >= zbuf[pix]
        tri_id[pix[win]] = tri[win]
        bary[pix[win]] = np.stack([a[win], b[win], c[win]], axis=1)

    k = 1
    while k <= _BIG:
        sel = np.nonzero((size > k // 2 if k > 1 else size > 0) & (size <= k))[0]
        for s in range(0, len(sel), max(1, budget // (k * k))):
            fragments(sel[s: s + max(1, budget // (k * k))], k, k)
        k *= 2
    for t in np.nonzero(size > _BIG)[0]:
        fragments(np.array([t]), int(pxmax[t] - pxmin[t] + 1), int(pymax[t] - pymin[t] + 1))
    return tri_id.reshape(H, W), bary.reshape(H, W, 3)


def material_albedo(mesh: UniformMesh, colors: dict[str, list[int]], brightness: float = 1.0) -> np.ndarray:
    """(n_materials, 3) albedo in 0..1 from the uniform's colours. Only materials flagged "jitter" (the cloth) are scaled."""
    base = np.zeros((len(mesh.materials), 3))
    for i, m in enumerate(mesh.materials):
        if m.get("texture") is not None:                      # textured: the image supplies the colour (see _shade)
            continue
        if m["region"] not in colors:
            raise ValueError(f"uniform has no colour for region '{m['region']}' (needed by the mesh)")
        c = np.asarray(colors[m["region"]], float) / 255.0
        base[i] = np.clip(c * (brightness if m["jitter"] else 1.0), 0.0, 1.0)
    return base


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_MIPS: dict = {}       # id(texture array) -> (the array, its mip chain): meshes made per sample (baton, podium) share the model's textures


def _mip_chain(tex: np.ndarray, cache: dict, key: int) -> list[np.ndarray]:
    """The texture and its half-size reductions (box filter), float32 0..1; cached per texture array (not per mesh: a mesh with the
    baton or the podium added is a new object every sample, and rebuilding the chains of three 4k maps took 85 % of a frame)."""
    hit = _MIPS.get(id(tex))
    if hit is not None and hit[0] is tex:
        return hit[1]
    if len(_MIPS) > 16:
        _MIPS.clear()
    ck = ("mips", key)
    cache = {}
    if ck not in cache:
        levels = [tex.astype(np.float32) / 255.0]
        while min(levels[-1].shape[:2]) > 8:
            a = levels[-1]
            h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
            levels.append(a[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)))
        cache[ck] = levels
    _MIPS[id(tex)] = (tex, cache[ck])
    return cache[ck]


def _bilinear(t: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = t.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    y = np.clip(uv[:, 1], 0.0, 1.0) * (h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
    bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _sample_lod(levels: list[np.ndarray], uv: np.ndarray, lod: np.ndarray) -> np.ndarray:
    """Trilinear sample of a mip chain at glTF uvs (clamped); `lod` per sample in levels (0 = full resolution). Floats 0..1."""
    lod = np.clip(lod, 0.0, len(levels) - 1.0)
    l0 = np.floor(lod).astype(np.int64)
    fr = (lod - l0)[:, None]
    out = np.zeros((len(uv), 3), np.float32)
    for lv in np.unique(l0):
        sel = np.nonzero(l0 == lv)[0]
        v = _bilinear(levels[lv], uv[sel])
        if lv + 1 < len(levels):
            v = v * (1 - fr[sel]) + _bilinear(levels[lv + 1], uv[sel]) * fr[sel]
        out[sel] = v
    return out


def _ambient_occlusion(P: np.ndarray, N: np.ndarray, xs: np.ndarray, ys: np.ndarray, pos_img: np.ndarray, dist_cam: np.ndarray,
                       focal_px: float) -> np.ndarray:
    """Screen-space ambient occlusion of the covered samples: 1 = open, towards 0 = in a crease (armpit, under the collar, between
    the arm and the body). The neighbours within AO_RADIUS_CM (world) are looked up in the image of surface positions."""
    H, W = pos_img.shape[:2]
    ang = np.random.default_rng(7).uniform(0, 2 * np.pi, len(xs))           # fixed pattern: the picture does not depend on a seed
    occ = np.zeros(len(xs), np.float32)
    r_px = AO_RADIUS_CM * focal_px / np.maximum(dist_cam, 1.0)
    for k in range(AO_SAMPLES):
        a = ang + k * 2.399963
        rad = np.sqrt((k + 0.5) / AO_SAMPLES) * r_px
        nx = np.clip(np.rint(xs + np.cos(a) * rad).astype(np.int64), 0, W - 1)
        ny = np.clip(np.rint(ys + np.sin(a) * rad).astype(np.int64), 0, H - 1)
        v = pos_img[ny, nx] - P
        dist = np.linalg.norm(v, axis=1)
        valid = np.isfinite(dist) & (dist > 1e-4) & (dist < AO_RADIUS_CM)
        vd = np.where(valid, dist, 1.0)
        cosang = np.where(valid, np.nan_to_num((v * N).sum(axis=1)) / vd, 0.0)
        occ += np.where(valid, np.clip(cosang - 0.15, 0.0, 1.0) * (1.0 - vd / AO_RADIUS_CM), 0.0).astype(np.float32)
    return np.clip(1.0 - AO_STRENGTH * occ / AO_SAMPLES * 2.2, 0.0, 1.0)


def _shade(mesh: UniformMesh, fk_world: FKResult, camera: Camera, colors: dict[str, list[int]], lighting: dict,
           brightness: float) -> tuple[np.ndarray, np.ndarray]:
    """Skin, project, rasterise and shade at SUPERSAMPLE x resolution. Returns (colour (H*S, W*S, 3) float32 in sRGB, zero where the
    mesh is absent; covered (H*S, W*S) bool; covered by the officer's own triangles, which differs from `covered` only when the mesh
    carries scenery such as the podium).

    The shading is physically motivated but cheap: the albedo is linearised, the model's normal map bends the shading normal, the
    light is a sun (Lambert + GGX highlight from the model's roughness/metal map), a weaker fill from the other side, a sky/ground
    ambient with screen-space occlusion and a faint rim; a soft shoulder curve takes the result back to sRGB."""
    W, H, S = camera.width, camera.height, SUPERSAMPLE
    pos, nrm = skin_mesh(mesh, fk_world)
    has_tex = mesh.uv is not None and bool(mesh.textures)
    if has_tex:
        Tb, Bb = tangent_frame(mesh)
        tan, bit = _blend_dirs(mesh, fk_world, Tb), _blend_dirs(mesh, fk_world, Bb)
    uv, depth = camera.project(pos)
    tris = mesh.triangles
    behind = (depth[tris] <= 1.0).any(axis=1) | ~np.isfinite(uv[tris]).all(axis=(1, 2))
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)
    toward_cam = ((np.asarray(camera.position) - p0) * face_n).sum(axis=1) > 0
    keep = toward_cam & ~behind                                            # back-face + near-plane culling
    t = tris[keep]
    kept_idx = np.nonzero(keep)[0]
    tri_mat = mesh.tri_material[keep]

    sx, sy = np.nan_to_num(uv[:, 0]) * S, np.nan_to_num(uv[:, 1]) * S
    iz = 1.0 / np.maximum(depth, 1e-6)
    tri_id, bary = rasterize(sx, sy, iz, t, W * S, H * S)

    base = material_albedo(mesh, colors, brightness)
    out = np.zeros((H * S, W * S, 3), np.float32)
    cov = tri_id >= 0
    ids = tri_id[cov]
    w = bary[cov] * iz[t[ids]]                                              # perspective-correct weights
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    n_geo = (w[..., None] * nrm[t[ids]]).sum(axis=1)
    n_geo /= np.maximum(np.linalg.norm(n_geo, axis=1, keepdims=True), 1e-12)
    P = (w[..., None] * pos[t[ids]]).sum(axis=1)
    view = np.asarray(camera.position, float) - P
    dist_cam = np.linalg.norm(view, axis=1)
    view /= np.maximum(dist_cam[:, None], 1e-9)

    albedo = _srgb_to_linear(base[tri_mat[ids]])
    rough = np.full(len(ids), 0.7)
    metal = np.zeros(len(ids))
    n = n_geo.copy()
    mat_tex = _tex_of(mesh)
    tri_lod = np.zeros(len(t))                  # texels per pixel of each triangle -> mip level: the 4k texture must be filtered down
    if has_tex:
        screen_a = np.abs((sx[t[:, 1]] - sx[t[:, 0]]) * (sy[t[:, 2]] - sy[t[:, 0]]) - (sx[t[:, 2]] - sx[t[:, 0]]) * (sy[t[:, 1]] - sy[t[:, 0]]))
        uvt = np.asarray(mesh.uv, float)
        uv_a = np.abs((uvt[t[:, 1], 0] - uvt[t[:, 0], 0]) * (uvt[t[:, 2], 1] - uvt[t[:, 0], 1])
                      - (uvt[t[:, 2], 0] - uvt[t[:, 0], 0]) * (uvt[t[:, 1], 1] - uvt[t[:, 0], 1]))
        tex_px = max(float(mesh.textures[0].shape[0] * mesh.textures[0].shape[1]), 1.0)
        tri_lod = 0.5 * np.log2(np.maximum(uv_a * tex_px, 1e-12) / np.maximum(screen_a, 1e-6))
    for ti, tex in enumerate(mesh.textures):
        sel = np.nonzero(mat_tex[tri_mat[ids]] == ti)[0]
        if sel.size == 0:
            continue
        tri_v = t[ids[sel]]
        uvq = (w[sel][..., None] * mesh.uv[tri_v]).sum(axis=1)             # (n, 2), glTF convention: v from the top
        lod = np.maximum(tri_lod[ids[sel]], 0.0)
        col = _sample_lod(_mip_chain(tex, mesh.cache, ti), uvq, lod)
        if mesh.vertex_jitter is not None:                                    # the brightness range only scales the cloth part of the texture
            jit = (w[sel] * mesh.vertex_jitter[tri_v]).sum(axis=1)
            col = col * (1.0 + (brightness - 1.0) * jit)[:, None]
        albedo[sel] = _srgb_to_linear(np.clip(col, 0.0, 1.0))
        for mi in np.unique(tri_mat[ids[sel]]):
            m0 = mesh.materials[int(mi)]
            part = np.nonzero(tri_mat[ids[sel]] == mi)[0]
            idx = sel[part]
            if m0.get("normal_texture") is not None:                         # tangent-space normal map -> shading normal
                nm = _sample_lod(_mip_chain(mesh.textures[m0["normal_texture"]], mesh.cache, m0["normal_texture"]), uvq[part], lod[part]) * 2.0 - 1.0
                Tq = (w[idx][..., None] * tan[t[ids[idx]]]).sum(axis=1)
                Bq = (w[idx][..., None] * bit[t[ids[idx]]]).sum(axis=1)
                nn = nm[:, 0:1] * Tq + nm[:, 1:2] * Bq + np.maximum(nm[:, 2:3], 0.05) * n_geo[idx]
                nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
                n[idx] = nn
            if m0.get("mr_texture") is not None:                             # glTF: G = roughness, B = metallic
                mr = _sample_lod(_mip_chain(mesh.textures[m0["mr_texture"]], mesh.cache, m0["mr_texture"]), uvq[part], lod[part])
                rough[idx], metal[idx] = np.clip(mr[:, 1], 0.2, 1.0), np.clip(mr[:, 2], 0.0, 1.0)

    rows, cols = np.nonzero(cov)
    pos_img = None
    if AO_SAMPLES:
        pos_img = np.full((H * S, W * S, 3), np.nan, np.float32)
        pos_img[cov] = P
    ao = _ambient_occlusion(P, n_geo, cols, rows, pos_img, dist_cam, camera.fx * S)[:, None] if AO_SAMPLES else np.ones((len(P), 1))

    sun = np.asarray(lighting["sun_direction"], float)
    sun /= np.linalg.norm(sun)
    sun_c = np.asarray(lighting["sun_color_rgb"], float) * lighting["sun_intensity"]
    fill = np.array([-sun[0], -sun[1], 0.35])
    fill /= np.linalg.norm(fill)
    ndl = np.maximum(n @ sun, 0.0)[:, None]
    ndf = np.maximum(n @ fill, 0.0)[:, None]
    sky = (0.5 + 0.5 * n[:, 2])[:, None]                                     # up-facing surfaces see the sky, down-facing the ground
    amb = lighting["ambient_intensity"] * (0.55 + 0.45 * sky) * np.array([0.96, 0.99, 1.04])
    f0 = 0.04 * (1 - metal)[:, None] + albedo * metal[:, None]
    h = sun + view
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
    a2 = np.clip(rough, 0.2, 1.0) ** 4
    nh = np.maximum((n * h).sum(axis=1), 0.0)
    ggx = a2 / (np.pi * (nh * nh * (a2 - 1.0) + 1.0) ** 2)
    vh = np.maximum((view * h).sum(axis=1), 0.0)[:, None]
    fres = f0 + (1.0 - f0) * (1.0 - vh) ** 5
    spec = (ggx * 0.25 * (1.0 - 0.6 * rough))[:, None] * fres * ndl * sun_c
    rim = ((1.0 - np.maximum((n * view).sum(axis=1), 0.0)) ** 3)[:, None] * 0.12 * lighting["ambient_intensity"] * ao
    key = ndl * sun_c * (0.55 + 0.45 * ao)
    fill_l = 0.22 * ndf * np.array([0.9, 0.95, 1.1]) * lighting["sun_intensity"] * ao
    lit = (1 - metal)[:, None] * albedo * (amb * ao + key + fill_l) + spec + rim * albedo
    lit = lit * (1.0 + lit / 4.0) / (1.0 + lit * 0.35)                       # soft shoulder: bright surfaces roll off instead of clipping
    out[cov] = _linear_to_srgb(lit).astype(np.float32)
    inst = cov
    if mesh.instance_tris is not None:
        inst = np.zeros_like(cov)
        inst[cov] = kept_idx[ids] < mesh.instance_tris
    return out, cov, inst


def _tex_of(mesh: UniformMesh) -> np.ndarray:
    """Texture index per material (-1 = untextured)."""
    return np.array([m["texture"] if m.get("texture") is not None else -1 for m in mesh.materials], np.int64)


def render(mesh: UniformMesh, fk_world: FKResult, camera: Camera, backplate: Image.Image, colors: dict[str, list[int]],
           lighting: dict, brightness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb uint8 HxWx3, mask bool HxW). `brightness` scales the albedo of materials flagged "jitter" (the cloth)."""
    W, H, S = camera.width, camera.height, SUPERSAMPLE
    shade, cov, inst = _shade(mesh, fk_world, camera, colors, lighting, brightness)
    img = np.asarray(backplate.convert("RGB").resize((W * S, H * S), Image.BILINEAR), dtype=np.float32) / 255.0
    img[cov] = shade[cov]
    rgb = img.reshape(H, S, W, S, 3).mean(axis=(1, 3))
    rgb = np.clip(rgb * 2.0 ** lighting["exposure_ev"], 0, 1)
    mask = inst.reshape(H, S, W, S).mean(axis=(1, 3)) >= 0.5          # the officer only: scenery in front of it hides it, scenery itself is not the instance
    return (rgb * 255 + 0.5).astype(np.uint8), mask


def render_layer(mesh: UniformMesh, fk_world: FKResult, camera: Camera, colors: dict[str, list[int]], lighting: dict,
                 brightness: float = 1.0) -> np.ndarray:
    """The officer alone as an RGBA uint8 image (alpha = anti-aliased coverage, colour = average of the covered samples), for drawing
    over any background. Compositing it over a constant background reproduces `render` (to rounding)."""
    W, H, S = camera.width, camera.height, SUPERSAMPLE
    shade, cov, _ = _shade(mesh, fk_world, camera, colors, lighting, brightness)
    count = cov.reshape(H, S, W, S).sum(axis=(1, 3)).astype(np.float32)
    rgb = shade.reshape(H, S, W, S, 3).sum(axis=(1, 3)) / np.maximum(count, 1.0)[..., None]
    rgb = np.clip(rgb * 2.0 ** lighting["exposure_ev"], 0, 1)
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = (rgb * 255 + 0.5).astype(np.uint8)
    out[..., 3] = (count / (S * S) * 255 + 0.5).astype(np.uint8)
    return out
