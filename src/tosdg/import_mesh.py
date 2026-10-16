"""Import a static (unrigged) textured GLB as a skinned uniform mesh for a skeleton.

Steps (all parameters live in a small "fit" JSON next to the asset, so the import is reproducible):
  1. read geometry/UV/texture from the GLB (data only, nothing is executed);
  2. convert to the character frame, scale to `height_cm`, stand on z = 0, centre on the skeleton axis (`offset_cm`);
  3. bind pose = skeleton posed with `bind_pose` (an A-pose when the model's arms are not hanging straight down);
  4. skin weights from bone proximity (distance to the bone segments of the bind pose), 4 influences, smoothed over the mesh edges;
  5. per-vertex "cloth" factor from the texture (skin-coloured texels are not scaled by the brightness range).

Contents:
  constants: MAX_BYTES, DEFAULT_FIT
  read_static_glb — Positions (glTF axes, metres), uv, triangles and the…
  skin_weights — 4 influences per vertex
  cloth_factor — 1 for cloth-like texels, towards 0 for skin-coloured ones…
  import_static_glb
  map_model_joints — Model joint name -> skeleton joint name
  skeleton_usable — (True, []) when the model is skinned and every joint that…
  adjust_model_joints — The fit's hand-made joint corrections (shoulders, wrists, forearms)
  scale_forearm_vertices — Move the mesh with forearm_scale
  close_hand_holes — Fill the small open slits in the hands
  even_armpits — Take the baked dark crease out of the texture around the armpits
  strip_stray_arm_weights — Take the arm's share off belt/hip vertices
  head_landmarks — Face keypoints and head/cap capsule ends measured on the mesh
  fit_bind_pose — DOF degrees that put the skeleton's limbs into the model's…
  import_skinned_glb — A GLB that already has a skin
  import_glb — Import a user-supplied person model
  orient_outward — Make triangles counter-clockwise seen from outside (flip…
  (+ 9 private helpers)
"""
from __future__ import annotations

import io
import itertools
import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .skeleton import Skeleton
from .uniform_mesh import UniformMesh, _unit, from_gltf

_COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
MAX_BYTES = 200 * 1024 * 1024

DEFAULT_FIT = {
    "height_cm": 178.0,                 # total height of the model including headgear
    "offset_cm": [0.0, 0.0],            # forward, right shift applied after centring
    "centre_on": [100.0, 140.0],        # centre the x/y of the vertices between these heights on the skeleton axis
    "bind_pose": {},                    # DOF degrees of the skeleton that matches the model's pose
    "collapse_prefix": ["thumb_", "index_", "middle_", "ring_", "pinky_"],   # finger joints not used as skinning targets
    "blend_cm": 2.5,                    # softness of the proximity weights
    "smooth_iterations": 6,
    "landmarks_cm": {},                 # {joint: [x, y, z]} where the MODEL's own joint sits (character frame); used for skin weights only
    "wrist_drop_cm": 0.0,               # move the model's wrist joint this far along the forearm (towards the hand): rigs often put it in the sleeve
    "strip_stray_arm_weights": True,    # drop the arm's share from belt/hip/jacket-hem vertices (see strip_stray_arm_weights)
    "armpit_below_shoulder_cm": 15.0,   # ... below this distance under the shoulder joint
    "close_hand_holes": True,           # fill small open slits in the hands/fingers of the mesh (see close_hand_holes)
    "even_armpits": None,               # {"inner_cm", "outer_cm", "keep", "below_shoulder_cm"}: take the baked armpit shadow out of the texture (see even_armpits)
    "shoulder_out_cm": 0.0,             # move the model's shoulder joints this far outwards: auto-rigs put them inside the chest
    "forearm_scale": 1.0,               # scale each forearm (and move the hand with it); see adjust_model_joints
    "limb_scale": {},                   # {root joint: factor}: shrink/stretch the limb (the root and all its descendants) about the root
    "bone_radius_cm": {},               # per joint (segment starts at this joint) optional radius override
    "skin_hue_deg": [0.0, 50.0],
    "skin_saturation": [0.12, 0.70],
    "skin_value_min": 0.30,
}

# ----------------------------------------------------------------------------- models that already carry a skeleton
_TWIST = re.compile(r"^(?P<base>.+)_twist_\d+(?P<side>_[lr])?$")
# bone chains whose bind-pose DOFs are fitted so the proxy skeleton takes the model's own rest pose (A-pose, spread legs ...):
# (joints with DOFs, [(bone start, bone end)] whose direction must match the model's, optional bones, joints whose frame must be level)
# The foot is not fitted to the foot -> ball bone: that direction trades off against the ball's offset (any foot flexion fits once the
# skeleton is rebuilt from the model), and the fit settled on a tilted foot - at the rest pose the toes pointed up, the model stood on its
# heels. A model is delivered standing on the ground, so its foot frame is level in the bind pose: then the rest pose stands flat too.
_BIND_CHAINS = [
    (["upperarm_{s}", "lowerarm_{s}", "hand_{s}"], [("upperarm_{s}", "lowerarm_{s}"), ("lowerarm_{s}", "hand_{s}"), ("hand_{s}", "middle_01_{s}")],
     [("pinky_01_{s}", "index_01_{s}")], []),                   # optional: fixes the roll of the forearm (which way the palm faces)
    (["thigh_{s}", "calf_{s}", "foot_{s}"], [("thigh_{s}", "calf_{s}"), ("calf_{s}", "foot_{s}")], [], ["foot_{s}"]),
]




def _load_glb(path: str | Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    if len(raw) < 20 or len(raw) > MAX_BYTES:
        raise ValueError(f"{path}: unusable GLB size {len(raw)}")
    magic, ver, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or ver != 2 or length != len(raw):
        raise ValueError(f"{path}: not a GLB 2.0 file")
    off, g, blob = 12, None, b""
    while off < length:
        n, t = struct.unpack_from("<II", raw, off)
        c = raw[off + 8: off + 8 + n]
        if t == 0x4E4F534A:
            g = json.loads(c)
        elif t == 0x004E4942:
            blob = c
        off += 8 + n
    return g, blob


def _accessor(g: dict, blob: bytes, i: int) -> np.ndarray:
    a = g["accessors"][i]
    v = g["bufferViews"][a["bufferView"]]
    dt = np.dtype(_COMP[a["componentType"]]).newbyteorder("<")
    nc = _NCOMP[a["type"]]
    base = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride")
    if stride and stride != dt.itemsize * nc:
        arr = np.array([np.frombuffer(blob, dt, nc, base + k * stride) for k in range(a["count"])]).reshape(-1)
    else:
        arr = np.frombuffer(blob, dt, a["count"] * nc, base)
    return arr.reshape(a["count"], nc) if nc > 1 else arr


def _base_colour_texture(g: dict, blob: bytes) -> np.ndarray | None:
    """The material's baseColorTexture image (not simply image 0: exporters put normal/ORM maps first)."""
    mats = g.get("materials") or [{}]
    ti = mats[0].get("pbrMetallicRoughness", {}).get("baseColorTexture", {}).get("index")
    if ti is not None:
        src = g["textures"][ti]["source"]
    elif g.get("images"):
        src = 0
    else:
        return None
    bv = g["bufferViews"][g["images"][src]["bufferView"]]
    data = blob[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    with Image.open(io.BytesIO(data)) as pil:
        if pil.width * pil.height > 8192 * 8192:
            raise ValueError("texture too large")
        return np.asarray(pil.convert("RGB"))


def _extra_map(g: dict, blob: bytes, key: str) -> np.ndarray | None:
    """The material's normalTexture (key "normal") or metallicRoughnessTexture (key "mr") image as uint8 RGB, or None when it has none."""
    m = (g.get("materials") or [{}])[0]
    ref = m.get("normalTexture") if key == "normal" else m.get("pbrMetallicRoughness", {}).get("metallicRoughnessTexture")
    if not ref or "index" not in ref:
        return None
    im = g["images"][g["textures"][ref["index"]]["source"]]
    if "bufferView" not in im:
        return None
    bv = g["bufferViews"][im["bufferView"]]
    data = blob[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    with Image.open(io.BytesIO(data)) as pil:
        if pil.width * pil.height > 8192 * 8192:
            raise ValueError("texture too large")
        return np.asarray(pil.convert("RGB"))


def read_static_glb(path: str | Path) -> dict:
    """Positions (glTF axes, metres), uv, triangles and the base-colour image of the first mesh of a GLB.
    When the mesh is skinned, also `joints` (V, 4 indices into `joint_names`), `weights` (V, 4) and `joint_pos` (J, 3, glTF axes,
    metres, rest pose); otherwise those keys are None."""
    g, blob = _load_glb(path)
    pos, uvs, tris, jn, wt, off_v = [], [], [], [], [], 0
    skinned = bool(g.get("skins"))
    for pr in g["meshes"][0]["primitives"]:
        at = pr["attributes"]
        p = _accessor(g, blob, at["POSITION"]).astype(float)
        pos.append(p)
        uvs.append(_accessor(g, blob, at["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in at else np.zeros((len(p), 2), np.float32))
        idx = _accessor(g, blob, pr["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in pr else np.arange(len(p)).reshape(-1, 3)
        tris.append(idx + off_v)
        off_v += len(p)
        if skinned and "JOINTS_0" in at and "WEIGHTS_0" in at:
            jn.append(_accessor(g, blob, at["JOINTS_0"]).astype(np.int64))
            w = _accessor(g, blob, at["WEIGHTS_0"])
            wt.append(w.astype(np.float64) / (np.iinfo(w.dtype).max if w.dtype.kind == "u" else 1.0))
        else:
            skinned = False
    out = {"pos": np.vstack(pos), "uv": np.vstack(uvs), "tris": np.vstack(tris).astype(np.int32), "texture": _base_colour_texture(g, blob),
           "normal_map": _extra_map(g, blob, "normal"), "mr_map": _extra_map(g, blob, "mr"), "joints": None, "weights": None, "joint_names": None, "joint_pos": None}
    if skinned:
        out["joints"], out["weights"] = np.vstack(jn), np.vstack(wt)
        nodes = g["nodes"]
        parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

        def local(n):
            T = np.eye(4)
            if "matrix" in n:
                return np.array(n["matrix"], float).reshape(4, 4).T
            x, y, z, w = n.get("rotation", [0, 0, 0, 1])
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            T[:3, :3] = R * np.asarray(n.get("scale", [1, 1, 1]), float)
            T[:3, 3] = n.get("translation", [0, 0, 0])
            return T

        def world(i):
            m = local(nodes[i])
            while i in parent:
                i = parent[i]
                m = local(nodes[i]) @ m
            return m

        sj = g["skins"][0]["joints"]
        out["joint_names"] = [nodes[i].get("name", "") for i in sj]
        out["joint_pos"] = np.array([world(i)[:3, 3] for i in sj])
        if np.ptp(out["joint_pos"], axis=0).max() < 1e-3 and "inverseBindMatrices" in g["skins"][0]:
            # The bone nodes carry no transforms (some exporters keep the rest pose only in the inverse bind matrices, in a frame that
            # is not the mesh's): every joint would sit at the origin and the skin would collapse. Rebuild the joint positions.
            out["joint_pos"] = _joint_pos_from_inverse_bind(g, blob, out["pos"], out["joints"], out["weights"])
    return out


def _joint_pos_from_inverse_bind(g: dict, blob: bytes, pos: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rest-pose joint positions in the mesh's own axes from the skin's inverse bind matrices. inverse(IBM) places each joint in the
    exporter's armature frame; that frame differs from the mesh's by a rotation by multiples of 90 degrees plus a translation. Both are
    found by matching the joints to the centroids of the vertices they drive (best of the 24 axis rotations, median offset)."""
    ibm = _accessor(g, blob, g["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    arm = np.array([np.linalg.inv(m)[:3, 3] for m in ibm])
    top = joints[np.arange(len(pos)), weights.argmax(axis=1)]
    cen = np.full_like(arm, np.nan)
    for j in range(len(arm)):
        sel = top == j
        if sel.sum() >= 30:
            cen[j] = pos[sel].mean(axis=0)
    ok = ~np.isnan(cen[:, 0])
    if ok.sum() < 8:
        raise ValueError("cannot rebuild the joint positions of this model (bone nodes have no transforms, too few weighted joints)")
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for r, (c, sg) in enumerate(zip(perm, signs)):
                R[r, c] = sg
            if np.linalg.det(R) < 0:
                continue
            P = arm @ R.T
            t = np.median(cen[ok] - P[ok], axis=0)
            err = float(np.sqrt(((cen[ok] - P[ok] - t) ** 2).sum(axis=1).mean()))
            if best is None or err < best[0]:
                best = (err, R, t)
    return arm @ best[1].T + best[2]


def _segments(sk: Skeleton, bind_fk, fit: dict) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Skinning targets: one segment per joint, from the joint to its (first) kept child. Joint positions are the skeleton's bind
    pose unless the model's own joint place is given in `landmarks_cm` (the model's proportions never match a fixed skeleton exactly)."""
    names = list(sk.joints)
    prefixes = tuple(fit["collapse_prefix"])
    lm = {k: np.asarray(v, float) for k, v in fit["landmarks_cm"].items()}

    def at(n):
        return lm.get(n, bind_fk.positions[n])

    keep = [i for i, n in enumerate(names) if not n.startswith(prefixes)]
    children = {n: [] for n in names}
    for n, j in sk.joints.items():
        if j.parent:
            children[j.parent].append(n)
    a, b, r = [], [], []
    default_r = {"head": 9.0, "neck": 6.0, "spine": 15.0, "pelvis": 16.0, "clavicle": 6.0, "upperarm": 5.5, "lowerarm": 4.5, "hand": 4.0,
                 "thigh": 9.0, "calf": 6.0, "foot": 5.0, "ball": 4.0}
    for i in keep:
        n = names[i]
        kids = [c for c in children[n] if not c.startswith(prefixes)]
        p0 = at(n)
        if kids and not n.startswith("hand_"):
            # the child continuing the chain (longest segment)
            p1 = at(max(kids, key=lambda c: np.linalg.norm(at(c) - p0)))
        else:
            par = sk.joints[n].parent
            d = p0 - at(par) if par else np.array([0.0, 0.0, 1.0])
            p1 = p0 + d / max(np.linalg.norm(d), 1e-9) * (9.0 if n.startswith("hand_") else 5.0)     # end joint: a short stub along the bone
        a.append(p0)
        b.append(p1)
        rad = fit["bone_radius_cm"].get(n)
        if rad is None:
            rad = next((v for k, v in default_r.items() if n.startswith(k)), 5.0)
        r.append(rad)
    return keep, np.array(a), np.array(b), np.array(r)


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(V, S) distances from points to segments."""
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-9)
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(axis=2) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2)


def skin_weights(pos: np.ndarray, tris: np.ndarray, keep: list[int], a, b, r, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """4 influences per vertex: softmax of -(normalised distance) over the bone segments, then smoothing over the mesh edges.
    Vertices at the same place (UV-seam duplicates) are welded first so they always get identical weights (no cracks)."""
    key = np.round(pos / 0.01).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    wp = pos[first]                                                         # welded positions
    wt = inv[tris]
    d = _point_segment_distance(wp, a, b)                                   # (U, S)
    score = d / r[None]
    tau = fit["blend_cm"] / 5.0                                             # normalised units: blend over ~blend_cm for a 5 cm bone
    w = np.exp(-(score - score.min(axis=1, keepdims=True)) / max(tau, 1e-3))
    U = len(wp)
    e = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    deg = np.bincount(np.concatenate([e[:, 0], e[:, 1]]), minlength=U).astype(float)
    for _ in range(int(fit["smooth_iterations"])):
        acc = np.zeros_like(w)
        np.add.at(acc, e[:, 0], w[e[:, 1]])
        np.add.at(acc, e[:, 1], w[e[:, 0]])
        w = 0.5 * w + 0.5 * acc / np.maximum(deg, 1.0)[:, None]
    top = np.argsort(-w, axis=1)[:, :4]
    tw = np.take_along_axis(w, top, axis=1)
    tw[tw < 0.02 * tw.sum(axis=1, keepdims=True)] = 0.0
    tw /= np.maximum(tw.sum(axis=1, keepdims=True), 1e-12)
    joints = np.array(keep, np.int32)[top]
    return joints[inv].astype(np.int32), tw[inv].astype(np.float32)


def _is_descendant(sk: Skeleton, name: str, root: str) -> bool:
    while name is not None:
        if name == root:
            return True
        name = sk.joints[name].parent
    return False


def cloth_factor(tex: np.ndarray, uv: np.ndarray, fit: dict) -> np.ndarray:
    """1 for cloth-like texels, towards 0 for skin-coloured ones (so brightness jitter does not recolour faces/hands)."""
    h, w = tex.shape[:2]
    x = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    y = np.clip((uv[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
    c = tex[y, x].astype(float) / 255.0
    mx, mn = c.max(axis=1), c.min(axis=1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    d = np.maximum(mx - mn, 1e-9)
    hue = np.where(mx == c[:, 0], ((c[:, 1] - c[:, 2]) / d) % 6, np.where(mx == c[:, 1], (c[:, 2] - c[:, 0]) / d + 2, (c[:, 0] - c[:, 1]) / d + 4)) * 60
    skin = ((hue >= fit["skin_hue_deg"][0]) & (hue <= fit["skin_hue_deg"][1]) & (sat >= fit["skin_saturation"][0])
            & (sat <= fit["skin_saturation"][1]) & (mx >= fit["skin_value_min"]))
    return np.where(skin, 0.0, 1.0).astype(np.float32)


def _to_character(pos_gltf: np.ndarray, extra_gltf: np.ndarray | None, fit: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """glTF metres -> character frame (cm): scale to `height_cm`, stand on z = 0, centre on the skeleton axis. `extra_gltf` (e.g. the
    model's joint positions) gets exactly the same transform as the vertices."""
    p = from_gltf(pos_gltf)
    e = None if extra_gltf is None else from_gltf(extra_gltf)
    z0 = p[:, 2].min()
    scale = fit["height_cm"] / (p[:, 2].max() - z0)
    p = (p - np.array([0.0, 0.0, z0])) * scale
    sel = (p[:, 2] >= fit["centre_on"][0]) & (p[:, 2] <= fit["centre_on"][1])
    shift = np.array([fit["offset_cm"][0], fit["offset_cm"][1] - 0.5 * (p[sel, 1].min() + p[sel, 1].max()), 0.0])   # forward: keep the model's own placement
    p = p + shift
    if e is not None:
        e = (e - np.array([0.0, 0.0, z0])) * scale + shift
    return p, e


def import_static_glb(src: str | Path, sk: Skeleton, fit: dict, info_extra: dict | None = None) -> UniformMesh:
    fit = {**DEFAULT_FIT, **fit}
    g = read_static_glb(src)
    if g["texture"] is None:
        raise ValueError("the model has no base-colour texture")
    p, _ = _to_character(g["pos"], None, fit)
    bind_fk = sk.fk(fit["bind_pose"])
    keep, a, b, r = _segments(sk, bind_fk, fit)
    joints, weights = skin_weights(p, g["tris"], keep, a, b, r, fit)
    if fit["limb_scale"]:                                                    # match the model's limb lengths to the skeleton, then re-skin
        names_all = list(sk.joints)
        for root, factor in fit["limb_scale"].items():
            desc = {names_all.index(n) for n in names_all if _is_descendant(sk, n, root)}
            wl = (weights * np.isin(joints, list(desc))).sum(axis=1)
            p = p + (wl * (factor - 1.0))[:, None] * (p - bind_fk.positions[root])
        joints, weights = skin_weights(p, g["tris"], keep, a, b, r, fit)
    return _assemble(g, p, joints, weights, bind_fk, sk, fit, info_extra)


def smooth_normals(p: np.ndarray, tris: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Unit vertex normals (area-weighted face normals `fn`) shared by every vertex at the same position: exporters split vertices
    along UV seams, and normals computed per split vertex would show those seams as hard edges (faceted fingers, sleeves)."""
    _, group = np.unique(np.round(p, 3), axis=0, return_inverse=True)
    acc = np.zeros((group.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, group[tris[:, k]], fn)
    return _unit(acc[group.ravel()])


def close_hand_holes(p: np.ndarray, tris: np.ndarray, uv: np.ndarray, joints: np.ndarray, weights: np.ndarray, names: list[str],
                     max_edges: int = 10, max_extent_cm: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fill the small open slits in the hands and fingers. A glove modelled as one sheet can carry a slit (the generators of 3D
    models leave one where two patches meet); at rest it is a hairline, but a curled fist opens it into a wedge through which the
    background shows. A hole = a loop of boundary edges (edges of one triangle only, vertices at one position counted as one, so UV seams
    are not holes); it is filled with a fan from its centre when it has at most `max_edges` edges, is smaller than `max_extent_cm` and its
    vertices are mostly skinned to the hand/finger joints (cuffs, the neck and the belt openings are bigger or elsewhere and stay open).
    The patch's vertices are copies of the loop's (same weights) and a centre with their average weights. Returns (p, tris, uv, joints, weights, filled)."""
    hand = {i for i, n in enumerate(names) if n.startswith(("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_"))}
    _, group = np.unique(np.round(p, 3), axis=0, return_inverse=True)
    group = group.ravel()
    rep = np.zeros(group.max() + 1, np.int64)
    rep[group] = np.arange(len(group))
    directed = np.concatenate([group[tris[:, [0, 1]]], group[tris[:, [1, 2]]], group[tris[:, [2, 0]]]])
    key = np.sort(directed, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = directed[(cnt[inv.ravel()] == 1) & (directed[:, 0] != directed[:, 1])]
    nxt: dict[int, list[int]] = {}
    for a, b in boundary:
        nxt.setdefault(int(a), []).append(int(b))
    seen, new_p, new_uv, new_j, new_w, new_tris = set(), [], [], [], [], []
    base, filled = len(p), 0
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur, ok = [start], start, True
        while True:
            outs = nxt.get(cur, [])
            if len(outs) != 1:
                ok = False
                break
            cur = outs[0]
            if cur == start:
                break
            if cur in loop or len(loop) > max_edges:
                ok = False
                break
            loop.append(cur)
        seen.update(loop)
        if not ok or not 3 <= len(loop) <= max_edges:
            continue
        verts = rep[loop]
        if float(np.ptp(p[verts], axis=0).max()) > max_extent_cm:
            continue
        on_hand = np.isin(joints[verts][np.arange(len(verts)), np.argmax(weights[verts], axis=1)], list(hand)).mean()
        if on_hand < 0.6:
            continue
        dense: dict[int, float] = {}
        for v in verts:
            for j, w in zip(joints[v], weights[v]):
                dense[int(j)] = dense.get(int(j), 0.0) + float(w) / len(verts)
        top = sorted(dense.items(), key=lambda kv: -kv[1])[:4]
        j4 = np.zeros(4, joints.dtype)
        w4 = np.zeros(4, weights.dtype)
        for k, (j, w) in enumerate(top):
            j4[k], w4[k] = j, w
        w4 /= w4.sum()
        # The patch gets vertices of its own (copies of the loop's, plus the centre) all with ONE uv - the first loop vertex's, a texel of the
        # glove - because the two sides of a slit usually lie in different islands of the texture, and interpolating between their uvs
        # would paint the patch with whatever lies between them in the image.
        k = base + len(new_p)
        c = k + len(verts)
        for v in verts:
            new_p.append(p[v])
            new_uv.append(uv[verts[0]])
            new_j.append(joints[v])
            new_w.append(weights[v])
        new_p.append(p[verts].mean(axis=0))
        new_uv.append(uv[verts[0]])
        new_j.append(j4)
        new_w.append(w4)
        for i in range(len(verts)):
            a, b = k + i, k + (i + 1) % len(verts)
            new_tris.append([b, a, c])                      # the reverse of the boundary edge a -> b: the sheet continues, same winding
        filled += 1
    if not new_p:
        return p, tris, uv, joints, weights, 0
    return (np.vstack([p, np.array(new_p)]), np.vstack([tris, np.array(new_tris, tris.dtype)]), np.vstack([uv, np.array(new_uv, uv.dtype)]),
            np.vstack([joints, np.array(new_j)]), np.vstack([weights, np.array(new_w)]), filled)


def strip_stray_arm_weights(p: np.ndarray, dense: np.ndarray, names: list[str], z_cut: float) -> tuple[np.ndarray, int]:
    """Remove the arm's share from the skin weights of vertices below the armpit (z < z_cut, character frame at the bind pose) whose
    main bone is not an arm bone. An auto-rig lowers the arms beside the hips in its A-pose and hands the nearby belt, holster and jacket
    hem a share of the forearm; raise the arm and they are dragged along (the pistol holster is torn into black shreds). Sleeves are
    arm-dominant and keep theirs; the armpit and shoulder, above z_cut, keep their blend. Returns (dense, vertices changed)."""
    arm = np.array([n.startswith(("upperarm_", "lowerarm_", "hand_", "thumb_", "index_", "middle_", "ring_", "pinky_")) for n in names])
    stray = ~arm[np.argmax(dense, axis=1)] & (dense[:, arm].sum(axis=1) > 0) & (p[:, 2] < z_cut)
    if not stray.any():
        return dense, 0
    out = dense.copy()
    out[np.ix_(stray, arm)] = 0.0
    out[stray] /= np.maximum(out[stray].sum(axis=1, keepdims=True), 1e-12)
    return out, int(stray.sum())


def even_armpits(tex: np.ndarray, p: np.ndarray, tris: np.ndarray, uv: np.ndarray, axillae: list[np.ndarray], inner_cm: float = 9.0,
                 outer_cm: float = 15.0, keep: float = 0.25) -> tuple[np.ndarray, int]:
    """Take the baked dark crease out of the base-colour texture around the armpits. A model made in an A-pose has the shadow of the
    armpit painted into its texture (dark brown on khaki); raise the arm and that band shows on the side of the body and under the
    sleeve as a dirty stain. Texels of the triangles within `outer_cm` of an armpit point (fully within `inner_cm`, fading to 0 at
    `outer_cm`) that are darker than the cloth around them (the ring outer_cm .. outer_cm + 10 cm, its brighter half) take the cloth's
    hue and most of its brightness; `keep` of their darkness stays as fold detail (the normal map carries the folds). Only cloth-coloured
    texels change (red > green > blue): badges, skin, the white glove stay. The change is spread 6 texels past the triangles so the mip
    levels do not bring the old colour back at island edges. Returns (texture, texels changed)."""
    from .render_mesh import rasterize                                  # z-buffer rasteriser reused in uv space
    t0 = tex.astype(np.float32) / 255.0
    out = t0.copy()
    H, W = t0.shape[:2]
    luma = np.array([0.299, 0.587, 0.114])
    _, weld = np.unique(np.round(p, 3), axis=0, return_inverse=True)
    weld = weld.ravel()
    changed = 0
    for ax in axillae:
        d = np.linalg.norm(p - ax, axis=1)
        dw = np.zeros(weld.max() + 1)
        np.maximum.at(dw, weld, d)
        near = d < outer_cm + 10
        t = tris[near[tris].any(axis=1)]
        if len(t) == 0:
            continue
        tri_id, bary = rasterize(uv[:, 0] * W, uv[:, 1] * H, np.ones(len(p)), t, W, H)
        cov = tri_id >= 0
        ids = tri_id[cov]
        tx = t0[cov]
        cloth = (tx[:, 0] > tx[:, 1] - 0.02) & (tx[:, 1] > tx[:, 2] - 0.02) & (tx[:, 0] - tx[:, 2] > 0.1)
        dist = (bary[cov] * d[t[ids]]).sum(axis=1)
        ring = cloth & (dist > outer_cm) & (dist < outer_cm + 10)
        if ring.sum() < 100:
            continue
        L = tx[ring] @ luma
        ref = np.median(tx[ring][(L > np.percentile(L, 50)) & (L < np.percentile(L, 95))], axis=0)
        lref = float(ref @ luma)
        w = np.clip((outer_cm - dist) / max(outer_cm - inner_cm, 1e-6), 0, 1)
        lt = tx @ luma
        new_l = lt + (lref - lt) * (1 - keep)
        fixed = np.clip(ref[None, :] * (new_l / max(lref, 1e-6))[:, None], 0, 1)
        a = (w * np.clip((lref - lt) / max(lref, 1e-6) * 4.0, 0, 1) * cloth)[:, None]
        delta = np.zeros_like(t0)
        delta[cov] = (fixed - tx) * a
        changed += int((a[:, 0] > 0.01).sum())
        done = np.zeros((H, W), bool)
        done[cov] = True
        for _ in range(6):                                              # carry the change a few texels past the triangle edges
            acc = np.zeros_like(delta)
            n = np.zeros((H, W), np.float32)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sd = np.roll(np.roll(done, dy, 0), dx, 1)
                acc += np.where(sd[..., None], np.roll(np.roll(delta, dy, 0), dx, 1), 0)
                n += sd
            grow = ~done & (n > 0)
            delta[grow] = acc[grow] / n[grow][:, None]
            done |= grow
        out = np.clip(out + delta, 0, 1)
    return (out * 255 + 0.5).astype(np.uint8), changed


def _assemble(g: dict, p: np.ndarray, joints: np.ndarray, weights: np.ndarray, bind_fk, sk: Skeleton, fit: dict,
              info_extra: dict | None) -> UniformMesh:
    tris = g["tris"]
    # normals from the (character-frame) geometry; winding of the source is kept (checked by the caller via the orientation test)
    v0, v1, v2 = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    nrm = smooth_normals(p, tris, fn)
    nrm[np.linalg.norm(nrm, axis=1) < 0.5] = (0.0, 0.0, 1.0)      # vertices only in degenerate (zero-area) triangles have no normal; they cover no pixel, but a unit one keeps every normal valid
    names = list(sk.joints)
    rot = np.stack([bind_fk.rotations[n] for n in names])
    pos_j = np.stack([bind_fk.positions[n] for n in names])
    info = {"uniform": "imported", "bind_pose": fit["bind_pose"], "fit": fit, **(info_extra or {})}
    ea = fit.get("even_armpits")
    if ea:
        axillae = [bind_fk.positions[f"upperarm_{s}"] - np.array([0.0, 0.0, float(ea.get("below_shoulder_cm", 8.0))]) for s in "lr"]
        tex, n_even = even_armpits(g["texture"], p, tris, g["uv"], axillae, float(ea.get("inner_cm", 9.0)), float(ea.get("outer_cm", 15.0)),
                                   float(ea.get("keep", 0.25)))
        g = {**g, "texture": tex}
        info["armpit_texels_evened"] = n_even
    textures, material = [g["texture"]], {"name": "textured", "region": "texture", "jitter": True, "texture": 0}
    for key, field in (("normal_map", "normal_texture"), ("mr_map", "mr_texture")):       # the model's own normal and roughness/metal maps, for the renderer
        if g.get(key) is not None:
            material[field] = len(textures)
            textures.append(g[key])
    return UniformMesh(names, pos_j, p, nrm, joints, weights, tris, np.zeros(len(tris), np.int32), [material], info,
                       bind_rotations=rot, uv=g["uv"], textures=textures, vertex_jitter=cloth_factor(g["texture"], g["uv"], fit))


def map_model_joints(names: list[str], sk: Skeleton, fit: dict) -> list[str | None]:
    """Model joint name -> skeleton joint name. Same name = same joint; `*_twist_NN_x` helper bones fold into their base bone;
    `fit["joint_map"]` ({model name: skeleton joint or null}) overrides both. None = no counterpart."""
    over = fit.get("joint_map", {})
    out = []
    for n in names:
        if n in over:
            out.append(over[n])
        elif n in sk.joints:
            out.append(n)
        else:
            m = _TWIST.match(n)
            base = (m.group("base") + (m.group("side") or "")) if m else None
            out.append(base if base in sk.joints else None)
    return out


def skeleton_usable(g: dict, sk: Skeleton, fit: dict) -> tuple[bool, list[str]]:
    """(True, []) when the model is skinned and every joint that actually carries weight maps onto the skeleton;
    otherwise (False, names of the unmapped weighted joints) - the caller then falls back to rigging a static mesh."""
    if g.get("joints") is None:
        return False, []
    mapped = map_model_joints(g["joint_names"], sk, fit)
    used = np.unique(g["joints"][g["weights"] > 1e-4])
    bad = [g["joint_names"][i] for i in used if mapped[i] is None]
    return not bad, bad


_FINGERS = ("thumb_", "index_", "middle_", "ring_", "pinky_")


def adjust_model_joints(model_pos: dict[str, np.ndarray], fit: dict) -> dict[str, np.ndarray]:
    """The model's joint positions with the fit's hand-made corrections, in this order:
    `shoulder_out_cm` - moves each shoulder pivot (upperarm) outwards, horizontally along clavicle -> upperarm: auto-rigs put it inside
                        the chest, and an arm lowered from the A-pose then swings into the torso. Pivot only.
    `wrist_drop_cm`   - moves each wrist pivot (hand) along the forearm towards the hand: rigs put it inside the sleeve. Pivot only.
    `forearm_scale`   - shortens (< 1) or lengthens each forearm: the hand and the fingers move along the forearm; the mesh is moved with
                        them by `scale_forearm_vertices`.
    Other joints keep their place."""
    out = dict(model_pos)
    so = float(fit.get("shoulder_out_cm", 0.0))
    drop = float(fit.get("wrist_drop_cm", 0.0))
    scale = float(fit.get("forearm_scale", 1.0))
    for side in ("l", "r"):
        c, u, e, w = f"clavicle_{side}", f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}"
        if so and c in out and u in out:
            d = out[u] - out[c]
            d[2] = 0.0
            out[u] = out[u] + so * d / np.linalg.norm(d)
        if e not in out or w not in out:
            continue
        if drop:
            ax = out[w] - out[e]
            out[w] = out[w] + drop * ax / np.linalg.norm(ax)
        if scale != 1.0:
            delta = (scale - 1.0) * (out[w] - out[e])
            for n in out:
                if n == w or (n.startswith(_FINGERS) and n.endswith("_" + side)):
                    out[n] = out[n] + delta
    return out


def scale_forearm_vertices(p: np.ndarray, joints: np.ndarray, weights: np.ndarray, names: list[str], model_pos: dict, fit: dict) -> np.ndarray:
    """Move the mesh with `forearm_scale` (see adjust_model_joints): a vertex moves along the forearm by (scale - 1) x its distance from the
    elbow (clamped to the forearm), times its weight on the forearm, hand and fingers - the sleeve shortens evenly, the hand moves whole,
    the upper arm and the body stay put. `model_pos` = joints before the forearm scale (after the other corrections)."""
    scale = float(fit.get("forearm_scale", 1.0))
    if scale == 1.0:
        return p
    p = p.copy()
    for side in ("l", "r"):
        e, w = f"lowerarm_{side}", f"hand_{side}"
        if e not in model_pos or w not in model_pos:
            continue
        ax = model_pos[w] - model_pos[e]
        length = float(np.linalg.norm(ax))
        d = ax / length
        ids = [i for i, n in enumerate(names) if n in (e, w) or (n.startswith(_FINGERS) and n.endswith("_" + side))]
        wsum = (weights * np.isin(joints, ids)).sum(axis=1)
        t = np.clip((p - model_pos[e]) @ d, 0.0, length)
        p += (wsum * (scale - 1.0) * t)[:, None] * d
    return p


def head_landmarks(mesh: UniformMesh, bind_fk, cap_radius: float) -> dict[str, np.ndarray]:
    """Where the face and the cap of an imported model are, as offsets in the `head` joint's frame (cm; x forward, y right, z up) in
    the bind pose, measured on the mesh: the skeleton's keypoint offsets were made for another head, so the nose/eye/ear keypoints of
    an imported model sit where its face is not. The visor and the cap band are wider and further forward than the face, so the face
    is looked for below them (between the head joint and the brim).
      nose       - the forward-most vertex below the brim
      left/right_ear - the centre of the outermost 1.2 cm of the head on that side, at ear height (nose -2 .. +6 cm), not in front/behind
      left/right_eye - the centre of the darkest 3 % of the face texture (the iris/pupil) in a window 1..5.5 cm beside the middle,
                       1..7 cm above the nose tip, sampled densely over the face triangles (vertices are too sparse for the eyes)
      head_end   - end of the head capsule: centre of the skull front-to-back, 1 cm below the eyes
      cap_top    - end of the cap capsule, one capsule radius below the top of the cap, so the capsule reaches the top and no further
    Left = -y."""
    hid = mesh.joint_names.index("head")
    on_head = (mesh.weights * (mesh.joints == hid)).sum(axis=1) > 0.5
    H, R = bind_fk.positions["head"], bind_fk.rotations["head"]
    P = (mesh.positions - H) @ R                                       # head frame
    brim = 9.0
    below = on_head & (P[:, 2] > -4.0) & (P[:, 2] < brim)
    if not below.any():
        raise ValueError("no head vertices below the brim: cannot find the face")
    nose = P[np.nonzero(below)[0][np.argmax(P[below, 0])]]
    out = {"nose": nose}
    for sign, side in ((-1.0, "left"), (1.0, "right")):
        band = below & (P[:, 2] > nose[2] - 2) & (P[:, 2] < nose[2] + 6) & (np.abs(P[:, 0]) < 6) & (np.sign(P[:, 1]) == sign)
        far = np.abs(P[band, 1]).max()
        out[f"{side}_ear"] = P[band & (np.abs(P[:, 1]) > far - 1.2)].mean(axis=0)
    tex = mesh.textures[0]
    th, tw = tex.shape[:2]
    T = mesh.triangles
    c = P[T].mean(axis=1)
    face = on_head[T].all(axis=1) & (c[:, 0] > nose[0] - 6) & (c[:, 2] > nose[2] + 0.5) & (c[:, 2] < nose[2] + 8) & (np.abs(c[:, 1]) < 6.5)
    n = 8
    bc = np.array([(i / n, j / n, 1 - i / n - j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    pts = np.einsum("kb,tbd->tkd", bc, P[T[face]]).reshape(-1, 3)
    uvs = np.einsum("kb,tbd->tkd", bc, mesh.uv[T[face]]).reshape(-1, 2)
    lum = tex[np.clip((uvs[:, 1] * (th - 1)).astype(int), 0, th - 1), np.clip((uvs[:, 0] * (tw - 1)).astype(int), 0, tw - 1)].astype(float).mean(axis=1)
    for sign, side in ((-1.0, "left"), (1.0, "right")):
        win = ((np.sign(pts[:, 1]) == sign) & (np.abs(pts[:, 1]) > 1.0) & (np.abs(pts[:, 1]) < 5.5)
               & (pts[:, 2] > nose[2] + 1) & (pts[:, 2] < nose[2] + 7) & (pts[:, 0] > nose[0] - 5))
        if not win.any():
            raise ValueError(f"no face texture where the {side} eye should be")
        dark = win & (lum <= np.quantile(lum[win], 0.03))
        out[f"{side}_eye"] = pts[dark].mean(axis=0)
    skull = below & (P[:, 2] > 0)
    eyes_z = 0.5 * (out["left_eye"][2] + out["right_eye"][2])
    out["head_end"] = np.array([0.5 * (P[skull, 0].min() + P[skull, 0].max()), 0.0, eyes_z - 1.0])
    out["cap_top"] = np.array([0.0, 0.0, float(P[on_head, 2].max()) - cap_radius])
    return out


def fit_bind_pose(sk: Skeleton, model_pos: dict[str, np.ndarray], reg: float = 0.002) -> tuple[dict, float]:
    """DOF degrees that put the skeleton's limbs into the model's own rest pose: damped least squares on the bone directions
    (unit vectors), with a small pull towards 0. `model_pos` = the model's joint positions in the character frame.
    Returns (bind_pose, worst remaining bone-direction error in degrees)."""
    pose: dict = {}
    worst = 0.0
    for side in ("l", "r"):
        for joints, bones, optional, level in _BIND_CHAINS:
            joints = [j.format(s=side) for j in joints]
            bones = [(a.format(s=side), b.format(s=side)) for a, b in bones]
            if not all(a in model_pos and b in model_pos for a, b in bones):
                continue
            n_limb = len(bones)
            bones += [(a.format(s=side), b.format(s=side)) for a, b in optional if a.format(s=side) in model_pos and b.format(s=side) in model_pos]
            want = [(model_pos[b] - model_pos[a]) / np.linalg.norm(model_pos[b] - model_pos[a]) for a, b in bones]
            level = [j.format(s=side) for j in level]
            params = [(j, d["name"], float(d["min"]), float(d["max"])) for j in joints for d in sk.joints[j].dofs]

            def build(x):
                ps: dict = {}
                for (j, d, _, _), v in zip(params, x):
                    ps.setdefault(j, {})[d] = float(v)
                return ps

            def resid(x):
                fk = sk.fk(build(x))
                r = []
                for (a, b), w in zip(bones, want):
                    v = fk.positions[b] - fk.positions[a]
                    r.extend(v / np.linalg.norm(v) - w)
                for j in level:                                    # sole flat: the joint's forward and side axes stay horizontal
                    r.extend([fk.rotations[j][2, 0], fk.rotations[j][2, 1]])
                r.extend(reg * np.asarray(x))
                return np.array(r)

            lo, hi = np.array([q[2] for q in params]), np.array([q[3] for q in params])
            x, lam = np.zeros(len(params)), 1e-2
            r = resid(x)
            for _ in range(60):
                J = np.stack([(resid(x + np.eye(len(x))[k] * 0.5) - r) / 0.5 for k in range(len(x))], axis=1)
                dx = np.linalg.solve(J.T @ J + lam * np.eye(len(x)), -J.T @ r)
                xn = np.clip(x + dx, lo, hi)
                rn = resid(xn)
                if rn @ rn < r @ r:
                    x, r, lam = xn, rn, lam * 0.5
                    if np.abs(dx).max() < 1e-3:
                        break
                else:
                    lam *= 4.0
            fk = sk.fk(build(x))
            for (a, b), w in zip(bones[:n_limb], want):
                v = fk.positions[b] - fk.positions[a]
                worst = max(worst, float(np.degrees(np.arccos(np.clip(v @ w / np.linalg.norm(v), -1, 1)))))
            for j, dd in build(x).items():
                pose[j] = {k: round(v, 3) for k, v in dd.items() if abs(v) > 1e-3}
    return {j: d for j, d in pose.items() if d}, worst


def import_skinned_glb(src: str | Path, sk: Skeleton, fit: dict, info_extra: dict | None = None) -> UniformMesh:
    """A GLB that already has a skin: keep its own skin weights (twist bones folded into their base bone) and only fit the
    skeleton's bind pose to the model's rest pose. No weight computation."""
    fit = {**DEFAULT_FIT, **fit}
    g = read_static_glb(src)
    if g["texture"] is None:
        raise ValueError("the model has no base-colour texture")
    ok, bad = skeleton_usable(g, sk, fit)
    if not ok:
        raise ValueError(f"model joints without a counterpart in skeleton {sk.id}: {bad} (add them to fit['joint_map'])")
    p, jp = _to_character(g["pos"], g["joint_pos"], fit)
    mapped = map_model_joints(g["joint_names"], sk, fit)
    names = list(sk.joints)
    idx = np.array([names.index(m) if m else -1 for m in mapped])
    V = len(p)
    dense = np.zeros((V, len(names)))
    rows = np.arange(V)
    for c in range(4):
        col = idx[g["joints"][:, c]]
        keep = col >= 0
        np.add.at(dense, (rows[keep], col[keep]), g["weights"][keep, c])
    stripped = 0
    if fit.get("strip_stray_arm_weights", True):
        z_arm = min(jp[k][2] for k, m in enumerate(mapped) if m in ("upperarm_l", "upperarm_r") and m == g["joint_names"][k])
        dense, stripped = strip_stray_arm_weights(p, dense, names, z_arm - float(fit.get("armpit_below_shoulder_cm", 15.0)))
    top = np.argsort(-dense, axis=1)[:, :4]
    tw = np.take_along_axis(dense, top, axis=1)
    tw /= np.maximum(tw.sum(axis=1, keepdims=True), 1e-12)
    holes = 0
    if fit.get("close_hand_holes", True):
        p, tris_f, uv_f, top, tw, holes = close_hand_holes(p, g["tris"], g["uv"], top, tw, names)
        g = {**g, "tris": tris_f, "uv": uv_f}
    model_pos = {}
    for k, m in enumerate(mapped):
        if m and m == g["joint_names"][k]:                     # the model's own joints only (not folded twist bones)
            model_pos[m] = jp[k]
    p = scale_forearm_vertices(p, top, tw, names, adjust_model_joints(model_pos, {**fit, "forearm_scale": 1.0}), fit)
    model_pos = adjust_model_joints(model_pos, fit)
    explicit = fit.get("bind_pose")
    bind_pose, err = (explicit, 0.0) if explicit else fit_bind_pose(sk, model_pos)
    fit = {**fit, "bind_pose": bind_pose}
    bind_fk = sk.fk(bind_pose)
    # proportions: the skeleton's joints sit where the proxy puts them, the model's where its author put them (shoulder width, head
    # height ...). Carry each vertex along with its joints (weighted joint offsets) so every pivot is exactly the skeleton's.
    off = np.zeros((len(names), 3))
    for n, q in model_pos.items():
        off[names.index(n)] = bind_fk.positions[n] - q
    shift = (tw[..., None] * off[top]).sum(axis=1)
    p = p + shift
    extra = {**(info_extra or {}), "skin": "model", "bind_direction_error_deg": round(err, 2), "hand_holes_closed": holes, "stray_arm_weights_removed": stripped,
             "joint_shift_cm": round(float(np.linalg.norm(off, axis=1).max()), 2)}
    return _assemble(g, p, top.astype(np.int32), tw.astype(np.float32), bind_fk, sk, fit, extra)


def import_glb(src: str | Path, sk: Skeleton, fit: dict, info_extra: dict | None = None) -> UniformMesh:
    """Import a user-supplied person model: use its own skeleton/skin when it has a usable one, otherwise rig it (proximity weights)."""
    g = read_static_glb(src)
    ok, _ = skeleton_usable(g, sk, {**DEFAULT_FIT, **fit})
    return (import_skinned_glb if ok else import_static_glb)(src, sk, fit, info_extra)


def orient_outward(mesh: UniformMesh) -> bool:
    """Make triangles counter-clockwise seen from outside (flip all if the mesh is inside-out). True if flipped."""
    p, t = mesh.positions, mesh.triangles
    c = p.mean(axis=0)
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    cen = p[t].mean(axis=1)
    score = (np.einsum("ij,ij->i", fn, cen - c) > 0).mean()
    if score < 0.5:
        mesh.triangles = t[:, [0, 2, 1]].copy()
        mesh.normals = -mesh.normals
        return True
    return False

