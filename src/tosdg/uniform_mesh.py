"""Skinned uniform mesh: build it from a garment spec, write / read GLB.

The *spec* (profiles/<CC>/assets/<uniform>.spec.json) is data: which tubes, boxes, straps ... make up the uniform and
which skeleton joints they follow. This module only knows generic primitives, so no country logic lives here.

Conventions
  * Character frame: Unreal world, X forward / Y right / Z up, centimetres, rest pose (all DOFs 0) is the bind pose.
  * GLB uses glTF conventions: right-handed, +Y up, the asset's front faces +Z, metres. Unreal is left-handed, so the
    conversion flips handedness and therefore also the triangle winding. `to_gltf` / `from_gltf` are exact inverses.
  * Skinning is 4 influences per vertex, joints are indices into the skeleton's joint order (same order as the GLB skin).

Contents:
  constants: GLTF_SCALE, SPEC_VERSION, MAX_GLB_BYTES, MAX_TEXTURE_PIXELS
  to_gltf
  from_gltf
  class UniformMesh — A skinned mesh
  spec_sha256
  build_uniform_mesh
  write_glb — Skinned mesh + skeleton as a binary glTF 2.0 file (nodes…
  read_glb — Read a GLB written by `write_glb` (one skin, one set of…
  check_against_skeleton — Problems that would make the asset unusable with this…
  (+ 9 private helpers)
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .skeleton import Skeleton

GLTF_SCALE = 0.01
SPEC_VERSION = "1.0"

# ----------------------------------------------------------------------------- GLB
# character frame -> glTF axes as a pure linear map (no scale): forward (+X) -> +Z, up (+Z) -> +Y, right (+Y) -> -X
_A = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
MAX_GLB_BYTES = 200 * 1024 * 1024
MAX_TEXTURE_PIXELS = 8192 * 8192




def to_gltf(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.stack([-p[..., 1], p[..., 2], p[..., 0]], axis=-1) * GLTF_SCALE


def from_gltf(g) -> np.ndarray:
    g = np.asarray(g, float) / GLTF_SCALE
    return np.stack([g[..., 2], -g[..., 0], g[..., 1]], axis=-1)


def _dir_to_gltf(n) -> np.ndarray:
    n = np.asarray(n, float)
    return np.stack([-n[..., 1], n[..., 2], n[..., 0]], axis=-1)


def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------- result container
@dataclass
class UniformMesh:
    """A skinned mesh. Bind pose = the skeleton posed with `info["bind_pose"]` (DOF degrees; {} = rest pose): `bind_positions` /
    `bind_rotations` are the world transforms of every joint in that pose, in the character frame. Vertices live in the bind pose."""
    joint_names: list[str]
    bind_positions: np.ndarray       # (J, 3) cm, character frame
    positions: np.ndarray            # (V, 3) cm
    normals: np.ndarray              # (V, 3)
    joints: np.ndarray               # (V, 4) int32, indices into joint_names
    weights: np.ndarray              # (V, 4) float32, rows sum to 1
    triangles: np.ndarray            # (T, 3) int32, counter-clockwise seen from outside (character frame)
    tri_material: np.ndarray         # (T,) int32
    materials: list[dict]            # {"name", "region", "jitter"} (+ "texture": index into `textures` for textured materials)
    info: dict
    bind_rotations: np.ndarray | None = None      # (J, 3, 3); None = identity (rest pose bind)
    uv: np.ndarray | None = None                  # (V, 2) glTF convention (v measured from the top of the image)
    textures: list = field(default_factory=list)  # uint8 (h, w, 3) base-colour images
    vertex_jitter: np.ndarray | None = None       # (V,) 0..1: how much of this vertex's texture is cloth that the brightness range may scale
    instance_tris: int | None = None              # only the first N triangles are the officer (instance mask / bbox); None = all. The rest is scenery (podium)
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.bind_rotations is None:
            self.bind_rotations = np.tile(np.eye(3), (len(self.joint_names), 1, 1))

    @property
    def bind_is_rest(self) -> bool:
        return bool(np.allclose(self.bind_rotations, np.eye(3)))

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


# ----------------------------------------------------------------------------- spec handling
def spec_sha256(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _mirror_y(obj):
    if isinstance(obj, list) and len(obj) == 3 and all(isinstance(x, (int, float)) for x in obj):
        return [obj[0], -obj[1], obj[2]]
    return obj


def _expand_sides(spec: dict) -> list[dict]:
    """A part with "for_sides": ["l", "r"] is written once for the right side; the left copy is mirrored across Y.
    ({s} in names becomes the side letter; "sides" is the number of segments around a tube.)"""
    out = []
    for part in spec["parts"]:
        sides = part.get("for_sides")
        if not sides:
            out.append(part)
            continue
        for s in sides:
            raw = json.dumps(part).replace("{s}", s)
            p = json.loads(raw)
            p.pop("for_sides", None)
            p["id"] = f"{part['id']}_{s}"
            p["side"] = s
            if s == "l":
                for st in p.get("path", []):
                    if "d" in st:
                        st["d"] = _mirror_y(st["d"])
                if isinstance(p.get("center"), list) and len(p["center"]) == 2:
                    p["center"] = [p["center"][0], _mirror_y(p["center"][1])]
                if "d" in p:
                    p["d"] = _mirror_y(p["d"])
            out.append(p)
    return out


# ----------------------------------------------------------------------------- builder
class _Part:
    def __init__(self, name, material, verts, tris, joints, weights, normals=None):
        self.name, self.material = name, material
        self.verts, self.tris = np.asarray(verts, float), np.asarray(tris, np.int64).reshape(-1, 3)
        self.joints, self.weights = np.asarray(joints, np.int64), np.asarray(weights, float)
        self.normals = normals


class _Builder:
    def __init__(self, spec: dict, sk: Skeleton):
        self.spec, self.sk = spec, sk
        self.joint_names = list(sk.joints)
        self.jidx = {n: i for i, n in enumerate(self.joint_names)}
        fk = sk.fk({})
        for n, r in fk.rotations.items():
            if not np.allclose(r, np.eye(3)):
                raise ValueError(f"rest pose is not identity at joint {n}: the bind pose assumes identity rest rotations")
        self.rest = np.array([fk.positions[n] for n in self.joint_names])
        self.rest_by_name = {n: fk.positions[n] for n in self.joint_names}
        self.materials: list[dict] = []
        self.mat_idx: dict[str, int] = {}
        self.parts: list[_Part] = []
        self.tubes: dict[str, dict] = {}
        self.jitter = set(spec.get("jitter_regions", []))

    # -- helpers
    def material(self, region: str) -> int:
        if region not in self.mat_idx:
            self.mat_idx[region] = len(self.materials)
            self.materials.append({"name": region, "region": region, "jitter": region in self.jitter})
        return self.mat_idx[region]

    def jpos(self, name: str) -> np.ndarray:
        if name not in self.rest_by_name:
            raise ValueError(f"spec refers to unknown joint '{name}'")
        return self.rest_by_name[name]

    def at(self, at, d=None) -> np.ndarray:
        p = self.jpos(at[0]).copy()
        if len(at) == 3:
            p = p + float(at[2]) * (self.jpos(at[1]) - p)
        return p + (np.asarray(d, float) if d is not None else 0.0)

    def chain_weights(self, chain: list[str], point, blend_cm: float):
        pos = np.array([self.jpos(j) for j in chain])
        best = (1e18, 0, 0.0)
        for i in range(len(chain) - 1):
            a, b = pos[i], pos[i + 1]
            ab = b - a
            t = float(np.clip((point - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
            dist = float(np.linalg.norm(point - (a + t * ab)))
            if dist < best[0] - 1e-9:
                best = (dist, i, t)
        _, i, t = best
        seg = float(np.linalg.norm(pos[i + 1] - pos[i]))
        bl = min(0.5, blend_cm / max(seg, 1e-6))
        w: dict[str, float] = {}
        wn = 0.5 * float(_smooth((t - (1 - bl)) / bl)) if bl > 0 else 0.0
        wp = 0.5 * (1 - float(_smooth(t / bl))) if (i > 0 and bl > 0) else 0.0
        w[chain[i]] = 1.0 - wn - wp
        w[chain[i + 1]] = w.get(chain[i + 1], 0.0) + wn
        if i > 0:
            w[chain[i - 1]] = w.get(chain[i - 1], 0.0) + wp
        return w

    def skin_for(self, skin: dict, point):
        if "joint" in skin:
            w = {skin["joint"]: 1.0}
        else:
            w = self.chain_weights(skin["chain"], np.asarray(point, float), float(skin.get("blend_cm", 6.0)))
        items = sorted(((k, v) for k, v in w.items() if v > 1e-6), key=lambda kv: -kv[1])[:4]
        tot = sum(v for _, v in items)
        j = [self.jidx[k] for k, _ in items] + [0] * (4 - len(items))
        ww = [v / tot for _, v in items] + [0.0] * (4 - len(items))
        return np.array(j), np.array(ww)

    # -- primitives ------------------------------------------------------------------
    def tube(self, p: dict):
        st = p["path"]
        n, sides = len(st), int(p.get("sides", 16))
        c = np.array([self.at(s["at"], s.get("d")) for s in st])
        ab = np.array([s["ab"] for s in st], float)
        ref = {"x": np.array([1.0, 0, 0]), "z": np.array([0, 0, 1.0]), "y": np.array([0, 1.0, 0])}[p.get("ref", "x")]
        tang = np.zeros_like(c)
        tang[0], tang[-1] = c[1] - c[0], c[-1] - c[-2]
        tang[1:-1] = c[2:] - c[:-2]
        tang = _unit(tang)
        e1 = _unit(ref - (tang @ ref)[:, None] * tang)
        e2 = np.cross(tang, e1)
        phi = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        cs, sn = np.cos(phi), np.sin(phi)
        ring = (c[:, None, :] + ab[:, 0, None, None] * cs[None, :, None] * e1[:, None, :]
                + ab[:, 1, None, None] * sn[None, :, None] * e2[:, None, :])        # (n, sides, 3)
        verts = [ring.reshape(-1, 3)]
        tris, expect = [], []
        for k in range(n - 1):
            for i in range(sides):
                i2 = (i + 1) % sides
                a, b, c2, d = k * sides + i, k * sides + i2, (k + 1) * sides + i2, (k + 1) * sides + i
                for tri in ((a, b, c2), (a, c2, d)):
                    tris.append(tri)
                    vs = ring.reshape(-1, 3)[list(tri)].mean(axis=0)
                    out = vs - c[k]
                    out = out - (out @ tang[k]) * tang[k]
                    expect.append(out)
        tris, expect = np.array(tris), np.array(expect)
        nrm = np.cross(ring.reshape(-1, 3)[tris[:, 1]] - ring.reshape(-1, 3)[tris[:, 0]],
                       ring.reshape(-1, 3)[tris[:, 2]] - ring.reshape(-1, 3)[tris[:, 0]])
        if (nrm * expect).sum() < 0:
            tris = tris[:, [0, 2, 1]]
        base = len(verts[0])
        extra_v, extra_t, owner = [], [], list(np.repeat(np.arange(n), sides))
        for which, k in (("start", 0), ("end", n - 1)):
            if not p.get("cap_" + which):
                continue
            ctr = len(verts[0]) + len(extra_v)
            extra_v.append(c[k])
            owner.append(k)
            rv = ring[k]
            start = len(verts[0]) + len(extra_v)
            extra_v.extend(rv)
            owner.extend([k] * sides)
            want = -tang[k] if which == "start" else tang[k]
            for i in range(sides):
                tri = (ctr, start + i, start + (i + 1) % sides)
                v0, v1, v2 = np.array(extra_v[tri[0] - base]), np.array(extra_v[tri[1] - base]), np.array(extra_v[tri[2] - base])
                if np.cross(v1 - v0, v2 - v0) @ want < 0:
                    tri = (tri[0], tri[2], tri[1])
                extra_t.append(tri)
        allv = np.vstack([verts[0]] + ([np.array(extra_v)] if extra_v else []))
        allt = np.vstack([tris] + ([np.array(extra_t)] if extra_t else []))
        cache = {o: self.skin_for(p["skin"], c[o]) for o in set(owner)}      # one weight set per ring: a ring bends as one
        joints = np.array([cache[o][0] for o in owner])
        weights = np.array([cache[o][1] for o in owner])
        self.tubes[p["id"]] = {"centers": c, "ab": ab}
        self.parts.append(_Part(p["id"], self.material(p["material"]), allv, allt, joints, weights))

    def ellipsoid(self, p: dict):
        ctr = self.at([p["center"][0]], p["center"][1])
        a, b, c = p["radii"]
        rings, sides = int(p.get("rings", 10)), int(p.get("sides", 16))
        verts = [ctr + np.array([0, 0, c])]
        for r in range(1, rings):
            th = np.pi * r / rings
            for s in range(sides):
                ph = 2 * np.pi * s / sides
                verts.append(ctr + np.array([a * np.sin(th) * np.cos(ph), b * np.sin(th) * np.sin(ph), c * np.cos(th)]))
        verts.append(ctr + np.array([0, 0, -c]))
        tris = []
        for s in range(sides):
            tris.append((0, 1 + s, 1 + (s + 1) % sides))
        for r in range(rings - 2):
            for s in range(sides):
                a0, a1 = 1 + r * sides + s, 1 + r * sides + (s + 1) % sides
                b0, b1 = a0 + sides, a1 + sides
                tris += [(a0, b0, b1), (a0, b1, a1)]
        last = len(verts) - 1
        for s in range(sides):
            tris.append((last, 1 + (rings - 2) * sides + (s + 1) % sides, 1 + (rings - 2) * sides + s))
        verts, tris = np.array(verts), np.array(tris)
        nrm = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
        out = verts[tris].mean(axis=1) - ctr
        if (nrm * out).sum() < 0:
            tris = tris[:, [0, 2, 1]]
        self._rigid_or_chain(p, verts, tris, ctr)

    def _rigid_or_chain(self, p, verts, tris, ref_point):
        skin = p["skin"]
        if "joint" in skin:
            j1, w1 = self.skin_for(skin, ref_point)
            joints, weights = np.tile(j1, (len(verts), 1)), np.tile(w1, (len(verts), 1))
        else:  # per-vertex weights from the vertex position
            jw = [self.skin_for(skin, v) for v in verts]
            joints, weights = np.array([a for a, _ in jw]), np.array([b for _, b in jw])
        self.parts.append(_Part(p["id"], self.material(p["material"]), verts, tris, joints, weights))

    def box(self, p: dict):
        ctr = self.at([p["center"][0]], p["center"][1])
        sx, sy, sz = (np.array(p["size"], float) / 2.0)
        rot = np.radians(p.get("rot", [0, 0, 0]))
        Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
        Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
        Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]), ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
                 ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]), ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
                 ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]), ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
        verts, tris = [], []
        for nrm, corners in faces:
            base = len(verts)
            for cx_, cy_, cz_ in corners:
                verts.append(ctr + R @ np.array([cx_ * sx, cy_ * sy, cz_ * sz]))
            tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        verts, tris = np.array(verts), np.array(tris)
        n_world = [R @ np.array(f[0]) for f in faces for _ in range(2)]
        fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
        flip = np.array([fn[i] @ n_world[i] < 0 for i in range(len(tris))])
        tris[flip] = tris[flip][:, [0, 2, 1]]
        self._rigid_or_chain(p, verts, tris, ctr)

    def sector(self, p: dict):
        """A thin flat sector (a kepi peak): top and bottom faces so it is visible from both sides."""
        ctr = self.at([p["center"][0]], p["center"][1])
        r, (a0, a1) = float(p["radius"]), p["angle"]
        n = int(p.get("steps", 10))
        tilt = np.radians(p.get("tilt_deg", 0.0))
        th = np.radians(np.linspace(a0, a1, n))
        pts = [ctr] + [ctr + np.array([r * np.cos(t) * np.cos(tilt), r * np.sin(t), -r * np.cos(t) * np.sin(tilt)]) for t in th]
        top = np.array(pts)
        nrm_up = np.cross(top[2] - top[0], top[1] - top[0])
        nrm_up = nrm_up if nrm_up[2] >= 0 else -nrm_up
        verts = np.vstack([top, top - 0.0])
        tris = []
        for i in range(1, n):
            f = (0, i, i + 1)
            fn = np.cross(top[f[1]] - top[f[0]], top[f[2]] - top[f[0]])
            if fn[2] < 0:
                f = (0, i + 1, i)
            tris.append(f)
            tris.append((n + 1 + f[0], n + 1 + f[2], n + 1 + f[1]))
        self._rigid_or_chain(p, verts, np.array(tris), ctr)

    def ribbon(self, p: dict):
        t = self.tubes[p["on"]]
        c, ab = t["centers"], t["ab"]
        order = np.argsort(c[:, 2])
        zs = c[order, 2]
        way = np.array(p["waypoints"], float)      # [z, theta_deg]; theta 0 = forward (+X), 90 = right (+Y)
        lift, width = float(p.get("lift", 0.6)), float(p["width"])
        cx, cy = np.interp(way[:, 0], zs, c[order, 0]), np.interp(way[:, 0], zs, c[order, 1])
        a, b = np.interp(way[:, 0], zs, ab[order, 0]) + lift, np.interp(way[:, 0], zs, ab[order, 1]) + lift
        th = np.radians(way[:, 1])
        pts = np.stack([cx + a * np.cos(th), cy + b * np.sin(th), way[:, 0]], axis=1)
        nrm = _unit(np.stack([b * np.cos(th), a * np.sin(th), np.zeros_like(th)], axis=1))
        tan = np.zeros_like(pts)
        tan[0], tan[-1] = pts[1] - pts[0], pts[-1] - pts[-2]
        tan[1:-1] = pts[2:] - pts[:-2]
        side = _unit(np.cross(nrm, _unit(tan)))
        left, right = pts - side * width / 2, pts + side * width / 2
        verts = np.vstack([left, right])
        m = len(pts)
        tris = []
        for i in range(m - 1):
            q = [(i, i + 1, m + i + 1), (i, m + i + 1, m + i)]
            for f in q:
                fn = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
                tris.append(f if fn @ nrm[i] > 0 else (f[0], f[2], f[1]))
        skin = p["skin"]
        jw = [self.skin_for(skin, v) for v in verts]
        self.parts.append(_Part(p["id"], self.material(p["material"]), verts, np.array(tris),
                                np.array([a_ for a_, _ in jw]), np.array([b_ for _, b_ in jw])))

    def fingers(self, p: dict):
        seg_from = {s["from"]: s for s in self.sk.data["segments"]}
        sides = p.get("only_sides") or list(self.sk.fingers)
        scale, nsides = float(p.get("radius_scale", 1.0)), int(p.get("sides", 8))
        for side in sides:
            for name, joints in self.sk.fingers[side].items():
                tip = seg_from[joints[-1]]["to"]
                chain = list(joints) + [tip]
                path = []
                for j in joints:
                    path.append({"at": [j], "ab": [seg_from[j]["radius"] * scale] * 2})
                path.append({"at": [tip], "ab": [seg_from[joints[-1]]["radius"] * scale * 0.8] * 2})
                self.tube({"id": f"{p['id']}_{name}_{side}", "material": p["material"], "path": path, "sides": nsides, "cap_end": True,
                           "skin": {"chain": chain, "blend_cm": float(p.get("blend_cm", 1.2))}})

    # -- assemble --------------------------------------------------------------------
    def build(self) -> UniformMesh:
        for part in _expand_sides(self.spec):
            getattr(self, {"tube": "tube", "ellipsoid": "ellipsoid", "box": "box", "sector": "sector", "ribbon": "ribbon", "fingers": "fingers"}[part["type"]])(part)
        verts, nrms, jts, wts, tris, tmat, off = [], [], [], [], [], [], 0
        for p in self.parts:
            fn = np.cross(p.verts[p.tris[:, 1]] - p.verts[p.tris[:, 0]], p.verts[p.tris[:, 2]] - p.verts[p.tris[:, 0]])
            vn = np.zeros_like(p.verts)
            for k in range(3):
                np.add.at(vn, p.tris[:, k], fn)
            verts.append(p.verts)
            nrms.append(_unit(vn))
            jts.append(p.joints)
            wts.append(p.weights)
            tris.append(p.tris + off)
            tmat.append(np.full(len(p.tris), p.material))
            off += len(p.verts)
        info = {"uniform": self.spec.get("id"), "character": self.sk.id, "spec_sha256": spec_sha256(self.spec),
                "units": "glTF metres; character frame cm", "parts": [p.name for p in self.parts]}
        return UniformMesh(self.joint_names, self.rest, np.vstack(verts), np.vstack(nrms), np.vstack(jts).astype(np.int32),
                           np.vstack(wts).astype(np.float32), np.vstack(tris).astype(np.int32), np.concatenate(tmat).astype(np.int32),
                           self.materials, info)


def build_uniform_mesh(spec: dict, sk: Skeleton) -> UniformMesh:
    if spec.get("schema_version") != SPEC_VERSION:
        raise ValueError(f"garment spec schema_version must be {SPEC_VERSION}")
    if spec.get("character") not in (None, sk.id):
        raise ValueError(f"spec is for character '{spec['character']}', skeleton is '{sk.id}'")
    return _Builder(copy.deepcopy(spec), sk).build()


def _srgb_to_linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _quat_from_matrix(R: np.ndarray) -> list[float]:
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return [float(v) for v in q / np.linalg.norm(q)]


def write_glb(mesh: UniformMesh, sk: Skeleton, path: str | Path, colours: dict | None = None) -> None:
    """Skinned mesh + skeleton as a binary glTF 2.0 file (nodes carry the bind pose; textures are embedded as PNG). `colours`
    (region -> [r,g,b]) only sets the untextured materials' base colour for viewers / Unreal; the renderer in this project takes
    colours from the profile at render time."""
    colours = colours or {}
    chunks: list[bytes] = []
    views, accessors = [], []

    def add_bytes(data: bytes, target: int | None = None) -> int:
        pad = (-len(data)) % 4
        off = sum(len(c) for c in chunks)
        chunks.append(data + b"\0" * pad)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), **({"target": target} if target else {})})
        return len(views) - 1

    def add(arr: np.ndarray, comp: int, typ: str, target: int | None, minmax=False) -> int:
        bv = add_bytes(np.ascontiguousarray(arr).tobytes(), target)
        acc = {"bufferView": bv, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"], acc["max"] = [float(x) for x in arr.min(axis=0)], [float(x) for x in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    J = len(mesh.joint_names)
    jidx = {n: i for i, n in enumerate(mesh.joint_names)}
    Rb, Pb = np.asarray(mesh.bind_rotations, float), np.asarray(mesh.bind_positions, float)
    # nodes: one per skeleton joint, local transform in the bind pose (translation; rotation only when not identity), then the mesh node
    nodes = []
    for n in mesh.joint_names:
        j = sk.joints[n]
        i = jidx[n]
        if j.parent is None:
            Rl, tl = Rb[i], Pb[i]
        else:
            ip = jidx[j.parent]
            Rl, tl = Rb[ip].T @ Rb[i], Rb[ip].T @ (Pb[i] - Pb[ip])
        node = {"name": n, "translation": [float(x) for x in to_gltf(tl)]}
        if not np.allclose(Rl, np.eye(3), atol=1e-12):
            node["rotation"] = _quat_from_matrix(_A @ Rl @ _A.T)
        nodes.append(node)
    for n in mesh.joint_names:
        j = sk.joints[n]
        if j.parent is not None:
            nodes[jidx[j.parent]].setdefault("children", []).append(jidx[n])
    roots = [jidx[n] for n in mesh.joint_names if sk.joints[n].parent is None]
    # inverse bind matrices (column-major 4x4): inverse of the joint's world transform in the bind pose, in glTF axes
    ibm = np.zeros((J, 4, 4), np.float32)
    for i in range(J):
        Rg = _A @ Rb[i] @ _A.T
        tg = to_gltf(Pb[i])
        ibm[i] = np.eye(4)
        ibm[i, :3, :3] = Rg.T
        ibm[i, :3, 3] = -(Rg.T @ tg)
    ibm_acc = add(np.transpose(ibm, (0, 2, 1)).reshape(J, 16), 5126, "MAT4", None)
    pos_g = to_gltf(mesh.positions).astype(np.float32)
    nrm_g = _unit(_dir_to_gltf(mesh.normals)).astype(np.float32)
    tris_g = mesh.triangles[:, [0, 2, 1]]            # handedness flips -> flip the winding as well
    # textures (PNG, embedded)
    images, textures = [], []
    for tex in mesh.textures:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(tex, np.uint8)).save(buf, "PNG")
        images.append({"bufferView": add_bytes(buf.getvalue()), "mimeType": "image/png"})
        textures.append({"sampler": 0, "source": len(images) - 1})
    prims, materials = [], []
    for mi, m in enumerate(mesh.materials):
        sel = mesh.tri_material == mi
        if not sel.any():
            materials.append({"name": m["name"]})
            continue
        extras = {"tosdg_region": m["region"], "tosdg_jitter": bool(m["jitter"])}
        if m.get("texture") is not None:
            pbr = {"baseColorTexture": {"index": int(m["texture"])}, "baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}
            if m.get("mr_texture") is not None:
                pbr = {**pbr, "metallicFactor": 1.0, "roughnessFactor": 1.0, "metallicRoughnessTexture": {"index": int(m["mr_texture"])}}
        else:
            col = _srgb_to_linear(colours.get(m["region"], [180, 180, 180]))
            pbr = {"baseColorFactor": [float(col[0]), float(col[1]), float(col[2]), 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}
        gm = {"name": m["name"], "pbrMetallicRoughness": pbr, "extras": extras}
        if m.get("normal_texture") is not None:
            gm["normalTexture"] = {"index": int(m["normal_texture"])}
        materials.append(gm)
        prims.append({"tri_idx": tris_g[sel].astype(np.uint32), "material": mi})
    attrs = {}
    attrs["POSITION"] = add(pos_g, 5126, "VEC3", 34962, minmax=True)
    attrs["NORMAL"] = add(nrm_g, 5126, "VEC3", 34962)
    if mesh.uv is not None:
        attrs["TEXCOORD_0"] = add(np.asarray(mesh.uv, np.float32), 5126, "VEC2", 34962)
    attrs["JOINTS_0"] = add(mesh.joints.astype(np.uint16), 5123, "VEC4", 34962)
    attrs["WEIGHTS_0"] = add(mesh.weights.astype(np.float32), 5126, "VEC4", 34962)
    if mesh.vertex_jitter is not None:
        attrs["_TOSDG_JITTER"] = add(np.asarray(mesh.vertex_jitter, np.float32), 5126, "SCALAR", 34962)
    primitives = []
    for pr in prims:
        ia = add(pr["tri_idx"].reshape(-1), 5125, "SCALAR", 34963)
        primitives.append({"attributes": attrs, "indices": ia, "material": pr["material"], "mode": 4})
    nodes.append({"name": "uniform_mesh", "mesh": 0, "skin": 0})
    gltf = {
        "asset": {"version": "2.0", "generator": "tosdg uniform_mesh", "extras": {"tosdg": mesh.info}},
        "scene": 0, "scenes": [{"nodes": roots + [len(nodes) - 1]}], "nodes": nodes,
        "skins": [{"joints": list(range(J)), "inverseBindMatrices": ibm_acc, "skeleton": roots[0]}],
        "meshes": [{"name": mesh.info.get("uniform") or "uniform", "primitives": primitives}],
        "materials": materials, "accessors": accessors, "bufferViews": views,
    }
    if images:
        gltf["images"], gltf["textures"] = images, textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    blob = b"".join(chunks)
    gltf["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


_TEXTURES: dict = {}    # sha1 of the encoded image -> decoded read-only array


def read_glb(path: str | Path) -> UniformMesh:
    """Read a GLB written by `write_glb` (one skin, one set of shared vertex attributes for all primitives)."""
    raw = Path(path).read_bytes()
    if len(raw) < 20 or len(raw) > MAX_GLB_BYTES:
        raise ValueError(f"{path}: not a usable GLB (size {len(raw)} bytes)")
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError(f"{path}: not a valid GLB 2.0 file")
    off, js, blob = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", raw, off)
        data = raw[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(data)
        elif ctype == 0x004E4942:
            blob = data
        off += 8 + clen
    g = js

    def acc(i):
        a = g["accessors"][i]
        v = g["bufferViews"][a["bufferView"]]
        dt = np.dtype(_COMP[a["componentType"]]).newbyteorder("<")
        n = a["count"] * _NCOMP[a["type"]]
        arr = np.frombuffer(blob, dtype=dt, count=n, offset=v.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], -1) if _NCOMP[a["type"]] > 1 else arr

    skin = g["skins"][0]
    names = [g["nodes"][j]["name"] for j in skin["joints"]]
    ibm = np.transpose(acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4), (0, 2, 1)).astype(float)    # (J,4,4) row-major
    Rg = np.transpose(ibm[:, :3, :3], (0, 2, 1))                       # joint world rotation in glTF axes (inverse of the stored one)
    tg = -np.einsum("jab,jb->ja", Rg, ibm[:, :3, 3])
    bind_p = from_gltf(tg)
    bind_r = np.einsum("ab,jbc,dc->jad", _A.T, Rg, _A.T)               # A^T Rg A
    prim0 = g["meshes"][0]["primitives"][0]["attributes"]
    pos = from_gltf(acc(prim0["POSITION"]).astype(float))
    nn = acc(prim0["NORMAL"]).astype(float)
    nrm = _unit(np.stack([nn[:, 2], -nn[:, 0], nn[:, 1]], axis=1))
    joints, weights = acc(prim0["JOINTS_0"]).astype(np.int32), acc(prim0["WEIGHTS_0"]).astype(np.float32)
    uv = acc(prim0["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in prim0 else None
    vj = acc(prim0["_TOSDG_JITTER"]).astype(np.float32) if "_TOSDG_JITTER" in prim0 else None
    textures = []
    for t in g.get("textures", []):
        im = g["images"][t["source"]]
        bv = g["bufferViews"][im["bufferView"]]
        data = blob[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
        key = hashlib.sha1(data).hexdigest()                            # decoding a 4k PNG takes ~0.5 s; the same image is decoded once
        arr = _TEXTURES.get(key)
        if arr is None:
            with Image.open(io.BytesIO(data)) as pil:
                if pil.width * pil.height > MAX_TEXTURE_PIXELS:
                    raise ValueError(f"{path}: texture {pil.width}x{pil.height} is too large")
                arr = np.asarray(pil.convert("RGB"))
            arr.flags.writeable = False                                 # shared between meshes: never written in place
            if len(_TEXTURES) > 32:
                _TEXTURES.clear()
            _TEXTURES[key] = arr
        textures.append(arr)
    tris, tmat, mats = [], [], []
    for mi, m in enumerate(g["materials"]):
        ex = m.get("extras", {})
        d = {"name": m["name"], "region": ex.get("tosdg_region", m["name"]), "jitter": bool(ex.get("tosdg_jitter", False))}
        bct = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if bct is not None:
            d["texture"] = int(bct["index"])
        mrt = m.get("pbrMetallicRoughness", {}).get("metallicRoughnessTexture")
        if mrt is not None:
            d["mr_texture"] = int(mrt["index"])
        if m.get("normalTexture") is not None:
            d["normal_texture"] = int(m["normalTexture"]["index"])
        mats.append(d)
    for pr in g["meshes"][0]["primitives"]:
        t = acc(pr["indices"]).reshape(-1, 3).astype(np.int32)[:, [0, 2, 1]]    # undo the winding flip
        tris.append(t)
        tmat.append(np.full(len(t), pr["material"], np.int32))
    info = g["asset"].get("extras", {}).get("tosdg", {})
    return UniformMesh(names, bind_p, pos, nrm, joints, weights, np.vstack(tris), np.concatenate(tmat), mats, info,
                       bind_rotations=bind_r, uv=uv, textures=textures, vertex_jitter=vj)


def check_against_skeleton(mesh: UniformMesh, sk: Skeleton, tol_cm: float = 1e-3, tol_rot: float = 1e-4) -> list[str]:
    """Problems that would make the asset unusable with this skeleton (empty list = fine). The bind pose stored in the file must be
    exactly the skeleton posed with `info["bind_pose"]` (DOF degrees, default = rest pose)."""
    errs = []
    if mesh.joint_names != list(sk.joints):
        errs.append("joint names / order differ from the skeleton")
        return errs
    bind_pose = mesh.info.get("bind_pose", {})
    try:
        fk = sk.fk(bind_pose)
    except KeyError as e:
        return [f"bind_pose refers to an unknown joint {e}"]
    errs += [f"bind_pose: {m}" for m in sk.pose_errors(bind_pose)]
    d = max(float(np.linalg.norm(mesh.bind_positions[i] - fk.positions[n])) for i, n in enumerate(mesh.joint_names))
    if d > tol_cm:
        errs.append(f"bind pose differs from the skeleton posed with the declared bind_pose by up to {d:.4f} cm")
    r = max(float(np.abs(mesh.bind_rotations[i] - fk.rotations[n]).max()) for i, n in enumerate(mesh.joint_names))
    if r > tol_rot:
        errs.append(f"bind rotations differ from the skeleton posed with the declared bind_pose (max {r:.2e})")
    if not np.allclose(mesh.weights.sum(axis=1), 1.0, atol=1e-4):
        errs.append("skin weights do not sum to 1")
    if mesh.joints.max() >= len(mesh.joint_names) or (mesh.weights < 0).any():
        errs.append("invalid skin joint index or negative weight")
    if mesh.triangles.max() >= mesh.n_vertices:
        errs.append("triangle index out of range")
    if not np.isfinite(mesh.positions).all() or not np.isfinite(mesh.normals).all():
        errs.append("non-finite vertex data")
    textured = [m for m in mesh.materials if m.get("texture") is not None]
    if textured and mesh.uv is None:
        errs.append("textured material but no TEXCOORD_0")
    if any(m["texture"] >= len(mesh.textures) for m in textured):
        errs.append("material refers to a missing texture")
    if mesh.uv is not None and (len(mesh.uv) != mesh.n_vertices or not np.isfinite(mesh.uv).all()):
        errs.append("bad TEXCOORD_0")
    return errs
