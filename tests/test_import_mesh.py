"""Imported (static, textured) uniform: A-pose bind, GLB round trip with texture, skinning, stale-fit detection.

Contents:
  tiny_mesh()
  ImportedMeshTests (7 tests)
  CommittedImportTests (3 tests)
  SkinnedModelTests (3 tests)
  BoneNodesWithoutTransformsTests (1 tests)
  SmoothNormalsTests (1 tests)
  HandRollTests (5 tests)
  HandHoleTests (4 tests)
"""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import ROOT, skeleton
from test_mesh import LIGHT, make_cam, pose_of
from tosdg.import_mesh import (DEFAULT_FIT, adjust_model_joints, close_hand_holes, scale_forearm_vertices, _accessor, _load_glb, read_static_glb, _segments, cloth_factor, fit_bind_pose, orient_outward,
                               skin_weights, smooth_normals)
from tosdg.render_mesh import render, skin_mesh
from tosdg.schema import load_json
from tosdg.uniform_mesh import UniformMesh, check_against_skeleton, read_glb, spec_sha256, write_glb

PROFILE = ROOT / "profiles" / "VN"
UNI = next(u for u in load_json(PROFILE / "uniforms.json", "uniforms")["uniforms"] if u.get("weight", 0) > 0)       # the active model


def tiny_mesh(sk, bind_pose):
    """A textured quad strip skinned to two joints, bound in a non-rest pose."""
    fk = sk.fk(bind_pose)
    names = list(sk.joints)
    rot = np.stack([fk.rotations[n] for n in names])
    pos = np.stack([fk.positions[n] for n in names])
    a, b = fk.positions["upperarm_l"], fk.positions["lowerarm_l"]
    v = np.array([a + [3, 0, 0], a - [3, 0, 0], b + [3, 0, 0], b - [3, 0, 0]], float)
    tris = np.array([[0, 1, 2], [2, 1, 3]], np.int32)
    j = np.zeros((4, 4), np.int32)
    j[:2, 0], j[2:, 0] = names.index("upperarm_l"), names.index("lowerarm_l")
    w = np.zeros((4, 4), np.float32)
    w[:, 0] = 1
    tex = np.zeros((4, 4, 3), np.uint8)
    tex[..., 0] = 200
    return UniformMesh(names, pos, v, np.tile([0, -1.0, 0], (4, 1)), j, w, tris, np.zeros(2, np.int32),
                       [{"name": "t", "region": "texture", "jitter": True, "texture": 0}], {"bind_pose": bind_pose},
                       bind_rotations=rot, uv=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), textures=[tex],
                       vertex_jitter=np.ones(4, np.float32))


class ImportedMeshTests(unittest.TestCase):
    def test_round_trip_keeps_bind_pose_uv_and_texture(self):
        sk = skeleton()
        bp = {"upperarm_l": {"abduction": 30.0, "twist": -40.0}, "lowerarm_l": {"flexion": 20.0}}
        m = tiny_mesh(sk, bp)
        self.assertEqual(check_against_skeleton(m, sk), [])
        with tempfile.TemporaryDirectory() as d:
            write_glb(m, sk, Path(d) / "t.glb")
            r = read_glb(Path(d) / "t.glb")
        self.assertEqual(check_against_skeleton(r, sk), [])
        np.testing.assert_allclose(r.bind_rotations, m.bind_rotations, atol=1e-5)
        np.testing.assert_allclose(r.bind_positions, m.bind_positions, atol=1e-3)
        np.testing.assert_allclose(r.positions, m.positions, atol=1e-3)
        np.testing.assert_allclose(r.uv, m.uv)
        self.assertEqual(r.textures[0].shape, (4, 4, 3))
        self.assertEqual(r.materials[0]["texture"], 0)

    def test_skinning_in_the_bind_pose_is_identity_and_follows_the_joint(self):
        sk = skeleton()
        bp = {"upperarm_l": {"abduction": 30.0}}
        m = tiny_mesh(sk, bp)
        p, _ = skin_mesh(m, sk.fk(bp))
        np.testing.assert_allclose(p, m.positions, atol=1e-6)
        moved = sk.fk({"upperarm_l": {"abduction": 60.0}})
        q, _ = skin_mesh(m, moved)
        self.assertGreater(np.linalg.norm(q[2] - m.positions[2]), 10.0)                # the elbow end swings with the arm
        self.assertAlmostEqual(float(np.linalg.norm(q[0] - moved.positions["upperarm_l"])), 3.0, places=6)   # the shoulder end stays 3 cm from the joint

    def test_bind_pose_mismatch_is_reported(self):
        sk = skeleton()
        m = tiny_mesh(sk, {"upperarm_l": {"abduction": 30.0}})
        m.info["bind_pose"] = {"upperarm_l": {"abduction": 10.0}}
        self.assertTrue(any("bind" in e for e in check_against_skeleton(m, sk)))

    @staticmethod
    def _faces(m, fk, az):
        p = m.positions
        n = np.cross(p[m.triangles[0, 1]] - p[m.triangles[0, 0]], p[m.triangles[0, 2]] - p[m.triangles[0, 0]])
        return n[1] > 0 if az == 90 else n[1] < 0

    def test_texture_colour_is_rendered_and_brightness_scales_it(self):
        sk = skeleton()
        m = tiny_mesh(sk, {})
        fk = sk.fk({})
        cam = make_cam(90, 200, 160, 240) if self._faces(m, fk, 90) else make_cam(270, 200, 160, 240)
        bg = Image.new("RGB", (160, 240), (0, 0, 0))
        out = {}
        for b in (1.0, 0.5):
            rgb, mask = render(m, fk, cam, bg, {}, LIGHT, b)
            self.assertTrue(mask.any())
            out[b] = rgb[mask].astype(float).mean(axis=0)
        self.assertGreater(out[1.0][0], out[1.0][1] + 20)                               # red texture
        self.assertLess(out[0.5][0], out[1.0][0])

    def test_cloth_factor_excludes_skin_texels(self):
        tex = np.zeros((2, 2, 3), np.uint8)
        tex[0, 0] = (210, 160, 130)                  # skin
        tex[1, 0] = (30, 30, 30)
        tex[1, 1] = (40, 120, 200)
        uv = np.array([[0, 0], [0, 1], [1, 1]], np.float32)
        f = cloth_factor(tex, uv, DEFAULT_FIT)
        self.assertEqual(list(f), [0.0, 1.0, 1.0])

    def test_skin_weights_are_welded_and_normalised(self):
        sk = skeleton()
        pos = np.array([[0, -30, 120], [0, -30, 120], [0, -35, 118], [0, -33, 110]], float)    # vertices 0 and 1 coincide
        tris = np.array([[0, 2, 3], [1, 2, 3]], np.int32)
        keep, a, b, r = _segments(sk, sk.fk({}), dict(DEFAULT_FIT))
        j, w = skin_weights(pos, tris, keep, a, b, r, dict(DEFAULT_FIT))
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(j[0], j[1])
        np.testing.assert_allclose(w[0], w[1])

    def test_orient_outward_flips_an_inside_out_mesh_once(self):
        sk = skeleton()
        m = tiny_mesh(sk, {})
        m.positions = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], float) * 10 + [0, 0, 100]
        m.normals = np.tile([0, 0, 1.0], (4, 1))
        m.triangles = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], np.int32)
        m.tri_material = np.zeros(4, np.int32)

        def outward():
            p, t = m.positions, m.triangles
            fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
            return bool((np.einsum("ij,ij->i", fn, p[t].mean(axis=1) - p.mean(axis=0)) > 0).all())

        before = outward()
        flipped = orient_outward(m)
        self.assertEqual(flipped, not before)
        self.assertTrue(outward())
        self.assertFalse(orient_outward(m))


class CommittedImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sk = skeleton("tripo_v1")                       # the active model is built on the Tripo skeleton
        cls.mesh = read_glb(PROFILE / UNI["mesh"]["asset"])
        cls.fit = json.loads((PROFILE / UNI["mesh"]["fit"]).read_text(encoding="utf-8"))

    def test_asset_matches_skeleton_and_fit(self):
        self.assertEqual(check_against_skeleton(self.mesh, self.sk), [])
        self.assertEqual(self.mesh.info["fit_sha256"], spec_sha256(self.fit), "run tosdg import-model")
        self.assertTrue(self.mesh.textures and self.mesh.uv is not None)
        self.assertTrue(np.allclose(self.mesh.weights.sum(axis=1), 1, atol=1e-4))

    def test_hands_follow_their_bones(self):
        for pose, side in (("stop_bent_l", "l"), ("stop_handup_r", "r"), ("slow_down_r", "r")):
            fk = self.sk.fk(pose_of(pose))
            pos, _ = skin_mesh(self.mesh, fk)
            near = np.linalg.norm(pos - fk.positions[f"hand_{side}"], axis=1).min()
            self.assertLess(near, 12.0, f"{pose}: no mesh vertex within 12 cm of the hand joint")

    def test_keypoints_land_on_the_body(self):
        names = ["head", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "thigh_l", "thigh_r", "calf_l", "calf_r", "spine_02"]
        cam = make_cam(0, 300, 360, 540)
        for pose in ("neutral_stand", "palms_forward_both", "stop_bent_l"):
            fk = self.sk.fk(pose_of(pose))
            _, mask = render(self.mesh, fk, cam, Image.new("RGB", (360, 540), (200, 200, 200)), {}, LIGHT, 1.0)
            for n in names:
                uv, _ = cam.project(fk.positions[n][None])
                x, y = int(round(uv[0, 0])), int(round(uv[0, 1]))
                self.assertTrue(mask[max(y - 5, 0): y + 6, max(x - 5, 0): x + 6].any(), f"{pose}: joint {n} not near the rendered body")


def boundary_loops(positions, tris):
    """Loops of boundary edges (edges of one triangle only; vertices at one position count as one): lists of welded vertex ids."""
    _, group = np.unique(np.round(positions, 3), axis=0, return_inverse=True)
    group = group.ravel()
    directed = np.concatenate([group[tris[:, [0, 1]]], group[tris[:, [1, 2]]], group[tris[:, [2, 0]]]])
    _, inv, cnt = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    edges = directed[cnt[inv.ravel()] == 1]
    nxt = {}
    for a, b in edges:
        nxt.setdefault(int(a), []).append(int(b))
    seen, loops = set(), []
    for start in nxt:
        if start in seen:
            continue
        loop, cur = [start], start
        while len(nxt.get(cur, [])) == 1 and nxt[cur][0] != start and nxt[cur][0] not in loop:
            cur = nxt[cur][0]
            loop.append(cur)
        seen.update(loop)
        loops.append(loop)
    return loops, group


class HandHoleTests(unittest.TestCase):
    """A glove modelled as one sheet can carry a slit; a curled fist opens it into a wedge through which the background shows."""

    @staticmethod
    def sheet(n=6, hole=((2, 2), (2, 3), (3, 3), (3, 2))):
        """A flat n x n grid of vertices (1 cm apart) with the quads at `hole` removed; all skinned to joint 0 (the 'hand'); 4 weights."""
        idx = lambda i, j: i * n + j
        pos = np.array([[i, j, 0.0] for i in range(n) for j in range(n)])
        tris = []
        for i in range(n - 1):
            for j in range(n - 1):
                if (i, j) in hole:
                    continue
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                tris += [[a, b, c], [a, c, d]]
        uv = np.array([[i / n, j / n] for i in range(n) for j in range(n)], np.float32)
        joints = np.zeros((n * n, 4), np.int32)
        weights = np.zeros((n * n, 4), np.float32)
        weights[:, 0] = 1.0
        return pos, np.array(tris, np.int32), uv, joints, weights

    def test_a_small_slit_in_the_hand_is_filled_with_one_texel_and_the_same_winding(self):
        pos, tris, uv, joints, weights = self.sheet()
        before = len(boundary_loops(pos, tris)[0])
        p2, t2, uv2, j2, w2, filled = close_hand_holes(pos, tris, uv, joints, weights, ["hand_r"] + ["x"] * 3)
        self.assertEqual(filled, 1)
        loops, _ = boundary_loops(p2, t2)
        self.assertEqual(len(loops), before - 1)                                          # only the outer rim of the sheet is left
        new = np.arange(len(pos), len(p2))
        self.assertEqual(len(np.unique(uv2[new], axis=0)), 1)                             # one texel: not a blend between islands
        np.testing.assert_allclose(w2.sum(axis=1), 1.0, atol=1e-6)
        normals = np.cross(p2[t2[:, 1]] - p2[t2[:, 0]], p2[t2[:, 2]] - p2[t2[:, 0]])
        self.assertTrue((normals[:, 2] > 0).all(), "the patch must wind like the sheet around it")

    def test_holes_elsewhere_and_big_ones_stay_open(self):
        pos, tris, uv, joints, weights = self.sheet()
        _, _, _, _, _, filled = close_hand_holes(pos, tris, uv, joints, weights, ["spine_01"] + ["x"] * 3)     # the body: not a hand
        self.assertEqual(filled, 0)
        pos, tris, uv, joints, weights = self.sheet(n=14, hole=tuple((i, j) for i in range(3, 11) for j in range(3, 11)))
        self.assertEqual(close_hand_holes(pos, tris, uv, joints, weights, ["hand_r"] + ["x"] * 3)[-1], 0)        # a cuff-sized opening

    def test_a_uv_seam_is_not_a_hole(self):
        pos, tris, uv, joints, weights = self.sheet(hole=())
        n = 6
        pos2 = np.vstack([pos, pos[[2 * n + 2, 2 * n + 3]]])                              # copies of two vertices: a seam in the sheet
        uv2 = np.vstack([uv, uv[[2 * n + 2, 2 * n + 3]] + 0.5]).astype(np.float32)
        out = close_hand_holes(pos2, tris, uv2, np.vstack([joints, joints[:2]]), np.vstack([weights, weights[:2]]), ["hand_r"] + ["x"] * 3)
        self.assertEqual(out[-1], 0)

    def test_the_imported_model_has_no_open_slit_in_a_hand(self):
        """Every small boundary loop of the shipped model that sits on a hand is closed; what is left (cuffs, collar, belt openings) is
        bigger than 6 cm or not on a hand."""
        mesh = read_glb(PROFILE / UNI["mesh"]["asset"])
        loops, group = boundary_loops(mesh.positions, mesh.triangles)
        first = np.zeros(group.max() + 1, np.int64)
        first[group] = np.arange(len(group))
        hand_ids = [i for i, n in enumerate(mesh.joint_names) if n.startswith(("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_"))]
        for loop in loops:
            v = first[loop]
            dom = mesh.joints[v][np.arange(len(v)), np.argmax(mesh.weights[v], axis=1)]
            small = len(loop) <= 10 and float(np.ptp(mesh.positions[v], axis=0).max()) < 6.0
            self.assertFalse(small and np.isin(dom, hand_ids).mean() >= 0.6, f"an open slit with {len(loop)} edges is left in a hand")


if __name__ == "__main__":
    unittest.main()


class SkinnedModelTests(unittest.TestCase):
    """A GLB that already carries a UE5-named skeleton and skin keeps its own weights; only the bind pose is fitted."""
    MODEL = PROFILE / "models" / "beta.glb"

    def setUp(self):
        if not self.MODEL.exists():
            self.skipTest("profiles/VN/models/beta.glb not present")
        from tosdg.import_mesh import import_glb, map_model_joints, read_static_glb, skeleton_usable
        self.sk = skeleton()
        self.g = read_static_glb(self.MODEL)
        self.api = (import_glb, map_model_joints, skeleton_usable)

    def test_twist_bones_fold_into_their_base_bone_and_unknown_names_are_reported(self):
        _, mapj, usable = self.api
        m = dict(zip(self.g["joint_names"], mapj(self.g["joint_names"], self.sk, {})))
        self.assertEqual((m["upperarm_twist_01_l"], m["calf_twist_01_r"], m["hand_l"]), ("upperarm_l", "calf_r", "hand_l"))
        self.assertEqual(usable(self.g, self.sk, {}), (True, []))
        renamed = {**self.g, "joint_names": ["mixamorig:" + n for n in self.g["joint_names"]]}
        ok, bad = usable(renamed, self.sk, {})
        self.assertFalse(ok)
        self.assertIn("mixamorig:hand_l", bad)
        self.assertEqual(usable({**renamed, "joint_names": renamed["joint_names"]}, self.sk,
                                {"joint_map": {n: n[len("mixamorig:"):] for n in renamed["joint_names"]}}), (True, []))
        self.assertEqual(usable({**self.g, "joints": None}, self.sk, {})[0], False)       # static model -> rig it instead

    def test_import_keeps_model_weights_and_fits_a_valid_bind_pose(self):
        mesh = self.api[0](self.MODEL, self.sk, {"height_cm": 178.0})
        self.assertEqual(mesh.info["skin"], "model")
        self.assertEqual(check_against_skeleton(mesh, self.sk), [])
        self.assertLess(mesh.info["bind_direction_error_deg"], 5.0)
        self.assertGreater(abs(mesh.info["bind_pose"]["upperarm_l"]["abduction"]), 20)       # the model's A-pose arms
        self.assertAlmostEqual(float(mesh.positions[:, 2].max()), 178.0, delta=15.0)          # still a 178 cm person (+ retarget shift)
        names = mesh.joint_names
        used = {names[i] for i in np.unique(mesh.joints[mesh.weights > 0])}
        self.assertFalse(any("twist" in n for n in used))
        self.assertTrue({"upperarm_l", "hand_r", "pelvis", "index_01_l"} <= used)            # fingers keep their own weights

    def test_a_model_without_a_usable_skeleton_is_rigged_instead(self):
        from unittest import mock
        with mock.patch("tosdg.import_mesh.skeleton_usable", return_value=(False, ["mixamorig:Hips"])):
            mesh = self.api[0](self.MODEL, self.sk, {"height_cm": 178.0})
        self.assertNotIn("skin", mesh.info)                                                   # proximity weights, not the model's own
        self.assertEqual(check_against_skeleton(mesh, self.sk), [])
        np.testing.assert_allclose(mesh.weights.sum(axis=1), 1.0, atol=1e-4)


class BoneNodesWithoutTransformsTests(unittest.TestCase):
    """Some exporters leave the bone nodes without TRS and keep the rest pose only in the inverse bind matrices, in a frame rotated
    against the mesh. Reading such a file must still give the joints their real positions (not all at the origin)."""

    def test_joint_positions_are_rebuilt_from_the_inverse_bind_matrices(self):
        src = PROFILE / "models" / "beta.glb"
        ref = read_static_glb(src)
        g, blob = _load_glb(src)
        nodes, sk = g["nodes"], g["skins"][0]
        W = {j: np.linalg.inv(m.T) for j, m in zip(sk["joints"], _accessor(g, blob, sk["inverseBindMatrices"]).reshape(-1, 4, 4))}
        G = np.eye(4)                                   # armature frame -> mesh frame: (x, y, z) -> (y, z, x)
        G[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
        G[:3, 3] = [0.0, 0.5, -0.05]
        new_ibm = np.stack([np.linalg.inv(np.linalg.inv(G) @ W[j]).T for j in sk["joints"]]).astype("<f4")   # joint in the armature frame
        a = g["accessors"][sk["inverseBindMatrices"]]
        view = g["bufferViews"][a["bufferView"]]
        off = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        blob2 = blob[:off] + new_ibm.tobytes() + blob[off + new_ibm.nbytes:]
        for n in nodes:
            for k in ("translation", "rotation", "scale", "matrix"):
                n.pop(k, None)
        js = json.dumps(g).encode()
        js += b" " * (-len(js) % 4)
        total = 12 + 8 + len(js) + 8 + len(blob2)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "stripped.glb"
            f.write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                          + struct.pack("<II", len(blob2), 0x004E4942) + blob2)
            got = read_static_glb(f)
        self.assertEqual(got["joint_names"], ref["joint_names"])
        err = np.linalg.norm(got["joint_pos"] - ref["joint_pos"], axis=1)
        self.assertLess(float(np.median(err)), 0.06)    # metres; the translation is matched to the weighted vertices, not exact
        self.assertGreater(float(np.ptp(got["joint_pos"], axis=0).max()), 0.5)


class SmoothNormalsTests(unittest.TestCase):
    def test_vertices_split_along_a_seam_share_one_normal(self):
        """Two triangles meeting at a 90-degree fold, each with its own copy of the two shared vertices (a UV seam)."""
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]], float)
        tris = np.array([[0, 1, 2], [3, 5, 4]])
        fn = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
        n = smooth_normals(p, tris, fn)
        np.testing.assert_allclose(n[0], n[3])
        np.testing.assert_allclose(n[1], n[4])
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0)
        self.assertGreater(float(n[0] @ n[2]), 0.5)          # the tip vertices keep their own face's side of the fold


class HandRollTests(unittest.TestCase):
    """A model whose palms face another way than the skeleton's (fingers spread along another axis) is bound with the roll fitted."""

    def model_pos(self, sk, pose):
        fk = sk.fk(pose)
        return {n: p.copy() for n, p in fk.positions.items()}

    def test_forearm_roll_is_recovered_from_the_finger_spread(self):
        sk = skeleton()
        truth = {"upperarm_r": {"abduction": 40.0, "twist": -40.0}, "lowerarm_r": {"pronation": -35.0}}
        pose, err = fit_bind_pose(sk, self.model_pos(sk, truth))
        fk, ref = sk.fk(pose), sk.fk(truth)
        spread = lambda f: f.positions["index_01_r"] - f.positions["pinky_01_r"]
        u, v = spread(fk), spread(ref)
        self.assertLess(float(np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))), 3.0)
        self.assertLess(err, 3.0)

    def test_tripo_skeleton_takes_the_rig_joints_and_keeps_the_proxy_dofs(self):
        tripo, proxy = skeleton("tripo_v1"), skeleton()
        self.assertEqual(list(tripo.joints), list(proxy.joints))
        for n, j in proxy.joints.items():                                   # poses and limits keep their meaning
            self.assertEqual(tripo.joints[n].parent, j.parent)
            self.assertEqual([(d["name"], d["axis"], d["sign"], d["min"], d["max"]) for d in tripo.joints[n].dofs],
                             [(d["name"], d["axis"], d["sign"], d["min"], d["max"]) for d in j.dofs])
        arm = lambda sk, a, b: float(np.linalg.norm(sk.joints[b].offset))
        self.assertLess(arm(tripo, "upperarm_r", "lowerarm_r"), arm(proxy, "upperarm_r", "lowerarm_r") - 2.0)    # the Tripo arm is shorter than the proxy's
        moved = sum(float(np.linalg.norm(np.asarray(tripo.joints[n].offset) - np.asarray(proxy.joints[n].offset))) for n in proxy.joints)
        self.assertGreater(moved, 10.0)                                                            # the joints are the rig's, not proxy_v1's
        for n, j in tripo.joints.items():                                                           # symmetric: left = mirror of right
            if n.endswith("_r"):
                r, l = j.offset, tripo.joints[n[:-2] + "_l"].offset
                np.testing.assert_allclose([r[0], r[1], r[2]], [l[0], -l[1], l[2]], atol=1e-6, err_msg=n)
            elif not n.endswith("_l"):
                self.assertAlmostEqual(float(j.offset[1]), 0.0, places=6, msg=n)

    def test_wrist_drop_moves_only_the_wrist_pivot_along_the_forearm(self):
        sk = skeleton()
        pos = {n: p.copy() for n, p in sk.fk({}).positions.items()}
        out = adjust_model_joints(pos, {"wrist_drop_cm": 3.0})
        for n in pos:
            if n.startswith("hand_"):
                self.assertAlmostEqual(float(np.linalg.norm(out[n] - pos[n])), 3.0, places=6)
                self.assertGreater(float((out[n] - pos[n]) @ (pos[n] - pos["lowerarm_" + n[-1]])), 0.0)      # towards the hand
            else:
                np.testing.assert_array_equal(out[n], pos[n])
        same = adjust_model_joints(pos, {})
        self.assertEqual(sorted(same), sorted(pos))
        for n in pos:
            np.testing.assert_array_equal(same[n], pos[n])

    def test_shoulder_out_and_forearm_scale_move_the_right_joints(self):
        sk = skeleton()
        pos = {n: p.copy() for n, p in sk.fk({}).positions.items()}
        out = adjust_model_joints(pos, {"shoulder_out_cm": 5.0, "forearm_scale": 0.8})
        for side, sign in (("l", -1.0), ("r", 1.0)):
            d = out[f"upperarm_{side}"] - pos[f"upperarm_{side}"]
            self.assertAlmostEqual(float(np.linalg.norm(d)), 5.0, places=6)
            self.assertAlmostEqual(float(d[2]), 0.0, places=9)                           # horizontal, outwards
            self.assertGreater(float(d[1] * sign), 4.9)
            e, w = pos[f"lowerarm_{side}"], pos[f"hand_{side}"]
            np.testing.assert_allclose(out[f"hand_{side}"], e + 0.8 * (w - e), atol=1e-9)
            delta = out[f"hand_{side}"] - pos[f"hand_{side}"]
            np.testing.assert_allclose(out[f"pinky_03_{side}"] - pos[f"pinky_03_{side}"], delta, atol=1e-9)   # the hand moves whole
            np.testing.assert_array_equal(out[f"lowerarm_{side}"], pos[f"lowerarm_{side}"])           # the elbow stays
        np.testing.assert_array_equal(out["head"], pos["head"])

    def test_forearm_scale_moves_the_sleeve_evenly_and_the_hand_whole(self):
        sk = skeleton()
        names = list(sk.joints)
        pos = {n: p.copy() for n, p in sk.fk({}).positions.items()}
        e, w = pos["lowerarm_r"], pos["hand_r"]
        pts = np.array([e + 0.5 * (w - e), w + [0, 0, -5.0], pos["spine_02"]])            # mid-forearm, in the hand, on the chest
        joints = np.zeros((3, 4), np.int64)
        joints[:, 0] = [names.index("lowerarm_r"), names.index("hand_r"), names.index("spine_02")]
        weights = np.zeros((3, 4))
        weights[:, 0] = 1.0
        moved = scale_forearm_vertices(pts, joints, weights, names, pos, {"forearm_scale": 0.8})
        np.testing.assert_allclose(moved[0] - pts[0], -0.2 * 0.5 * (w - e), atol=1e-9)
        np.testing.assert_allclose(moved[1] - pts[1], -0.2 * (w - e), atol=1e-9)
        np.testing.assert_array_equal(moved[2], pts[2])
