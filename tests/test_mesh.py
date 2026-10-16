"""Skinned uniform mesh, GLB asset, software mesh renderer, brightness jitter, and the 'mesh' backend end to end.

Contents:
  uniform_entry()
  make_cam()
  pose_of()
  MeshAssetTests (18 tests)
  SpecTests (2 tests)
  RasterizerTests (3 tests)
  RenderTests (8 tests)
  PipelineTests (6 tests)
  SchemaTests (2 tests)
"""
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from helpers import ROOT, make_config, poses, skeleton
from tosdg import rng as rngmod
from tosdg.camera import Camera, look_at_rotator
from tosdg.generate import generate_sample, run
from tosdg.poses import merge
from tosdg.render_mesh import material_albedo, rasterize, render, render_layer, skin_mesh
from tosdg.rng import derive_sample_seed
from tosdg.scene import GenerationContext, build_scene
from tosdg.schema import load_json, schema_errors, validate_instance
from tosdg.uniform_mesh import (GLTF_SCALE, build_uniform_mesh, check_against_skeleton, from_gltf, read_glb, spec_sha256, to_gltf,
                                write_glb)
from tosdg.validate import check_metadata, validate_dataset

PROFILE = ROOT / "profiles" / "VN"
UNI_ID = "beta"        # the profile's model: a textured Tripo person model on the tripo_v1 skeleton (the only uniform, weight 1)
DEFAULT_ID = UNI_ID
LIGHT = {"sun_direction": [0.3, -0.2, 0.93], "sun_color_rgb": [1, 1, 1], "sun_intensity": 0.8, "ambient_intensity": 0.4, "exposure_ev": 0.0}


def uniform_entry():
    return next(u for u in load_json(PROFILE / "uniforms.json", "uniforms")["uniforms"] if u["id"] == UNI_ID)


def make_cam(az=0.0, dist=450.0, w=320, h=480, f=35.0):
    eye = np.array([dist * np.cos(np.radians(az)), dist * np.sin(np.radians(az)), 110.0])
    p, y = look_at_rotator(eye, [0, 0, 100])
    return Camera("t", tuple(eye), p, y, 0, f, 36, w, h)


def pose_of(name, hand="relaxed"):
    pl = poses()
    return merge(pl.body_pose(name), pl.hand_pose(hand, "l"), pl.hand_pose(hand, "r"))


class MeshAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sk = skeleton("tripo_v1")
        cls.uni = uniform_entry()
        cls.asset = PROFILE / cls.uni["mesh"]["asset"]
        cls.mesh = read_glb(cls.asset)

    def test_committed_asset_is_up_to_date_with_its_fit_and_source(self):
        fit = json.loads((PROFILE / self.uni["mesh"]["fit"]).read_text(encoding="utf-8"))
        src = PROFILE / fit["source"]["name"]
        if src.exists():
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), fit["source"]["sha256"], "model changed: run tosdg import-model")
        self.assertEqual(self.mesh.info["fit_sha256"], spec_sha256(fit), "fit changed: run tosdg import-model")

    def test_fits_the_skeleton(self):
        self.assertEqual(check_against_skeleton(self.mesh, self.sk), [])
        self.assertEqual(self.mesh.joint_names, list(self.sk.joints))

    def test_skin_weights(self):
        m = self.mesh
        np.testing.assert_allclose(m.weights.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue((m.weights >= 0).all() and m.joints.max() < len(m.joint_names))
        self.assertEqual(m.joints.shape[1], 4)

    def test_glb_uses_gltf_conventions(self):
        """+Y up, the asset's front faces +Z, metres, right-handed (so the winding is flipped w.r.t. Unreal)."""
        np.testing.assert_allclose(to_gltf([1, 0, 0]), [0, 0, GLTF_SCALE])      # forward -> +Z
        np.testing.assert_allclose(to_gltf([0, 0, 1]), [0, GLTF_SCALE, 0])      # up -> +Y
        np.testing.assert_allclose(to_gltf([0, 1, 0]), [-GLTF_SCALE, 0, 0])     # right -> -X (viewer's right is +X)
        pts = np.random.default_rng(0).normal(size=(5, 3)) * 50
        np.testing.assert_allclose(from_gltf(to_gltf(pts)), pts, atol=1e-9)
        raw = self.asset.read_bytes()
        jlen = struct.unpack_from("<I", raw, 12)[0]
        g = json.loads(raw[20:20 + jlen])
        pos = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        lo, hi = pos["min"], pos["max"]
        self.assertGreater(hi[2], 0.0)                    # the model faces +Z: its front extent is positive
        self.assertTrue(-0.08 < lo[1] < 0.03 and 1.7 < hi[1] < 1.95, (lo, hi))   # standing on the floor (soles may sit a few cm low: the model's legs are shorter than the skeleton's), ~1.8 m tall
        self.assertLess(abs(lo[0] + hi[0]), 0.05)          # left/right balanced about the body axis (the radio and holsters hang on one side, so not exact)
        self.assertEqual(g["asset"]["version"], "2.0")
        self.assertEqual(len(g["skins"][0]["joints"]), len(self.sk.joints))

    def test_triangles_wind_outward(self):
        m = self.mesh
        v = m.positions
        fn = np.cross(v[m.triangles[:, 1]] - v[m.triangles[:, 0]], v[m.triangles[:, 2]] - v[m.triangles[:, 0]])
        vn = m.normals[m.triangles].mean(axis=1)
        agree = (fn * vn).sum(axis=1) > 0
        self.assertGreater(agree.mean(), 0.97)             # face winding agrees with the (area-weighted) vertex normals

    def test_every_mesh_region_has_a_colour_and_cloth_is_flagged(self):
        self.assertTrue(all(m.get("texture") is not None for m in self.mesh.materials))      # the model's own texture is the appearance
        self.assertTrue(self.mesh.vertex_jitter is not None and 0.3 < float(self.mesh.vertex_jitter.mean()) < 1.0)   # cloth vs skin texels

    def test_glb_round_trip_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "a.glb"
            write_glb(self.mesh, self.sk, out)
            m2 = read_glb(out)
        np.testing.assert_allclose(m2.positions, self.mesh.positions, atol=1e-4)
        np.testing.assert_array_equal(m2.triangles, self.mesh.triangles)
        np.testing.assert_array_equal(m2.joints, self.mesh.joints)
        self.assertEqual(m2.materials, self.mesh.materials)

    def test_rest_pose_skinning_is_identity_and_bending_moves_the_right_vertices(self):
        sk, m = self.sk, self.mesh
        bind = m.info["bind_pose"]                                  # the model is bound in its own rest pose (A-pose), not the skeleton's
        pos, nrm = skin_mesh(m, sk.fk(bind))
        np.testing.assert_allclose(pos, m.positions, atol=1e-4)
        pose = pose_of("stop_bent_r")
        for j in ("thigh_l", "calf_l", "foot_l"):
            pose[j] = dict(bind.get(j, {}))                          # keep the left leg exactly in its bind pose
        up, _ = skin_mesh(m, sk.fk(pose))
        dom = m.joints[np.arange(m.n_vertices), m.weights.argmax(axis=1)]
        names = np.array(m.joint_names)[dom]
        right_arm = np.array([n in ("upperarm_r", "lowerarm_r", "hand_r") or "_r" in n and any(t in n for t in ("index", "middle", "ring", "pinky", "thumb")) for n in names])
        left_leg = np.array([n in ("thigh_l", "calf_l", "foot_l") for n in names]) & (m.weights.max(axis=1) > 0.999)    # fully owned by the (unmoved) left leg
        self.assertGreater(right_arm.sum(), 100)
        self.assertGreater(up[right_arm, 2].max(), m.positions[right_arm, 2].max() + 25)    # the raised arm goes up
        np.testing.assert_allclose(up[left_leg], m.positions[left_leg], atol=1e-4)         # the left leg does not move
        np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)

    # largest allowed distance (cm) from a rest-pose keypoint to the nearest mesh vertex. Joints sit on the axis of a tube, so they are
    # as far from the surface vertices as the tube is wide; face points must be close to the head surface. Measured values are
    # ~0.5 cm (face), 2 (wrists, ankles), 5 (elbows, knees), 7 (shoulders: the Tripo rig's joint is the humeral head, deep in the jacket), 9-10 (hips); the limits leave a margin.
    KEYPOINT_LIMIT = {"nose": 3.5, "left_eye": 3.5, "right_eye": 3.5, "left_ear": 3.5, "right_ear": 3.5, "left_shoulder": 8.0, "right_shoulder": 8.0,
                      "left_wrist": 3.0, "right_wrist": 3.0, "left_ankle": 5.0, "right_ankle": 5.0, "left_elbow": 7.0, "right_elbow": 7.0,
                      "left_hip": 10.0, "right_hip": 10.0, "left_knee": 10.0, "right_knee": 10.0}

    def test_body_keypoints_lie_inside_the_mesh_volume(self):
        names, pts = self.sk.keypoints(self.sk.fk(self.mesh.info["bind_pose"]), "coco17")
        for n, p in zip(names, pts):
            d = np.linalg.norm(self.mesh.positions - p, axis=1).min()
            self.assertLess(d, self.KEYPOINT_LIMIT[n], f"{n} is {d:.1f} cm from the nearest mesh vertex")

    def test_face_keypoints_sit_on_the_face(self):
        """Measured on the mesh (import_mesh.head_landmarks), not proxy_v1's offsets, which put them 2-5 cm too high on this head."""
        fk = self.sk.fk(self.mesh.info["bind_pose"])
        names, pts = self.sk.keypoints(fk, "coco17")
        kp = dict(zip(names, pts))
        for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            self.assertLess(float(np.linalg.norm(self.mesh.positions - kp[n], axis=1).min()), 1.0, n)   # on the surface
        for eye in ("left_eye", "right_eye"):
            self.assertTrue(2.0 < kp[eye][2] - kp["nose"][2] < 5.0, eye)              # eyes 2-5 cm above the nose tip
            self.assertLess(kp[eye][0], kp["nose"][0])                                 # and behind it
        for ear, eye in (("left_ear", "left_eye"), ("right_ear", "right_eye")):
            self.assertGreater(abs(kp[ear][1]) - abs(kp[eye][1]), 4.0)                  # ears at the side of the head
        top = float(skin_mesh(self.mesh, fk)[0][:, 2].max())
        cap = self.sk.segments["cap"]
        self.assertAlmostEqual(float(fk.positions[cap["to"]][2]) + cap["radius"], top, delta=1.0)   # the cap capsule ends at the cap

    def test_limb_proportions_are_anatomical(self):
        """Forearm 0.7-0.95 of the upper arm (adult average ~0.8), shank about the thigh, and a shoulder-to-shoulder width of an adult."""
        P = self.sk.fk(self.mesh.info["bind_pose"]).positions
        d = lambda a, b: float(np.linalg.norm(P[b] - P[a]))
        self.assertTrue(0.7 < d("lowerarm_r", "hand_r") / d("upperarm_r", "lowerarm_r") < 0.95)
        self.assertTrue(0.8 < d("calf_r", "foot_r") / d("thigh_r", "calf_r") < 1.15)
        self.assertTrue(26.0 < d("upperarm_l", "upperarm_r") < 46.0, d("upperarm_l", "upperarm_r"))
        for a, b in (("upperarm_l", "upperarm_r"), ("lowerarm_l", "lowerarm_r"), ("hand_l", "hand_r"), ("thigh_l", "thigh_r")):
            self.assertAlmostEqual(P[a][0], P[b][0], delta=1.0, msg=f"{a}/{b} front-back")       # left/right at the same depth and height
            self.assertAlmostEqual(P[a][2], P[b][2], delta=2.0, msg=f"{a}/{b} height")     # the model's own arms are posed a little unevenly

    def test_arms_do_not_sink_far_into_the_torso_when_standing_at_attention(self):
        """The sleeves touch the jacket, but must not pass through it: an arm vertex deeper than 5 cm inside the torso surface means the
        shoulder pivots are wrong (an auto-rig puts them inside the chest; the fit's shoulder_out_cm corrects that)."""
        pos, nrm = skin_mesh(self.mesh, self.sk.fk(pose_of("neutral_stand")))
        names = self.mesh.joint_names

        def weight_on(prefixes):
            ids = [i for i, n in enumerate(names) if n.startswith(prefixes)]
            return (self.mesh.weights * np.isin(self.mesh.joints, ids)).sum(axis=1)
        arm = np.nonzero(weight_on(("upperarm_", "lowerarm_", "hand_", "thumb", "index", "middle", "ring", "pinky")) > 0.6)[0]
        torso = np.nonzero(weight_on(("pelvis", "spine", "clavicle", "neck")) > 0.6)[0]
        depth = 0.0
        for s in range(0, len(arm), 400):
            a = pos[arm[s:s + 400]]
            d2 = ((a[:, None, :] - pos[torso][None]) ** 2).sum(axis=2)
            i = d2.argmin(axis=1)
            near = d2[np.arange(len(a)), i] < 36.0                                   # within 6 cm of the torso
            inside = -((a - pos[torso][i]) * nrm[torso][i]).sum(axis=1)               # how far behind the torso surface
            if near.any():
                depth = max(depth, float(inside[near].max()))
        self.assertLess(depth, 5.0, f"an arm vertex is {depth:.1f} cm inside the torso")

    def test_raising_an_arm_does_not_drag_the_belt_gear_with_it(self):
        """An auto-rig hands the holster and jacket hem beside the lowered arm a share of the forearm; raised, the arm tore the pistol
        holster into shreds. Vertices below the armpit whose main bone is not an arm bone carry no arm weight, and no belt-level
        triangle is stretched by more than 3x when the right arm goes straight up."""
        m, names = self.mesh, self.mesh.joint_names
        arm = np.array([n.startswith(("upperarm_", "lowerarm_", "hand_", "thumb_", "index_", "middle_", "ring_", "pinky_")) for n in names])
        bind = self.sk.fk(m.info["bind_pose"])
        z_cut = min(bind.positions["upperarm_l"][2], bind.positions["upperarm_r"][2]) - 18.0       # the importer cuts 15 cm under the shoulder; the mesh was then moved a few cm
        rest, _ = skin_mesh(m, bind)
        dominant = m.joints[np.arange(m.n_vertices), np.argmax(m.weights, axis=1)]
        arm_share = (m.weights * arm[m.joints]).sum(axis=1)
        self.assertEqual(int(((~arm[dominant]) & (arm_share > 1e-6) & (rest[:, 2] < z_cut)).sum()), 0)
        legs = {j: d for j, d in m.info["bind_pose"].items() if j.startswith(("thigh_", "calf_", "foot_"))}
        posed, _ = skin_mesh(m, self.sk.fk({**pose_of("arm_vertical_r"), **legs}))          # only the arm moves (the hips stay as bound)
        T = m.triangles[(rest[m.triangles][:, :, 2].max(axis=1) < 108.0) & (rest[m.triangles][:, :, 2].min(axis=1) > 88.0)]
        edge = lambda P: np.linalg.norm(P[T[:, [0, 1, 2]]] - P[T[:, [1, 2, 0]]], axis=2)
        stretch = (edge(posed) / np.maximum(edge(rest), 1e-6)).max()
        self.assertLess(float(stretch), 3.0, f"a belt-level triangle is stretched {stretch:.1f}x by raising the arm")

    def test_the_soles_are_flat_at_the_rest_pose(self):
        """The model stands flat in its own pose, so the skeleton's rest pose must too (the foot was bound 6 deg off: toes up, on the heels)."""
        pos, _ = skin_mesh(self.mesh, self.sk.fk({}))
        names = np.array(self.mesh.joint_names)
        dom = names[self.mesh.joints[np.arange(self.mesh.n_vertices), self.mesh.weights.argmax(axis=1)]]
        for s in "lr":
            f = pos[np.isin(dom, [f"foot_{s}", f"ball_{s}"])]
            toe = f[f[:, 0] > f[:, 0].max() - 3][:, 2].min()
            heel = f[f[:, 0] < f[:, 0].min() + 3][:, 2].min()
            self.assertLess(abs(toe - heel), 1.5, f"{s}: toe {toe:.1f} cm, heel {heel:.1f} cm")

    def test_forearm_roll_winds_up_along_the_forearm(self):
        """Pronation turns the wrist end of the forearm fully and the elbow end not at all (no candy-wrapper pinch at the elbow)."""
        from tosdg.render_mesh import TWIST_CHILD, _twist_fractions
        m = self.mesh
        bind = m.info["bind_pose"]
        i, h = m.joint_names.index("lowerarm_r"), m.joint_names.index("hand_r")
        a, b = m.bind_positions[i], m.bind_positions[h]
        t = (m.positions - a) @ (b - a) / ((b - a) @ (b - a))
        share = (m.weights * (m.joints == i)).sum(axis=1)
        near = np.nonzero((share > 0.9) & (t > 0.0) & (t < 0.12))[0]              # the elbow end of the sleeve
        far = np.nonzero((share > 0.95) & (t > 0.6) & (t < 0.85))[0]              # towards the cuff (the cuff itself shares the hand's weight)
        self.assertTrue(len(near) and len(far))
        turned = {**bind, "lowerarm_r": {**bind.get("lowerarm_r", {}), "pronation": bind["lowerarm_r"]["pronation"] + 60.0}}
        p0, _ = skin_mesh(m, self.sk.fk(bind))
        p1, _ = skin_mesh(m, self.sk.fk(turned))
        self.assertLess(float(np.linalg.norm(p1[near] - p0[near], axis=1).max()), 0.8)        # the elbow end stays
        self.assertGreater(float(np.linalg.norm(p1[far] - p0[far], axis=1).mean()), 1.0)       # the wrist end turns with the hand
        self.assertIn("lowerarm_r", TWIST_CHILD)
        self.assertEqual(len(_twist_fractions(m)), 6)

    def test_the_baked_armpit_shadow_is_taken_out_of_the_texture(self):
        """The A-pose model has a dark brown crease painted under each arm; even_armpits brings it to the cloth colour."""
        m = self.mesh
        self.assertGreater(m.info.get("armpit_texels_evened", 0), 10000)
        tex = m.textures[0].astype(float)
        bind = self.sk.fk(m.info["bind_pose"])
        for s in "lr":
            ax = bind.positions[f"upperarm_{s}"] - np.array([0.0, 0.0, 8.0])
            d = np.linalg.norm(m.positions - ax, axis=1)
            px = lambda sel: tex[np.clip((m.uv[sel, 1] * (tex.shape[0] - 1)).round().astype(int), 0, tex.shape[0] - 1),
                                 np.clip((m.uv[sel, 0] * (tex.shape[1] - 1)).round().astype(int), 0, tex.shape[1] - 1)]
            core, ring = px(d < 8), px((d > 16) & (d < 24))
            cloth = lambda c: c[(c[:, 0] > c[:, 1]) & (c[:, 1] > c[:, 2]) & (c[:, 0] - c[:, 2] > 25)]
            self.assertGreater(np.percentile(cloth(core) @ [0.299, 0.587, 0.114], 10), 0.7 * np.median(cloth(ring) @ [0.299, 0.587, 0.114]), s)

    def test_hanging_wrist_is_at_belt_height(self):
        """Standing at attention the wrist is at the belt, a little above the hip joint (the Tripo forearm was too long)."""
        fk = self.sk.fk(pose_of("neutral_stand"))
        for side in ("l", "r"):
            above_hip = float(fk.positions[f"hand_{side}"][2] - fk.positions[f"thigh_{side}"][2])
            self.assertTrue(4.0 < above_hip < 12.0, (side, above_hip))

    def test_the_model_is_a_standing_adult(self):
        whole = self.mesh.positions.max(axis=0) - self.mesh.positions.min(axis=0)
        self.assertTrue(165 < whole[2] < 195, whole)
        self.assertAlmostEqual(float(self.mesh.positions[:, 2].min()), 0.0, delta=8.0)      # stands on the floor


class SpecTests(unittest.TestCase):
    def spec(self, parts):
        return {"schema_version": "1.0", "id": "t", "character": "proxy_v1", "parts": parts}

    def test_for_sides_mirrors_the_left_copy(self):
        part = {"id": "b", "for_sides": ["l", "r"], "type": "box", "material": "metal", "center": ["pelvis", [0, 5, 0]], "size": [2, 2, 2], "skin": {"joint": "pelvis"}}
        m = build_uniform_mesh(self.spec([part]), skeleton())
        self.assertEqual(m.info["parts"], ["b_l", "b_r"])
        half = m.n_vertices // 2
        self.assertAlmostEqual(m.positions[:half, 1].mean(), -5.0, places=6)    # left = mirrored across Y
        self.assertAlmostEqual(m.positions[half:, 1].mean(), 5.0, places=6)

    def test_bad_specs_are_rejected(self):
        sk = skeleton()
        with self.assertRaises(ValueError):
            build_uniform_mesh({**self.spec([]), "schema_version": "9.9"}, sk)
        with self.assertRaises(ValueError):
            build_uniform_mesh({**self.spec([]), "character": "other"}, sk)
        with self.assertRaises(ValueError):
            build_uniform_mesh(self.spec([{"id": "x", "type": "box", "material": "m", "center": ["no_such_joint", [0, 0, 0]], "size": [1, 1, 1], "skin": {"joint": "pelvis"}}]), sk)


class RasterizerTests(unittest.TestCase):
    @staticmethod
    def tri_setup(tris_xy, inv_z):
        v = np.array([p for t in tris_xy for p in t], float)
        iz = np.repeat(np.array(inv_z, float), 3)
        return v[:, 0], v[:, 1], iz, np.arange(len(v)).reshape(-1, 3)

    def test_nearer_triangle_wins_regardless_of_draw_order(self):
        big = [(2, 2), (30, 2), (2, 30)]
        for order in ((0, 1), (1, 0)):
            tris = [big, big]
            iz = [0.5, 1.0]                       # triangle 1 is nearer (larger 1/z)
            sx, sy, izv, t = self.tri_setup(tris, iz)
            tid, _ = rasterize(sx, sy, izv, t[list(order)], 32, 32)
            winner_idx = np.where(tid[10, 10] == np.array([0, 1]))[0]
            self.assertEqual(order[int(winner_idx[0])], 1)

    def test_coverage_equals_area_and_clipping_is_safe(self):
        sx, sy, iz, t = self.tri_setup([[(1, 1), (21, 1), (1, 21)]], [1.0])
        tid, bary = rasterize(sx, sy, iz, t, 40, 40)
        self.assertAlmostEqual(float((tid >= 0).sum()), 200.0, delta=14)       # area 20*20/2
        self.assertTrue(np.allclose(bary[tid >= 0].sum(axis=1), 1.0, atol=1e-5))
        sx, sy, iz, t = self.tri_setup([[(-50, -50), (90, -30), (10, 120)]], [1.0])   # larger than the image and than the bucket size
        tid, _ = rasterize(sx, sy, iz, t, 40, 40)
        self.assertTrue((tid >= 0).all())

    def test_degenerate_and_offscreen_triangles_are_ignored(self):
        sx, sy, iz, t = self.tri_setup([[(5, 5), (5, 5), (5, 5)], [(100, 100), (120, 100), (100, 120)]], [1.0, 1.0])
        tid, _ = rasterize(sx, sy, iz, t, 40, 40)
        self.assertTrue((tid < 0).all())


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sk = skeleton("tripo_v1")
        cls.uni = uniform_entry()
        cls.mesh = read_glb(PROFILE / cls.uni["mesh"]["asset"])
        cls.bg = Image.new("RGB", (320, 480), (140, 145, 150))

    def draw(self, pose, cam, brightness=1.0):
        return render(self.mesh, self.sk.fk(pose), cam, self.bg, self.uni["proxy_colors"], LIGHT, brightness)

    def test_the_asset_carries_the_models_normal_and_roughness_maps_and_the_renderer_uses_them(self):
        """Tripo's normal and metallic/roughness maps are part of the imported asset; the pockets, seams and folds come from them."""
        m, bg = self.mesh, self.bg
        mat = m.materials[0]
        self.assertIsNotNone(mat.get("normal_texture"))
        self.assertIsNotNone(mat.get("mr_texture"))
        self.assertEqual(m.textures[mat["normal_texture"]].shape[2], 3)
        cam = make_cam(20)
        pose = pose_of("neutral_stand")
        flat = type(m)(**{**m.__dict__, "materials": [{k: v for k, v in mat.items() if k not in ("normal_texture", "mr_texture")}], "cache": {}})
        with_map = render(m, self.sk.fk(pose), cam, bg, self.uni["proxy_colors"], LIGHT)[0]
        without = render(flat, self.sk.fk(pose), cam, bg, self.uni["proxy_colors"], LIGHT)[0]
        changed = (np.abs(with_map.astype(int) - without.astype(int)).max(axis=2) > 5).sum()
        self.assertGreater(int(changed), 100)                                                            # the map visibly changes the shading

    def test_normals_are_skinned_in_the_bind_frame(self):
        """The model is bound in an A-pose: skinning with the bind pose itself must give back the stored normals (not normals rotated by the bind rotations)."""
        _, nrm = skin_mesh(self.mesh, self.sk.fk(self.mesh.info["bind_pose"]))
        np.testing.assert_allclose(nrm, self.mesh.normals, atol=2e-3)

    def test_deterministic_and_reasonably_fast(self):
        pose, cam = pose_of("palms_forward_both"), make_cam(30)
        t = time.time()
        a, ma = self.draw(pose, cam)
        self.assertLess(time.time() - t, 3.0)
        b, mb = self.draw(pose, cam)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ma, mb)
        self.assertGreater(int(ma.sum()), 2000)

    def test_mask_is_the_exact_silhouette_of_the_projected_mesh(self):
        pose, cam = pose_of("stop_bent_r"), make_cam(40)
        rgb, mask = self.draw(pose, cam)
        pos, _ = skin_mesh(self.mesh, self.sk.fk(pose))
        uv, depth = cam.project(pos)
        ys, xs = np.nonzero(mask)
        self.assertLess(abs(xs.min() - uv[:, 0].min()), 3)
        self.assertLess(abs(xs.max() + 1 - uv[:, 0].max()), 3)
        self.assertLess(abs(ys.min() - uv[:, 1].min()), 3)
        self.assertLess(abs(ys.max() + 1 - uv[:, 1].max()), 3)
        self.assertTrue((rgb[mask].std(axis=0) > 3).all())          # shaded, not a flat silhouette

    def test_in_frame_keypoints_fall_inside_the_mask_for_every_base_pose_and_side(self):
        """The generator rejects a sample whose keypoint is outside the mask bbox; the mesh must not be thinner than the skeleton."""
        bad = []
        for pose_name in sorted(poses().body):
            pose = pose_of(pose_name, "open_flat")
            fk = self.sk.fk(pose)
            for az in (30, 210):                                                              # one view of each side of the body
                cam = make_cam(az)
                _, mask = self.draw(pose, cam)
                ys, xs = np.nonzero(mask)
                x0, x1, y0, y1 = xs.min() - 2, xs.max() + 2, ys.min() - 2, ys.max() + 2      # the validator's 2 px tolerance
                names, pts = self.sk.keypoints(fk, "coco17")
                uv, depth = cam.project(pts)
                for n, p in zip(names, uv):
                    if 0 <= p[0] <= cam.width and 0 <= p[1] <= cam.height and not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
                        bad.append((pose_name, az, n))
        self.assertEqual(bad, [])

    def test_in_frame_wrists_and_nose_touch_the_mesh_itself(self):
        pose = pose_of("palms_forward_both", "open_flat")
        fk = self.sk.fk(pose)
        names, pts = self.sk.keypoints(fk, "coco17")
        for az in (0, 180):
            cam = make_cam(az)
            _, mask = self.draw(pose, cam)
            uv, _ = cam.project(pts)
            for n in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_elbow", "right_elbow", "nose"):
                x, y = uv[names.index(n)]
                win = mask[max(int(y) - 4, 0):int(y) + 5, max(int(x) - 4, 0):int(x) + 5]
                self.assertTrue(win.any(), f"{n} (az {az}) is not near any mesh pixel")

    def test_brightness_scales_only_the_cloth(self):
        base = material_albedo(self.mesh, self.uni["proxy_colors"], 1.0)
        dim = material_albedo(self.mesh, self.uni["proxy_colors"], 0.8)
        for i, m in enumerate(self.mesh.materials):
            if m["jitter"]:
                np.testing.assert_allclose(dim[i], base[i] * 0.8, atol=1e-9)
            else:
                np.testing.assert_array_equal(dim[i], base[i])
        bright = material_albedo(self.mesh, self.uni["proxy_colors"], 3.0)
        self.assertLessEqual(bright.max(), 1.0)              # clipped, never over-exposed
        pose, cam = pose_of("neutral_stand"), make_cam(20)
        a, ma = self.draw(pose, cam, 1.0)
        b, mb = self.draw(pose, cam, 0.7)
        np.testing.assert_array_equal(ma, mb)                 # the silhouette does not depend on colour
        self.assertLess(b[ma].mean(), a[ma].mean() * 0.9)

    def test_layer_composites_back_to_the_full_render_and_has_a_clean_edge(self):
        """The transparent layer the camera view draws must equal the full render: alpha = the anti-aliased coverage, colour not darkened at the edge."""
        for pose_name, az in (("palms_forward_both", 25), ("stop_bent_r", 200)):
            pose, cam = pose_of(pose_name), make_cam(az)
            fk = self.sk.fk(pose)
            plate = Image.new("RGB", (cam.width, cam.height), (96, 128, 160))
            full, mask = render(self.mesh, fk, cam, plate, self.uni["proxy_colors"], LIGHT, 0.9)
            layer = render_layer(self.mesh, fk, cam, self.uni["proxy_colors"], LIGHT, 0.9)
            self.assertEqual((layer.shape, layer.dtype), ((cam.height, cam.width, 4), np.uint8))
            alpha = layer[..., 3].astype(float) / 255.0
            comp = layer[..., :3].astype(float) * alpha[..., None] + np.array([96, 128, 160], float) * (1 - alpha[..., None])
            diff = np.abs(comp - full.astype(float))
            # same picture up to rounding; the few edge pixels of a very bright texture differ more because the full render clips after mixing
            # the background in, the layer before (white parts of the model: albedo x (ambient + sun) above 1)
            self.assertLess(float((diff > 2.5).mean()), 0.002)
            self.assertLess(float(diff.max()), 16.0)
            np.testing.assert_array_equal(alpha >= 0.5, mask)                         # same silhouette as the instance mask
            edge = (alpha > 0.1) & (alpha < 0.9)
            self.assertGreater(int(edge.sum()), 50)                                   # anti-aliased edge exists
            inside = layer[alpha > 0.99][:, :3].astype(float)
            self.assertGreater(layer[edge][:, :3].astype(float).mean(), inside.mean() * 0.6)   # edge colour is not blended with black
            self.assertEqual(int(layer[alpha == 0][:, :3].sum()), 0)                  # nothing outside the officer


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def cfg(self, name="c", **over):
        d = Path(self.td) / name
        d.mkdir(exist_ok=True)
        return make_config(d, **over)

    def test_brightness_has_its_own_stream_and_leaves_every_other_draw_unchanged(self):
        a = GenerationContext(self.cfg("a", uniform_brightness_range=[1.0, 1.0]))
        b = GenerationContext(self.cfg("b"))                      # smoke config: [0.9, 1.1]
        vals = []
        for seed in range(1, 9):
            sa, sb = build_scene(a, seed), build_scene(b, seed)
            self.assertEqual(sa.gesture["id"], sb.gesture["id"])
            self.assertEqual(sa.sampled_pose.pose, sb.sampled_pose.pose)
            self.assertEqual(sa.camera, sb.camera)
            self.assertEqual(sa.background["id"], sb.background["id"])
            self.assertEqual(sa.lighting, sb.lighting)
            self.assertEqual(sa.uniform_id, sb.uniform_id)
            self.assertEqual(sa.uniform_brightness, 1.0)
            self.assertAlmostEqual(sb.uniform_brightness, rngmod.uniform(rngmod.stream(seed, "uniform_brightness"), [0.9, 1.1]))
            self.assertTrue(0.9 <= sb.uniform_brightness <= 1.1)
            vals.append(sb.uniform_brightness)
        self.assertGreater(len(set(np.round(vals, 3))), 5)         # actually random

    def test_factor_is_recorded_in_the_metadata_and_validates(self):
        ctx = GenerationContext(self.cfg())
        res = generate_sample(ctx, 0, derive_sample_seed(ctx.config["master_seed"], 0), 0)
        self.assertTrue(res.ok, res.errors)
        scene = build_scene(ctx, res.meta["seed"]["attempt_seed"])
        self.assertAlmostEqual(res.meta["officer"]["uniform_brightness"], scene.uniform_brightness, places=5)
        self.assertEqual(res.meta["generator"]["backend"], "mesh")
        self.assertEqual(check_metadata(res.meta), [])
        old = json.loads(json.dumps(res.meta))
        del old["officer"]["uniform_brightness"]
        old["schema_version"] = "1.0"
        self.assertEqual(check_metadata(old), [])                   # datasets written before this change still validate

    def test_both_backends_run_end_to_end_validate_and_reproduce_byte_identically(self):
        for backend, over in (("mesh", {}), ("proxy", {"uniform_brightness_range": [1.0, 1.0], "uniform_weights": {UNI_ID: 1.0}})):
            with self.subTest(backend=backend):
                ctx = GenerationContext(self.cfg(backend, backend=backend, num_samples=2, **over))
                outs = []
                for k in range(2):
                    out = Path(self.td) / f"{backend}_{k}"
                    man = run(ctx, out, log=lambda *_: None)
                    self.assertEqual((len(man["samples"]), len(man["failures"])), (2, 0))
                    self.assertEqual(validate_dataset(out), [])
                    outs.append({s["sample_id"]: s["sha256"] for s in man["samples"]})
                self.assertEqual(outs[0], outs[1])

    def test_mesh_images_differ_from_the_capsule_proxy(self):
        ctx_m = GenerationContext(self.cfg("m", num_samples=1, uniform_brightness_range=[1.0, 1.0]))
        ctx_p = GenerationContext(self.cfg("p", num_samples=1, backend="proxy", uniform_brightness_range=[1.0, 1.0]))
        sc = build_scene(ctx_m, 5)
        from tosdg.generate import render_scene
        (rm, mm), (rp, _) = render_scene(ctx_m, sc), render_scene(ctx_p, build_scene(ctx_p, 5))
        self.assertGreater(float(np.abs(rm.astype(int) - rp.astype(int)).mean()), 0.2)
        self.assertGreater(int(mm.sum()), 100)

    def test_config_problems_are_rejected_early_and_clearly(self):
        with self.assertRaisesRegex(ValueError, "mesh.*only"):
            GenerationContext(self.cfg("a", backend="proxy"))                      # jitter range needs the mesh backend
        with self.assertRaisesRegex(ValueError, "lo <= hi"):
            GenerationContext(self.cfg("b", uniform_brightness_range=[1.1, 0.9]))
        with self.assertRaisesRegex(ValueError, "unknown"):
            GenerationContext(self.cfg("c", uniform_weights={"no_such_model": 1.0}))
        for bad in ([0.9], [0.9, 1.1, 1.2], [0.1, 1.0], [0.9, 5.0], "x"):
            self.assertTrue(schema_errors({**json.loads(self.cfg("e").read_text()), "uniform_brightness_range": bad}, "generation_config"), bad)

    def test_stale_or_missing_assets_are_detected_and_hashed(self):
        ctx = GenerationContext(self.cfg())
        self.assertTrue(any(k.replace("\\", "/").endswith("assets/beta.glb") for k in ctx.asset_hashes))
        self.assertTrue(any(k.replace("\\", "/").endswith("assets/beta.fit.json") for k in ctx.asset_hashes))
        root = Path(self.td) / "root"
        for d in ("profiles", "characters", "poses", "schemas"):
            shutil.copytree(ROOT / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
        fit = root / "profiles/VN/assets/beta.fit.json"
        doc = json.loads(fit.read_text(encoding="utf-8"))
        doc["height_cm"] = 170.0
        fit.write_text(json.dumps(doc), encoding="utf-8")
        with mock.patch.dict(os.environ, {"TOSDG_ROOT": str(root)}):
            with self.assertRaisesRegex(ValueError, "stale"):
                GenerationContext(self.cfg("s"))
            (root / "profiles/VN/assets/beta.glb").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                GenerationContext(self.cfg("t"))


class SchemaTests(unittest.TestCase):
    def test_uniform_mesh_field(self):
        doc = load_json(PROFILE / "uniforms.json")
        validate_instance(doc, "uniforms")
        bad = json.loads(json.dumps(doc))
        for u in bad["uniforms"]:
            if "mesh" in u:
                u["mesh"]["extra"] = 1
        self.assertTrue(schema_errors(bad, "uniforms"))
        for path in ("../../etc/x.glb", "assets/a.obj", "/abs/a.glb", "assets/..\\x.glb"):
            bad = json.loads(json.dumps(doc))
            bad["uniforms"][0]["mesh"]["asset"] = path
            self.assertTrue(schema_errors(bad, "uniforms"), path)
        old = json.loads(json.dumps(doc))
        for u in old["uniforms"]:
            u.pop("mesh", None)
        old["schema_version"] = "1.0"
        self.assertEqual(schema_errors(old, "uniforms"), [])           # 1.0 files (no mesh) stay valid

    def test_default_uniform_is_first_and_is_the_only_one_sampled(self):
        doc = load_json(PROFILE / "uniforms.json")
        self.assertEqual(doc["uniforms"][0]["id"], DEFAULT_ID)
        self.assertEqual([u["id"] for u in doc["uniforms"] if u["weight"] > 0], [DEFAULT_ID])
        self.assertTrue(all(len(u["proxy_colors"][k]) == 3 for u in doc["uniforms"][:1] for k in ("red", "leather", "metal")))


if __name__ == "__main__":
    unittest.main()
