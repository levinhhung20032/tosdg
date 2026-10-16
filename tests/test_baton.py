"""The optional baton prop: grip geometry, fist loosening, body collisions, sampling, rendering, ground truth.

Contents:
  spec()
  grip()
  GeometryTests (7 tests)
  SamplingTests (3 tests)
  RenderAndGroundTruthTests (8 tests)
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import ROOT, make_config, poses, skeleton, vn_gestures
from tosdg import baton, rng
from tosdg.generate import run
from tosdg.render_mesh import skin_mesh
from tosdg.scene import GenerationContext
from tosdg.skeleton import segment_distance
from tosdg.validate import validate_dataset


def spec(hand="r", **kw):
    return baton.normalise({"enabled": True, "hand": hand, **kw})


def grip(pid, mode="across", hand="r", character="proxy_v1"):
    sk, pl = skeleton(character), poses()
    p, info = baton.apply_grip(sk, pl.body_pose(pid), spec(hand), mode, pl.hand_pose("fist", hand))
    fk = sk.fk(p)
    a, b = baton.segment(sk, fk, spec(hand), mode)
    return sk, p, info, fk, a, b


class GeometryTests(unittest.TestCase):
    def test_disabled_baton_is_none_and_bad_specs_are_rejected(self):
        self.assertIsNone(baton.normalise(None))
        self.assertIsNone(baton.normalise({"enabled": False}))
        for bad in ({"hand": "x"}, {"length_cm": -1}, {"handle_cm": 60}):
            with self.assertRaises(ValueError):
                baton.normalise({"enabled": True, **bad})

    def test_arm_forward_gives_a_baton_pointing_straight_up(self):
        for pid, hand in (("arm_forward_r", "r"),):
            _, _, info, _, a, b = grip(pid, hand=hand)
            d = (b - a) / np.linalg.norm(b - a)
            self.assertGreater(float(d @ [0, 0, 1]), 0.98)                  # perpendicular to the forearm, vertical
        self.assertGreater(info["closure"], 0.75)                          # a real fist

    def test_left_hand_baton_is_the_mirror_image(self):
        for name in ("stop_bent", "speed_up_chest"):
            _, _, _, _, ar, br = grip(f"{name}_r", hand="r")
            _, _, _, _, al, bl = grip(f"{name}_l", hand="l")
            np.testing.assert_allclose(bl * [1, -1, 1], br, atol=0.6)

    def test_baton_along_the_forearm_when_asked(self):
        sk, p, info, fk, a, b = grip("arm_vertical_r", "along")
        forearm = fk.positions["hand_r"] - fk.positions["lowerarm_r"]
        d = b - a
        cos = float(d @ forearm / np.linalg.norm(d) / np.linalg.norm(forearm))
        self.assertGreater(cos, np.cos(np.radians(15)))
        self.assertGreater(p["hand_r"]["deviation"], 10.0)                # the wrist deviates to line the baton up with the forearm
        self.assertEqual(p["hand_r"]["flexion"], 0.0)

    def test_the_grip_is_a_tight_overhand_fist_with_a_straight_wrist(self):
        for pid, mode in (("arm_forward_r", "across"), ("stop_bent_r", "across"), ("arm_vertical_r", "along")):
            sk, p, info, fk, a, b = grip(pid, mode)
            self.assertEqual(p["hand_r"]["flexion"], 0.0, pid)             # wrist straight: the hand continues the forearm
            self.assertGreaterEqual(min(info["finger_closure"].values()), 0.75, f"{pid}: every finger is wrapped tightly")
            self.assertGreaterEqual(info["closure"], 0.85, pid)
            # the thumb rests against the fist on the thumb (radial) side, near the index finger, not sticking out
            tip = fk.positions["thumb_tip_r"]
            index2 = fk.positions["index_02_r"]
            self.assertLess(float(np.linalg.norm(tip - index2)), 6.0, pid)

    def test_no_finger_penetrates_the_baton(self):
        for pid, mode in (("arm_forward_r", "across"), ("stop_bent_r", "across"), ("arm_vertical_r", "along"), ("arm_horizontal_r", "along")):
            sk, p, info, fk, a, b = grip(pid, mode)
            for seg in sk.segments.values():
                if seg["region"] == "finger" and seg["name"].endswith("_r") and not seg["name"].startswith("thumb_1"):   # thumb_1 = metacarpal, inside the thenar mass
                    d = segment_distance(a, b, fk.positions[seg["from"]], fk.positions[seg["to"]])
                    self.assertGreaterEqual(d, sk.collision_tolerance * (1.5 + seg["radius"]) - 1e-6, f"{pid} {seg['name']}")
            self.assertEqual(baton.body_collisions(sk, fk, spec(), mode), [], pid)

    def test_a_thick_baton_opens_the_hand_more(self):
        sk, pl = skeleton(), poses()
        fist = pl.hand_pose("fist", "r")
        thin = baton.apply_grip(sk, pl.body_pose("arm_forward_r"), spec(radius_cm=1.0), "across", fist)[1]["closure"]
        thick = baton.apply_grip(sk, pl.body_pose("arm_forward_r"), spec(radius_cm=2.2), "across", fist)[1]["closure"]
        self.assertGreater(thin, thick)


class SamplingTests(unittest.TestCase):
    def test_every_gesture_samples_valid_poses_with_a_baton_in_either_hand(self):
        lib, sk = vn_gestures(), skeleton("tripo_v1")
        for hand in ("r", "l"):
            sp = spec(hand)
            for gid in lib.gestures:
                for i in range(4):
                    s = lib.sample(gid, sk, rng.stream(100 + i, "pose"), baton=sp)
                    self.assertEqual(s.baton["hand"], hand)
                    self.assertGreaterEqual(s.baton["closure"], 0.0)
                    again = lib.sample(gid, sk, rng.stream(100 + i, "pose"), baton=sp)
                    self.assertEqual(s.pose, again.pose)                    # deterministic

    def test_vertical_and_horizontal_arm_gestures_hold_the_baton_along_the_arm(self):
        lib = vn_gestures()
        for gid in ("VN_ARM_VERTICAL", "VN_ARMS_HORIZONTAL"):
            self.assertTrue(all(v.get("baton_axis") == "along" for v in lib.gestures[gid]["pose_variants"]), gid)
        self.assertTrue(all("baton_axis" not in v for v in lib.gestures["VN_STOP_RIGHT"]["pose_variants"]))

    def test_without_a_baton_nothing_changes(self):
        lib, sk = vn_gestures(), skeleton()
        a = lib.sample("VN_STOP_RIGHT", sk, rng.stream(5, "pose"))
        b = lib.sample("VN_STOP_RIGHT", sk, rng.stream(5, "pose"), baton=None)
        self.assertEqual((a.pose, a.baton), (b.pose, None))


class RenderAndGroundTruthTests(unittest.TestCase):
    def test_baton_mesh_follows_the_hand_through_skinning(self):
        from helpers import ROOT as R
        from tosdg.uniform_mesh import read_glb
        mesh = read_glb(R / "profiles" / "VN" / "assets" / "beta.glb")
        sp = spec()
        with_b = baton.mesh_with_baton(mesh, sp, "across")
        self.assertEqual(with_b.n_vertices, mesh.n_vertices + 26)
        sk_, p, info, fk, a, b = grip("arm_forward_r", character="tripo_v1")
        pos, _ = skin_mesh(with_b, fk)
        new = pos[mesh.n_vertices:]
        ax = (b - a) / np.linalg.norm(b - a)
        d = np.linalg.norm((new - a) - np.outer((new - a) @ ax, ax), axis=1)       # distance to the baton's axis
        np.testing.assert_allclose(d[:24], sp["radius_cm"], atol=0.05)    # the side-wall vertices sit on the baton's surface
        self.assertEqual(len(mesh.materials) + 1, len(with_b.materials))

    def test_the_3d_baton_model_is_placed_along_the_segment_and_rendered(self):
        from tosdg.uniform_mesh import read_glb
        path = baton.prop_path(ROOT / "profiles" / "VN")
        if not path.exists():
            self.skipTest("no profiles/VN/props/baton.glb")
        prop = baton.load_prop(path)
        self.assertEqual(prop["texture"].shape[:2], (512, 512))
        mesh = read_glb(ROOT / "profiles" / "VN" / "assets" / "beta.glb")
        sp = spec()
        with_b = baton.mesh_with_baton(mesh, sp, "across", prop=prop)
        n = len(prop["pos"])
        self.assertEqual((with_b.n_vertices, len(with_b.textures)), (mesh.n_vertices + n, len(mesh.textures) + 1))
        sk_, p, info, fk, a, b = grip("arm_forward_r", character="tripo_v1")
        pos, _ = skin_mesh(with_b, fk)
        new = pos[mesh.n_vertices:]
        ax = (b - a) / np.linalg.norm(b - a)
        t = (new - a) @ ax
        self.assertAlmostEqual(float(t.min()), 0.0, delta=1.0)                       # handle end at the segment's start
        self.assertAlmostEqual(float(t.max()), sp["length_cm"], delta=1.5)           # tip at its end: the rod is `length_cm` long
        self.assertGreater(float(ax @ [0, 0, 1]), 0.98)                              # arm forward, baton straight up

    def beta(self):
        from tosdg.uniform_mesh import read_glb
        return read_glb(ROOT / "profiles" / "VN" / "assets" / "beta.glb")

    def test_not_holding_a_baton_hangs_it_from_the_belt(self):
        """The model carries no baton: the prop hangs from the belt at the place its fit gives, rigid with the pelvis."""
        mesh, prop = self.beta(), baton.load_prop(baton.prop_path(ROOT / "profiles" / "VN"))
        cfg = baton.belt_config(mesh)
        self.assertIsNotNone(cfg, "beta.fit.json has no belt_baton")
        stowed = baton.mesh_with_officer_baton(mesh, None, "across", prop=prop)
        self.assertEqual(stowed.n_vertices, mesh.n_vertices + len(prop["pos"]))
        new = stowed.positions[mesh.n_vertices:]
        pelvis = stowed.joint_names.index("pelvis")
        self.assertTrue((stowed.joints[mesh.n_vertices:, 0] == pelvis).all())                    # rigid on the pelvis
        np.testing.assert_allclose(stowed.weights[mesh.n_vertices:, 0], 1.0)
        H = np.asarray(cfg["handle_cm"])
        f1 = np.asarray(cfg["direction"], float) / np.linalg.norm(cfg["direction"])
        t = (new - H) @ f1
        self.assertAlmostEqual(float(t.min()), 0.0, delta=1.0)                                    # the handle end is at the belt
        self.assertAlmostEqual(float(t.max()), cfg["length_cm"], delta=1.5)                        # and the rod hangs `length_cm` below it
        self.assertLess(float(new[:, 2].max()), 108.0)                                             # nothing sticks up above the belt
        self.assertGreater(float(new[:, 2].min()), 40.0)                                           # and it ends above the knee
        sk = skeleton("tripo_v1")
        up, _ = skin_mesh(stowed, sk.fk({"thigh_r": {"flexion": 60.0}}))
        np.testing.assert_allclose(up[mesh.n_vertices:], new, atol=1e-6)                           # moving a thigh does not move the baton
        self.assertEqual(len(stowed.materials), len(mesh.materials) + 1)

    def test_holding_a_baton_takes_it_off_the_belt(self):
        """One baton per picture: held, the prop is in the hand and nothing hangs from the belt."""
        mesh, prop = self.beta(), baton.load_prop(baton.prop_path(ROOT / "profiles" / "VN"))
        held = baton.mesh_with_officer_baton(mesh, spec(), "across", prop=prop)
        self.assertEqual(held.n_vertices, mesh.n_vertices + len(prop["pos"]))                      # one prop, not two
        self.assertEqual(len(held.materials), len(mesh.materials) + 1)
        hand = held.joint_names.index("hand_r")
        self.assertTrue((held.joints[mesh.n_vertices:, 0] == hand).all())
        belt = baton.mesh_with_officer_baton(mesh, None, "across", prop=prop)
        self.assertFalse(np.allclose(held.positions[mesh.n_vertices:].mean(axis=0), belt.positions[mesh.n_vertices:].mean(axis=0), atol=5.0))

    def test_a_model_without_belt_settings_shows_no_baton_when_it_is_not_held(self):
        mesh = self.beta()
        plain = type(mesh)(**{**mesh.__dict__, "info": {k: v for k, v in mesh.info.items() if k != "fit"}})
        self.assertIsNone(baton.belt_config(plain))
        self.assertIs(baton.mesh_with_officer_baton(plain, None, "across"), plain)

    def test_the_belt_baton_is_a_plain_cylinder_without_the_3d_model(self):
        mesh = self.beta()
        cyl = baton.mesh_with_officer_baton(mesh, None, "across", prop=None)
        self.assertEqual(cyl.n_vertices, mesh.n_vertices + 26)

    def test_every_sample_has_one_baton_in_the_picture(self):
        """The generator draws the held baton or the belt one, never both: the mesh of a sample has exactly one extra 'baton' material."""
        from unittest import mock
        seen = []
        real = baton.mesh_with_officer_baton

        def spy(mesh, spec_, mode, prop=None):
            out = real(mesh, spec_, mode, prop=prop)
            seen.append((spec_ is not None, sum(m["name"] == "baton" for m in out.materials)))
            return out
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with mock.patch("tosdg.baton.mesh_with_officer_baton", spy):
                for sub, over in (("held", {"baton": {"enabled": True, "hand": "r"}}), ("belt", {})):
                    (td / sub).mkdir()
                    run(GenerationContext(make_config(td / sub, num_samples=1, backend="mesh", **over)), td / sub / "out", log=lambda *_: None)
        self.assertEqual(sorted(set(seen)), [(False, 1), (True, 1)])

    def test_generation_records_the_baton_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = make_config(td, num_samples=3, baton={"enabled": True, "hand": "r"}, max_attempts_per_sample=50)
            ctx = GenerationContext(cfg)
            man = run(ctx, td / "out", log=lambda *_: None)
            self.assertEqual(validate_dataset(td / "out"), [])
            for s in man["samples"]:
                meta = json.loads((td / "out" / s["metadata"]).read_text())
                b = meta["officer"]["baton"]
                self.assertEqual((meta["schema_version"], b["enabled"], b["hand"]), ("1.8", True, "r"))
                length = np.linalg.norm(np.array(b["tip_world"]) - np.array(b["handle_world"]))
                self.assertAlmostEqual(float(length), b["length_cm"], delta=0.05)
            (td / "c2").mkdir()
            cfg2 = make_config(td / "c2", num_samples=1)
            man2 = run(GenerationContext(cfg2), td / "out2", log=lambda *_: None)
            m2 = json.loads((td / "out2" / man2["samples"][0]["metadata"]).read_text())
            self.assertEqual(m2["officer"]["baton"], {"enabled": False})


if __name__ == "__main__":
    unittest.main()
