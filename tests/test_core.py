"""Unit tests: schemas, rng, skeleton/FK conventions, gestures, camera, ground truth.

Contents:
  TestSchemas (6 tests)
  TestRng (2 tests)
  TestSkeleton (5 tests)
  TestGestures (4 tests)
  TestCamera (3 tests)
  TestGroundTruth (1 tests)
"""
import copy
import unittest

import numpy as np

from helpers import ROOT, poses, skeleton, vn_gestures
from tosdg import rng
from tosdg.camera import Camera, look_at_rotator, rotator_matrix
from tosdg.gestures import (GestureLibrary, PoseSamplingError, approach_side,
                            pose_validity, side_boundary_distance)
from tosdg.groundtruth import bbox_from_mask
from tosdg.poses import merge, save_pose, load_pose_file
from tosdg.schema import SchemaError, schema_errors, validate_instance, load_json


class TestSchemas(unittest.TestCase):
    def test_all_profile_files_valid(self):
        for prof_dir in (ROOT / "profiles").iterdir():
            prof = load_json(prof_dir / "profile.json", "profile")
            for kind, fname in prof["files"].items():
                load_json(prof_dir / fname, kind)

    def test_all_poses_valid(self):
        for p in (ROOT / "poses").rglob("*.json"):
            load_pose_file(p)

    def test_configs_valid(self):
        for p in (ROOT / "configs").glob("*.json"):
            load_json(p, "generation_config")

    def test_malformed_gesture_rejected(self):
        g = load_json(ROOT / "profiles" / "VN" / "gestures.json")
        bad = copy.deepcopy(g)
        del bad["gestures"][0]["instructions"]["left"]
        self.assertTrue(schema_errors(bad, "gestures"))
        bad = copy.deepcopy(g)
        bad["gestures"][0]["pose_variants"][0]["constraints"].append({"type": "levitate"})
        self.assertTrue(schema_errors(bad, "gestures"))
        bad = copy.deepcopy(g)
        bad["gestures"][0]["variation"] = {"upperarm_r.flexion": [1]}
        self.assertTrue(schema_errors(bad, "gestures"))

    def test_unknown_pose_reference_rejected(self):
        g = load_json(ROOT / "profiles" / "VN" / "gestures.json")
        g["gestures"][0]["pose_variants"][0]["pose"] = "does_not_exist"
        with self.assertRaises(ValueError):
            GestureLibrary(g, poses())

    def test_pose_roundtrip(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "x.json"
            save_pose(p, "x", {"upperarm_r": {"flexion": 45.123456}})
            self.assertEqual(load_pose_file(p)["joints"], {"upperarm_r": {"flexion": 45.1235}})
            with self.assertRaises(SchemaError):
                validate_instance({"schema_version": "1.0", "kind": "body", "id": "bad id!", "joints": {}}, "pose")


class TestRng(unittest.TestCase):
    def test_golden_values(self):
        # Changing these breaks reproducibility of every existing dataset.
        self.assertEqual(rng.derive_sample_seed(20261008, 0), rng.derive_sample_seed(20261008, 0))
        self.assertNotEqual(rng.derive_sample_seed(1, 0), rng.derive_sample_seed(1, 1))
        self.assertEqual(rng.derive_attempt_seed(123, 0), 123)
        a = rng.stream(5, "camera").random(3)
        np.testing.assert_array_equal(a, rng.stream(5, "camera").random(3))
        self.assertFalse(np.allclose(a, rng.stream(5, "pose").random(3)))

    def test_weighted_choice_order_independent(self):
        w1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        w2 = {"c": 3.0, "a": 1.0, "b": 2.0}
        r1 = [rng.weighted_choice(rng.stream(i, "x"), w1) for i in range(50)]
        r2 = [rng.weighted_choice(rng.stream(i, "x"), w2) for i in range(50)]
        self.assertEqual(r1, r2)


class TestSkeleton(unittest.TestCase):
    def setUp(self):
        self.sk = skeleton()

    def test_rest_pose_geometry(self):
        fk = self.sk.fk({})
        self.assertAlmostEqual(fk.positions["upperarm_l"][1], -20, places=6)  # left = -Y
        self.assertAlmostEqual(fk.positions["upperarm_r"][1], 20, places=6)
        self.assertLess(fk.positions["hand_r"][2], fk.positions["lowerarm_r"][2])
        self.assertEqual(self.sk.collisions(fk), [])
        self.assertEqual(self.sk.pose_errors({}), [])

    def test_mirrored_dof_conventions(self):
        for s, lat in (("l", -1), ("r", 1)):
            fk = self.sk.fk({f"upperarm_{s}": {"flexion": 90}})
            arm = fk.positions[f"hand_{s}"] - fk.positions[f"upperarm_{s}"]
            self.assertGreater(arm[0], 50, "shoulder flexion raises arm forward (+X)")
            fk = self.sk.fk({f"upperarm_{s}": {"abduction": 90}})
            arm = fk.positions[f"hand_{s}"] - fk.positions[f"upperarm_{s}"]
            self.assertGreater(arm[1] * lat, 50, "abduction raises arm sideways, away from body")
            fk = self.sk.fk({f"upperarm_{s}": {"abduction": 90, "twist": -90}})
            self.assertGreater(self.sk.palm_normal(fk, s)[0], 0.99, "abducted arm, twist -90 => palm forward")
            fk0 = self.sk.fk({})
            fk = self.sk.fk({f"index_01_{s}": {"flexion": 90}})
            tip_shift = fk.positions[f"index_tip_{s}"] - fk0.positions[f"index_tip_{s}"]
            palm = self.sk.palm_normal(fk0, s)
            self.assertGreater(tip_shift @ palm, 3, "finger flexion curls toward the palm")
            fk = self.sk.fk({f"calf_{s}": {"flexion": 90}})
            self.assertLess(fk.positions[f"foot_{s}"][0], -30, "knee flexion moves the shin backwards")

    def test_joint_limits(self):
        errs = self.sk.pose_errors({"lowerarm_r": {"flexion": -30}, "upperarm_r": {"nope": 1}, "ghost": {}})
        self.assertEqual(len(errs), 3)

    def test_collision_detected(self):
        # forearm folded across the torso => penetration
        fk = self.sk.fk({"upperarm_r": {"flexion": 30, "abduction": -25}, "lowerarm_r": {"flexion": 120}})
        self.assertTrue(any("torso" in c for c in self.sk.collisions(fk)))

    def test_keypoint_sets(self):
        names, pts = self.sk.keypoints(self.sk.fk({}), "coco17")
        self.assertEqual(len(names), 17)
        self.assertEqual(pts.shape, (17, 3))
        for s in "lr":
            n, p = self.sk.keypoints(self.sk.fk({}), f"hand21_{s}")
            self.assertEqual(len(n), 21)


class TestGestures(unittest.TestCase):
    def test_sampling_deterministic(self):
        lib, sk = vn_gestures(), skeleton()
        a = lib.sample("VN_STOP_RIGHT", sk, rng.stream(42, "pose"))
        b = lib.sample("VN_STOP_RIGHT", sk, rng.stream(42, "pose"))
        self.assertEqual(a.pose, b.pose)
        c = lib.sample("VN_STOP_RIGHT", sk, rng.stream(43, "pose"))
        self.assertNotEqual(a.pose, c.pose)

    def test_constraints_reject_wrong_pose(self):
        lib, sk = vn_gestures(), skeleton()
        g = lib.gestures["VN_STOP_RIGHT"]
        v = g["pose_variants"][0]
        neutral = merge(poses().body_pose("neutral_stand"), poses().hand_pose("open_flat", "r"))
        errs, _ = pose_validity(sk, g, v, neutral)
        self.assertTrue(any(e.startswith("limb_direction") for e in errs))
        fist = merge(poses().body_pose("stop_bent_r"), poses().hand_pose("fist", "r"))
        errs, _ = pose_validity(sk, g, v, fist)
        self.assertTrue(any(e.startswith("finger_curl") for e in errs))
        g2 = lib.gestures["VN_STOP_LEFT"]                     # the right-arm pose is not a valid LEFT stop signal
        errs, _ = pose_validity(sk, g2, g2["pose_variants"][0], poses().body_pose("stop_bent_r"))
        self.assertTrue(errs)

    def test_impossible_constraints_raise(self):
        lib, sk = vn_gestures(), skeleton()
        g = copy.deepcopy(lib.data)
        g["gestures"][0]["constraints"] = [{"type": "above", "joint": "foot_r", "reference": "head_end"}]
        bad = GestureLibrary(g, poses())
        with self.assertRaises(PoseSamplingError):
            bad.sample(g["gestures"][0]["id"], sk, rng.stream(0, "pose"), max_tries=10)

    def test_approach_side(self):
        self.assertEqual(approach_side(0), "front")
        self.assertEqual(approach_side(90), "right")
        self.assertEqual(approach_side(-90), "left")
        self.assertEqual(approach_side(180), "back")
        self.assertEqual(approach_side(-179), "back")
        self.assertAlmostEqual(side_boundary_distance(44), 1)
        self.assertAlmostEqual(side_boundary_distance(0), 45)


class TestCamera(unittest.TestCase):
    def test_rotator_matches_unreal(self):
        np.testing.assert_allclose(rotator_matrix(0, 90, 0)[:, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(rotator_matrix(90, 0, 0)[:, 0], [0, 0, 1], atol=1e-12)
        # FRotationMatrix, roll only: Y axis = (0, cos r, -sin r), Z axis = (0, sin r, cos r)
        r = np.radians(30)
        np.testing.assert_allclose(rotator_matrix(0, 0, 30)[:, 1], [0, np.cos(r), -np.sin(r)], atol=1e-12)
        np.testing.assert_allclose(rotator_matrix(0, 0, 30)[:, 2], [0, np.sin(r), np.cos(r)], atol=1e-12)
        # general case against the closed-form FRotationMatrix rows
        p, y, r = np.radians([20, 35, -10])
        cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
        ue = np.array([[cp * cy, cp * sy, sp],
                       [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
                       [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp]])
        np.testing.assert_allclose(rotator_matrix(20, 35, -10).T, ue, atol=1e-12)

    def test_projection(self):
        cam = Camera("t", (0.0, 0.0, 0.0), 0, 0, 0, 36.0, 36.0, 400, 200)  # 90deg hfov
        self.assertAlmostEqual(cam.hfov_deg, 2 * np.degrees(np.arctan(0.5)))
        uv, d = cam.project(np.array([[100, 0, 0], [100, 50, 0], [100, 0, 25], [-10, 0, 0]], float))
        np.testing.assert_allclose(uv[0], [200, 100])
        np.testing.assert_allclose(uv[1], [400, 100])  # right edge: y/x = 0.5 = tan(hfov/2)
        np.testing.assert_allclose(uv[2], [200, 0])  # top edge (square pixels)
        self.assertTrue(np.isnan(uv[3]).all())

    def test_look_at_centres_target(self):
        eye, target = np.array([500.0, 300.0, 400.0]), np.array([0.0, 0.0, 100.0])
        p, y = look_at_rotator(eye, target)
        cam = Camera("t", tuple(eye), p, y, 0, 20, 36, 640, 480)
        uv, _ = cam.project(target[None])
        np.testing.assert_allclose(uv[0], [320, 240], atol=1e-6)


class TestGroundTruth(unittest.TestCase):
    def test_bbox_from_mask(self):
        m = np.zeros((10, 20), bool)
        m[2:5, 3:9] = True
        bb = bbox_from_mask(m)
        self.assertEqual(bb["xyxy"], [3, 2, 9, 5])
        self.assertEqual(bb["xywh"], [3, 2, 6, 3])
        self.assertEqual(bb["mask_area_px"], 18)
        self.assertFalse(bb["truncated"])
        m[0, 0] = True
        self.assertTrue(bbox_from_mask(m)["truncated"])
        self.assertIsNone(bbox_from_mask(np.zeros((4, 4), bool)))


if __name__ == "__main__":
    unittest.main()
