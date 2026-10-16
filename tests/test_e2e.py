"""Validation-failure tests and the end-to-end smoke test (5 samples).

Contents:
  TestSampleValidation (5 tests)
  TestEndToEnd (5 tests)
"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import make_config
from tosdg.generate import generate_sample, run
from tosdg.rng import derive_sample_seed
from tosdg.scene import GenerationContext, build_scene
from tosdg.validate import check_ground_truth, check_metadata, check_render, validate_dataset
from tosdg import groundtruth as gt

EXPECTED_SUFFIXES = (".rgb.png", ".mask.png", ".json")


class TestSampleValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp()
        cls.ctx = GenerationContext(make_config(cls.td))
        res = generate_sample(cls.ctx, 0, derive_sample_seed(cls.ctx.config["master_seed"], 0), 0)
        assert res.ok, res.errors
        cls.res = res

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td)

    def test_render_failures(self):
        rgb, mask = self.res.rgb, self.res.mask
        H, W = mask.shape
        self.assertEqual(check_render(rgb, mask, W, H, 2.0), [])
        self.assertTrue(check_render(np.zeros_like(rgb), mask, W, H, 2.0))  # blank render
        self.assertTrue(check_render(rgb, np.zeros_like(mask), W, H, 2.0))  # subject missing
        self.assertTrue(check_render(rgb[:-1], mask, W, H, 2.0))  # wrong size
        dark = rgb.copy(); dark[mask] = 3
        self.assertTrue(check_render(dark, mask, W, H, 2.0, (12.0, 245.0)))    # a black officer is rejected
        white = rgb.copy(); white[mask] = 255
        self.assertTrue(check_render(white, mask, W, H, 2.0, (12.0, 245.0)))   # so is a blown-out one

    def test_ground_truth_failures(self):
        # strict rules: the subject must be fully in frame (v2 behaviour, still available through the config)
        rules = {**self.ctx.validation, "max_bbox_edge_touch": False, "min_keypoints_in_frame_frac": 1.0, "require_signal_arm_in_frame": True}
        meta = self.res.meta
        sides = meta["officer"]["gesture"]["signal_arms"]
        self.assertEqual(check_ground_truth(meta, rules, sides), [])
        self.assertEqual(check_ground_truth(meta, self.ctx.validation, sides), [])
        self.assertEqual(self.ctx.validation["max_bbox_edge_touch"], True)             # v3 default: partly outside the frame is allowed
        full = meta["officer"]["bbox"]["full_xyxy"]
        self.assertEqual(len(full), 4)
        self.assertLess(full[0], full[2])
        m = copy.deepcopy(meta)
        m["officer"]["bbox"]["xyxy"] = [5, 5, 5, 9]  # zero width
        self.assertTrue(check_ground_truth(m, rules, sides))
        m = copy.deepcopy(meta)
        m["officer"]["bbox"]["truncated"] = True
        self.assertTrue(any("truncated" in e for e in check_ground_truth(m, rules, sides)))
        m = copy.deepcopy(meta)
        m["officer"]["keypoints_2d"]["coco17"]["uv"][0] = [0.5, 0.5]  # nose far from the body
        self.assertTrue(any("outside bbox" in e for e in check_ground_truth(m, rules, sides)))
        m = copy.deepcopy(meta)
        m["officer"]["keypoints_2d"]["coco17"]["in_frame"][9] = False  # left wrist out of frame
        self.assertTrue(any("in frame" in e for e in check_ground_truth(m, rules, ["l"])))

    def test_metadata_failures(self):
        meta = self.res.meta
        self.assertEqual(check_metadata(meta), [])
        m = copy.deepcopy(meta)
        del m["seed"]
        self.assertTrue(check_metadata(m))
        m = copy.deepcopy(meta)
        m["files"]["rgb"] = "other_000000.rgb.png"
        self.assertTrue(any("does not match" in e for e in check_metadata(m)))
        m = copy.deepcopy(meta)
        m["officer"]["bbox"]["mask_area_px"] = 0
        self.assertTrue(check_metadata(m))

    def test_keypoints_match_projection_of_scene(self):
        meta = self.res.meta
        seed = meta["seed"]["attempt_seed"]
        scene = build_scene(self.ctx, seed)
        names, pts = self.ctx.skeleton.keypoints(scene.fk_world, "coco17")
        uv, _ = scene.camera.project(pts)
        np.testing.assert_allclose(np.array(meta["officer"]["keypoints_2d"]["coco17"]["uv"]), uv, atol=1e-3)

    def test_instruction_for_the_camera_side(self):
        scene = build_scene(self.ctx, self.res.meta["seed"]["attempt_seed"])
        scene.camera_profile = {**scene.camera_profile, "observer_role": "road_user"}
        scene.camera_placement = {**scene.camera_placement, "azimuth_deg": -90.0}  # the officer's left side
        scene.gesture = self.ctx.gestures.gestures["VN_STOP_LEFT"]
        tc = gt.traffic_control(scene, 10)
        self.assertEqual((tc["approach_side"], tc["instruction_for_observer"]), ("left", "STOP"))
        scene.gesture = self.ctx.gestures.gestures["VN_NO_SIGNAL"]
        self.assertEqual(gt.traffic_control(scene, 10)["instruction_for_observer"], "NONE")
        scene.camera_profile = {**scene.camera_profile, "observer_role": "infrastructure"}
        self.assertIsNone(gt.traffic_control(scene, 10)["instruction_for_observer"])


class TestEndToEnd(unittest.TestCase):
    """Minimum E2E smoke test: generate 5 samples and verify every expected output."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td)

    def test_smoke_3_samples(self):
        ctx = GenerationContext(make_config(self.td, num_samples=3))
        out = self.td / "out"
        man = run(ctx, out, log=lambda *_: None)
        self.assertEqual(len(man["samples"]), 3)
        self.assertEqual(man["failures"], [])
        files = sorted(p.name for p in out.iterdir())
        for i in range(3):
            for suf in EXPECTED_SUFFIXES:
                self.assertIn(f"smoke_{i:06d}{suf}", files)
        self.assertEqual(len(files), 3 * 3 + 1)
        self.assertEqual(validate_dataset(out), [])
        seeds = set()
        for s in man["samples"]:
            meta = json.loads((out / s["metadata"]).read_text())
            seeds.add(meta["seed"]["sample_seed"])
            self.assertEqual(meta["profile"]["country"], "VN")
            self.assertIn(meta["officer"]["gesture"]["id"], ctx.gestures.gestures)
            for key in ("camera", "background", "lighting", "traffic_control"):
                self.assertIn(key, meta)
            self.assertEqual(len(meta["officer"]["keypoints_2d"]["coco17"]["uv"]), 17)
        self.assertEqual(len(seeds), 3)

        # determinism: a fresh context regenerates samples by index byte-identically (the first and the last: the seeds of the
        # samples do not depend on how many came before)
        out2 = self.td / "out2"
        man2 = run(GenerationContext(make_config(self.td, num_samples=3)), out2, [0, 2], log=lambda *_: None)
        self.assertEqual([s["sha256"] for s in man2["samples"]], [man["samples"][0]["sha256"], man["samples"][2]["sha256"]])
        # different master seed => different data
        out4 = self.td / "out4"
        man4 = run(GenerationContext(make_config(self.td, num_samples=1, master_seed=1)), out4, log=lambda *_: None)
        self.assertNotEqual(man["samples"][0]["sha256"], man4["samples"][0]["sha256"])

    def test_dataset_validation_detects_corruption(self):
        ctx = GenerationContext(make_config(self.td, num_samples=2))
        out = self.td / "out"
        run(ctx, out, log=lambda *_: None)
        self.assertEqual(validate_dataset(out), [])
        rgb = out / "smoke_000000.rgb.png"
        data = bytearray(rgb.read_bytes())
        data[-20] ^= 0xFF
        rgb.write_bytes(bytes(data))
        (out / "smoke_000001.mask.png").unlink()
        (out / "stray.txt").write_text("x")
        errs = "\n".join(validate_dataset(out))
        self.assertIn("rgb sha256 mismatch", errs)
        self.assertIn("mask file missing", errs)
        self.assertIn("unlisted files", errs)

    def test_unsatisfiable_validation_emits_nothing(self):
        cfg = make_config(self.td, num_samples=1, max_attempts_per_sample=3,
                          validation={"min_bbox_height_px": 10000})
        out = self.td / "out"
        man = run(GenerationContext(cfg), out, log=lambda *_: None)
        self.assertEqual(man["samples"], [])
        self.assertEqual(len(man["failures"]), 1)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["manifest.json"])
        self.assertEqual(man["rejection_counts"], {"bbox": 3})

    def test_out_of_frame_subject_rejected_and_resampled(self):
        # very long lens: subject frequently exceeds the frame -> must be rejected, never emitted
        cfg = make_config(self.td, num_samples=3, camera_profile_weights={"roadside": 1}, backend="proxy", uniform_brightness_range=[1.0, 1.0],
                          max_attempts_per_sample=50)
        ctx = GenerationContext(cfg)
        ctx.validation = {**ctx.validation, "max_bbox_edge_touch": False, "min_keypoints_in_frame_frac": 1.0, "require_signal_arm_in_frame": True}
        ctx.camera_profiles["roadside"] = {**ctx.camera_profiles["roadside"], "focal_length_mm": [150, 400],
                                           "distance_cm": [1200, 2000]}
        out = self.td / "out"
        man = run(ctx, out, log=lambda *_: None)
        self.assertGreater(sum(man["rejection_counts"].values()), 0)
        self.assertEqual(validate_dataset(out), [])
        for s in man["samples"]:
            meta = json.loads((out / s["metadata"]).read_text())
            self.assertFalse(meta["officer"]["bbox"]["truncated"])
            self.assertTrue(all(meta["officer"]["keypoints_2d"]["coco17"]["in_frame"]))

    def test_partly_out_of_frame_subject_is_kept_with_an_unclipped_bbox(self):
        cfg = make_config(self.td, num_samples=4, camera_profile_weights={"roadside": 1}, backend="proxy", uniform_brightness_range=[1.0, 1.0], max_attempts_per_sample=50)
        ctx = GenerationContext(cfg)
        ctx.camera_profiles["roadside"] = {**ctx.camera_profiles["roadside"], "focal_length_mm": [150, 400],
                                           "distance_cm": [1200, 2000]}
        out = self.td / "out3"
        man = run(ctx, out, log=lambda *_: None)
        self.assertEqual(validate_dataset(out), [])
        outside = 0
        for s in man["samples"]:
            b = json.loads((out / s["metadata"]).read_text())["officer"]["bbox"]
            W, H = 1280, 720
            fx0, fy0, fx1, fy1 = b["full_xyxy"]
            outside += fx0 < 0 or fy0 < 0 or fx1 > W or fy1 > H
        self.assertGreater(outside, 0)          # the full bbox goes beyond the image; the mask bbox stays inside it


if __name__ == "__main__":
    unittest.main()
