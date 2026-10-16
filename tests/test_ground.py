"""Standable ground of a backplate (ground.py): the officer's feet stay on the road / sidewalk of the photo.

Covers the feet check and the sideways re-crop, the library's ground masks (index 1.1), the fixtures' exact masks,
generation with masks (metadata 1.7 background.ground_check), the CLI, and - when torch, transformers and the SegFormer
weights are already on this machine - the detector itself.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import fixture_library, make_config
from test_cli import FAST, run_cli
from tosdg import ground as groundmod
from tosdg.backgrounds import BackgroundLibrary, crop_backplate
from tosdg.generate import run
from tosdg.ground import PlacementError, fit_feet, footprint_points
from tosdg.scene import GenerationContext, build_scene
from tosdg.schema import validate_instance


def _no_rng():
    raise AssertionError("the 'ground' stream must only be drawn when the crop has to move")


class FitFeetTests(unittest.TestCase):
    # a 200 x 100 photo whose left half is road; composed 1:1 into a 100 x 100 frame (100 px of horizontal play)
    MASK = np.zeros((50, 100), bool)
    MASK[:, :50] = True
    COMP = {"scale": 1.0, "offset_px": [0, 0], "horizon_aligned": True, "camera_horizon_v": 10.0}

    def test_feet_on_the_road_keep_the_drawn_crop(self):
        comp, rep = fit_feet(self.MASK, (200, 100), self.COMP, np.array([[60.0, 90.0], [70.0, 95.0]]), 100, 100, _no_rng)
        self.assertEqual((comp, rep), (self.COMP, {"status": "ok", "points": 2}))

    def test_feet_on_a_wall_move_the_crop_sideways(self):
        comp = {**self.COMP, "offset_px": [-100, 0]}                    # photo column = u + 100: the right half, not road
        feet = np.array([[20.0, 90.0], [30.0, 95.0]])
        new, rep = fit_feet(self.MASK, (200, 100), comp, feet, 100, 100, lambda: np.random.default_rng(1))
        self.assertEqual(rep["status"], "shifted")
        ox = new["offset_px"][0]
        self.assertTrue(-100 <= ox <= 0)
        self.assertTrue(all(self.MASK[0, int(u - ox) // 2] for u in feet[:, 0]))     # mask is half the photo's size
        self.assertEqual(new["offset_px"][1], 0)                        # the horizon alignment (vertical) never changes
        again, rep2 = fit_feet(self.MASK, (200, 100), new, feet, 100, 100, _no_rng)
        self.assertEqual(rep2["status"], "ok")

    def test_no_road_anywhere_is_a_placement_error(self):
        with self.assertRaises(PlacementError):
            fit_feet(np.zeros((50, 100), bool), (200, 100), self.COMP, np.array([[50.0, 90.0]]), 100, 100, _no_rng)

    def test_feet_outside_the_picture_are_not_judged(self):
        uv = np.array([[50.0, 140.0], [np.nan, np.nan], [-5.0, 50.0]])
        comp, rep = fit_feet(np.zeros((50, 100), bool), (200, 100), self.COMP, uv, 100, 100, _no_rng)
        self.assertEqual(rep, {"status": "not_visible", "points": 0})

    def test_footprint_is_on_the_ground(self):
        class FK:
            positions = {j: np.array([float(i), 2.0, 9.0]) for i, j in enumerate(groundmod.FOOT_JOINTS)}
        pts = footprint_points(FK, np.zeros(3), None)
        self.assertEqual(pts.shape, (6, 3))
        self.assertTrue(np.all(pts[:, 2] == 0))
        pod = footprint_points(FK, np.array([10.0, 0.0, 40.0]), {"diameter_cm": 100.0})
        self.assertEqual(len(pod), 1 + groundmod.PODIUM_RIM_POINTS)
        self.assertTrue(np.allclose(np.hypot(pod[1:, 0] - 10.0, pod[1:, 1]), 50.0))
        self.assertTrue(np.all(pod[:, 2] == 0))


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.td, True)
        Image.new("RGB", (300, 150), (90, 90, 90)).save(self.td / "photo.png")
        self.lib = BackgroundLibrary(self.td / "lib")

    def test_a_mask_is_stored_hashed_and_checked(self):
        m = np.zeros((150, 300), np.uint8)
        m[100:] = 255
        e = self.lib.add(self.td / "photo.png", ["day"], ground=m)
        self.lib.save()
        g = e["ground"]
        self.assertEqual((g["file"], g["method"], g["standable_frac"]), ("images/photo.ground.png", "manual", 0.3333))
        idx = json.loads((self.td / "lib" / "index.json").read_text())
        self.assertEqual(idx["schema_version"], "1.2")
        validate_instance(idx, "background_library")
        lib = BackgroundLibrary(self.td / "lib")
        self.assertEqual(lib.integrity_errors(), [])
        self.assertTrue(lib.ground_mask(lib.by_id["photo"])[120, 10])
        self.assertFalse(lib.ground_mask(lib.by_id["photo"])[50, 10])
        Image.new("L", (300, 150), 255).save(self.td / "lib" / g["file"])        # edited behind the index's back
        self.assertIn("ground mask sha256 mismatch", " ".join(BackgroundLibrary(self.td / "lib").integrity_errors()))

    def test_a_mask_of_the_wrong_shape_is_refused_and_nothing_is_kept(self):
        with self.assertRaises(ValueError):
            self.lib.add(self.td / "photo.png", [], ground=np.ones((100, 100), np.uint8) * 255)
        self.assertNotIn("photo", self.lib.by_id)
        self.assertFalse((self.td / "lib" / "images" / "photo.png").exists())

    def test_old_index_without_masks_still_loads(self):
        e = self.lib.add(self.td / "photo.png", [])
        self.lib.index["schema_version"] = "1.0"
        (self.td / "lib" / "index.json").write_text(json.dumps(self.lib.index))
        lib = BackgroundLibrary(self.td / "lib")
        self.assertIsNone(lib.ground_mask(lib.by_id[e["id"]]))

    def test_fixtures_have_exact_masks(self):
        lib = BackgroundLibrary(fixture_library())
        for b in lib.index["backgrounds"]:
            m = lib.ground_mask(b)
            self.assertEqual(b["ground"]["method"], "horizon")
            row = int(np.ceil(b["horizon_y_frac"] * m.shape[0]))
            self.assertTrue(m[row:].all() and not m[:row].any(), b["id"])


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = Path(tempfile.mkdtemp())
        lib_dir = cls.td / "lib"
        shutil.copytree(fixture_library(), lib_dir)
        lib = BackgroundLibrary(lib_dir)
        for b in lib.index["backgrounds"]:            # road only in a narrow band in the middle of each photo
            m = lib.ground_mask(b).copy()
            w = m.shape[1]
            m[:, : int(w * 0.45)] = False
            m[:, int(w * 0.55):] = False
            lib.set_ground(b["id"], m.astype(np.uint8) * 255)
        lib.save()
        cls.lib_dir = lib_dir
        cls.ctx = GenerationContext(make_config(cls.td, background_library=str(lib_dir), **FAST))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, True)

    def test_feet_end_up_on_the_road_or_the_sample_is_rejected(self):
        status = {}
        for seed in range(40):
            try:
                s = build_scene(self.ctx, seed)
            except PlacementError:
                status["rejected"] = status.get("rejected", 0) + 1
                continue
            st = s.ground_check["status"]
            status[st] = status.get(st, 0) + 1
            if st in ("ok", "shifted"):                    # the final crop puts every visible foot point on the road
                lens_on = s.camera.lens is not None and s.camera.lens.active
                from tosdg.lens import canvas_camera
                cam = canvas_camera(s.camera) if lens_on else s.camera
                uv, _ = cam.project_pinhole(footprint_points(s.fk_world, s.officer_position, s.podium))
                bg = s.background
                with Image.open(self.ctx.backgrounds.path(bg)) as im:
                    _, rep = fit_feet(self.ctx.backgrounds.ground_mask(bg), im.size, s.composition, uv,
                                      s.backplate.width, s.backplate.height, _no_rng)
                    self.assertEqual(rep["status"], "ok", seed)
                    self.assertEqual(np.asarray(crop_backplate(im, s.composition, *s.backplate.size)).tobytes(),
                                     np.asarray(s.backplate).tobytes())
        self.assertGreater(status.get("shifted", 0), 0, status)

    def test_metadata_records_the_check_and_reproduces(self):
        out = self.td / "out"
        man = run(self.ctx, out, log=lambda *_: None, overwrite=True)
        self.assertTrue(man["samples"])
        for s in man["samples"]:
            meta = json.loads((out / s["metadata"]).read_text())
            self.assertEqual(meta["schema_version"], "1.8")
            self.assertIn(meta["background"]["ground_check"]["status"], ("ok", "shifted", "not_visible"))
        (self.td / "b").mkdir(exist_ok=True)
        again = run(GenerationContext(make_config(self.td / "b", background_library=str(self.lib_dir), **FAST)),
                    self.td / "out2", log=lambda *_: None)
        self.assertEqual([s["sha256"] for s in man["samples"]], [s["sha256"] for s in again["samples"]])

    def test_without_masks_nothing_is_checked(self):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        shutil.copytree(fixture_library(), td / "lib")
        lib = BackgroundLibrary(td / "lib")
        for i in list(lib.by_id):
            lib.clear_ground(i)
        lib.save()
        ctx = GenerationContext(make_config(td, background_library=str(td / "lib"), **FAST))
        self.assertEqual(build_scene(ctx, 3).ground_check, {"status": "no_mask", "points": 0})


class CliTests(unittest.TestCase):
    def test_set_and_clear_a_mask_by_hand(self):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        shutil.copytree(fixture_library(), td / "lib")
        b = BackgroundLibrary(td / "lib").by_id["fx_dusk"]
        Image.new("L", (b["width"], b["height"]), 255).save(td / "m.png")
        rc, out, err = run_cli("backgrounds", "ground", td / "lib", "--id", "fx_dusk", "--mask", td / "m.png")
        self.assertEqual(rc, 0, err)
        self.assertIn("100% standable", out)
        rc, out, _ = run_cli("backgrounds", "ground", td / "lib", "--id", "fx_dusk")
        self.assertIn("--force", out)                       # an existing mask is kept unless forced
        rc, out, _ = run_cli("backgrounds", "ground", td / "lib", "--id", "fx_dusk", "--clear")
        self.assertNotIn("ground", BackgroundLibrary(td / "lib").by_id["fx_dusk"])
        rc, _, err = run_cli("backgrounds", "ground", td / "lib", "--mask", td / "m.png")
        self.assertEqual(rc, 2)
        self.assertIn("--id", err)
        rc, out, _ = run_cli("backgrounds", "check", td / "lib")
        self.assertEqual(rc, 0, out)


def _segformer_cached() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(groundmod.DEFAULT_MODEL, "config.json"), str)


@unittest.skipUnless(_segformer_cached(), "torch / transformers / the SegFormer weights are not on this machine")
class DetectorTests(unittest.TestCase):
    def test_road_below_the_horizon_and_nothing_above(self):
        lib = BackgroundLibrary(fixture_library())
        b = lib.by_id["fx_day_clear"]
        with Image.open(lib.path(b)) as im:
            mask, method = groundmod.detect_ground(im, b["horizon_y_frac"])
        self.assertTrue(method.startswith("segformer:"))
        m = mask >= 128
        h, w = m.shape
        row = int(np.ceil(b["horizon_y_frac"] * h))
        self.assertFalse(m[:row].any())
        self.assertGreater(m[int(h * 0.75):, int(w * 0.3): int(w * 0.7)].mean(), 0.8)   # the road in front of the camera


if __name__ == "__main__":
    unittest.main()
