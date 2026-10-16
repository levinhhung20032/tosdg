"""Lens distortion: model, projection, remap (image + mask), black border, camera profiles, Studio parity.

Contents:
  cam_with
  LensMathTests (5 tests)
  RemapTests (4 tests)
  ProfileTests (5 tests)
  StudioLensTests (6 tests)
  GenerationTests (2 tests)
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from helpers import ROOT, make_config, skeleton
from tosdg import rng as rngmod
from tosdg.camera import Camera
from tosdg.generate import run
from tosdg.scene import GenerationContext
from tosdg.validate import validate_dataset
from tosdg.lens import Lens, canvas_size, distort_plate, layer_with_lens, render_with_lens, sample_lens
from tosdg.schema import load_json
from tosdg.studio import core


def cam_with(lens, w=320, h=180, f=3.6, sensor=6.17):
    return Camera("t", (-400.0, 0.0, 150.0), 0.0, 0.0, 0.0, f, sensor, w, h, lens=lens)


class LensMathTests(unittest.TestCase):
    def test_no_lens_is_pinhole(self):
        pts = np.array([[0.0, 30.0, 150.0], [0.0, -120.0, 40.0]])
        a, _ = cam_with(None).project(pts)
        b, _ = cam_with(Lens(0.0, 0.0, 0.0)).project(pts)
        np.testing.assert_array_equal(a, b)

    def test_barrel_pulls_points_towards_the_centre(self):
        pts = np.array([[0.0, 180.0, 300.0]])
        pin, _ = cam_with(None).project(pts)
        dist, _ = cam_with(Lens(-0.15, 0.0, 0.0)).project(pts)
        c = np.array([160.0, 90.0])
        self.assertLess(np.linalg.norm(dist - c), np.linalg.norm(pin - c))

    def test_undistort_inverts_distort(self):
        lens = Lens(-0.2, 0.03, 0.0)
        ys, xs = np.mgrid[0:180:7, 0:320:9]
        uv = np.stack([xs + 0.5, ys + 0.5], axis=-1).astype(float)
        back, ok = lens.undistort(lens.distort(uv, 160, 90, 207.8), 160, 90, 207.8)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(back, uv, atol=1e-3)

    def test_points_past_the_fold_are_nan(self):
        lens = Lens(-0.4, 0.0, 0.0)
        far = lens.distort(np.array([[160.0 + 207.8 * 3.0, 90.0]]), 160, 90, 207.8)
        self.assertTrue(np.isnan(far).all())

    def test_image_circle(self):
        lens = Lens(0.0, 0.0, 0.5)
        self.assertTrue(lens.inside_circle(np.array([160.0, 90.0]), 320, 180))
        self.assertFalse(lens.inside_circle(np.array([2.0, 2.0]), 320, 180))
        self.assertFalse(bool(cam_with(lens).in_image(np.array([2.0, 2.0]))))


class RemapTests(unittest.TestCase):
    @staticmethod
    def blob_renderer(world_pt):
        """A fake backend: a small white square at the pinhole image of one world point, on whatever backplate it is given."""
        def draw(cam, plate):
            rgb = np.asarray(plate.convert("RGB").resize((cam.width, cam.height)), dtype=np.uint8).copy()
            mask = np.zeros((cam.height, cam.width), bool)
            u, v = cam.project(np.array([world_pt]))[0][0]
            x, y = int(round(u)), int(round(v))
            mask[y - 3: y + 4, x - 3: x + 4] = True
            rgb[mask] = 255
            return rgb, mask
        return draw

    def test_mask_lands_where_project_says(self):
        checked = 0
        for lens in (Lens(-0.2, 0.02, 0.0), Lens(-0.35, 0.0, 0.0), Lens(0.1, 0.0, 0.0)):
            cam = cam_with(lens, 320, 180)
            for pt in ([0.0, 120.0, 230.0], [0.0, -90.0, 70.0], [0.0, 10.0, 160.0]):
                uv, _ = cam.project(np.array([pt]))
                if not cam.in_image(uv[0]):
                    continue
                rgb, mask = render_with_lens(self.blob_renderer(pt), cam, Image.new("RGB", (320, 180), (40, 90, 140)))
                ys, xs = np.nonzero(mask)
                self.assertGreater(len(xs), 5, (lens, pt))
                self.assertLess(abs((xs.mean() + 0.5) - uv[0, 0]), 1.2, (lens, pt))
                self.assertLess(abs((ys.mean() + 0.5) - uv[0, 1]), 1.2, (lens, pt))
                self.assertEqual(rgb.shape, (180, 320, 3))
                checked += 1
        self.assertGreaterEqual(checked, 6)

    def test_background_is_distorted_with_the_officer(self):
        """The backplate goes through the same lens: a vertical line far from the centre bends / moves exactly where Lens.distort says."""
        lens = Lens(-0.3, 0.06, 0.0)
        cam = cam_with(lens)
        Wr, Hr = canvas_size(cam)
        self.assertGreater(Wr, cam.width)                                           # the pinhole canvas holds what the distorted frame shows
        plate = Image.new("RGB", (Wr, Hr), (20, 20, 20))
        col = Wr // 2 + 150                                                         # a pinhole column near the right edge of the frame
        ImageDraw.Draw(plate).line([(col, 0), (col, Hr)], fill=(255, 255, 255), width=3)
        out = np.asarray(distort_plate(plate, cam).convert("L"))
        for row in (20, 90, 160):
            u_p = col + 0.5 - Wr / 2 + cam.cx
            v_p = row + 0.5 - Hr / 2 + cam.cy
            ud, vd = lens.distort(np.array([u_p, v_p]), cam.cx, cam.cy, cam.fx)
            x = int(ud)
            self.assertGreater(out[int(vd), x - 2: x + 3].max(), 150, row)
        # and the same plate through the full render path: the officer-free image equals the distorted plate
        rgb, _ = render_with_lens(lambda c, p: (np.asarray(p.convert("RGB").resize((c.width, c.height))), np.zeros((c.height, c.width), bool)), cam, plate)
        np.testing.assert_array_equal(rgb, np.asarray(distort_plate(plate, cam)))

    def test_black_border_has_no_pixels_and_no_mask(self):
        cam = cam_with(Lens(-0.1, 0.0, 0.6))
        full = lambda c, p: (np.full((c.height, c.width, 3), 200, np.uint8), np.ones((c.height, c.width), bool))
        rgb, mask = render_with_lens(full, cam, Image.new("RGB", (320, 180)))
        ys, xs = np.mgrid[0:180, 0:320]
        d = np.hypot(xs + 0.5 - 160, ys + 0.5 - 90)
        r = 0.6 * np.hypot(320, 180) / 2
        self.assertTrue((rgb[d > r + 1] == 0).all())
        self.assertFalse(mask[d > r + 1].any())
        self.assertTrue(mask[d < r - 1].all())
        self.assertTrue((rgb[d < r - 1] > 150).all())

    def test_layer_follows_the_same_mapping(self):
        cam = cam_with(Lens(-0.25, 0.0, 0.0))
        pt = [0.0, 100.0, 220.0]

        def layer(c):
            rgba = np.zeros((c.height, c.width, 4), np.uint8)
            u, v = c.project(np.array([pt]))[0][0]
            rgba[int(v) - 3: int(v) + 4, int(u) - 3: int(u) + 4] = (255, 0, 0, 255)
            return rgba
        out = layer_with_lens(layer, cam)
        ys, xs = np.nonzero(out[..., 3] > 127)
        uv, _ = cam.project(np.array([pt]))
        self.assertLess(abs(xs.mean() + 0.5 - uv[0, 0]), 1.2)
        self.assertLess(abs(ys.mean() + 0.5 - uv[0, 1]), 1.2)
        self.assertTrue((out[out[..., 3] > 250][:, 0] > 250).all())


class ProfileTests(unittest.TestCase):
    profiles = load_json(ROOT / "profiles" / "VN" / "camera_profiles.json", "camera_profiles")["profiles"]

    def test_the_road_user_camera_types_exist_with_a_lens(self):
        by = {p["id"]: p for p in self.profiles}
        for pid in ("driver_eye", "motorbike_rider", "dashcam"):
            self.assertIn(pid, by)
            self.assertIn("lens", by[pid])
            self.assertLess(by[pid]["lens"]["k1"][1], 0.0, pid)        # all of them are barrel lenses

    def test_mounting_heights_are_realistic(self):
        """In-car and motorbike cameras sit at seated eye height: the Studio presets are the mid of these ranges."""
        by = {p["id"]: p for p in self.profiles}
        self.assertLessEqual(max(by["dashcam"]["height_cm"]), 135)
        self.assertGreaterEqual(min(by["dashcam"]["height_cm"]), 100)
        self.assertLessEqual(max(by["driver_eye"]["height_cm"]), 145)
        self.assertLessEqual(max(by["motorbike_rider"]["height_cm"]), 140)
        self.assertNotIn("cctv_elevated", by)                      # removed (D40)
        for pid in ("dashcam", "driver_eye", "motorbike_rider"):
            self.assertLessEqual(max(by[pid]["roll_deg"]), 5.0)

    def test_a_fitted_lens_always_reaches_the_frame_corners(self):
        """A distortion too strong for the field of view is weakened (never left with unreachable, black corners)."""
        for p in self.profiles:
            spec = p.get("lens")
            if not spec:
                continue
            for f in p["focal_length_mm"]:
                fx = f * 1280 / p["sensor_width_mm"]
                for k1 in spec["k1"]:
                    for k2 in spec["k2"]:
                        lens = Lens(k1, k2, 0.0).fitted(1280, 720, fx)
                        self.assertGreaterEqual(lens.r_reach(), np.hypot(640, 360) / fx, f"{p['id']} f={f} k1={k1} k2={k2}")
                        self.assertLessEqual(abs(lens.k1), abs(k1) + 1e-9)

    def test_fitting_weakens_only_what_is_too_strong(self):
        mild = Lens(-0.05, 0.0, 0.0)
        self.assertIs(mild.fitted(1280, 720, 1280 * 3.6 / 6.17), mild)
        strong = Lens(-0.6, 0.0, 0.0)
        fit = strong.fitted(1280, 720, 1280 * 3.0 / 6.17)
        self.assertGreater(fit.k1, -0.6)
        self.assertLess(fit.k1, 0.0)

    def test_sampling_is_deterministic_and_has_its_own_stream(self):
        spec = {p["id"]: p for p in self.profiles}["dashcam"]["lens"]
        a = sample_lens(spec, rngmod.stream(5, "lens"))
        b = sample_lens(spec, rngmod.stream(5, "lens"))
        c = sample_lens(spec, rngmod.stream(6, "lens"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertIsNone(sample_lens(None, rngmod.stream(5, "lens")))
        self.assertIsNone(sample_lens({"k1": [0, 0], "k2": [0, 0], "image_circle": [0, 0]}, rngmod.stream(5, "lens")))
        mix = {"k1": [-0.2, -0.2], "k2": [0.05, 0.05], "image_circle": [0.8, 0.9], "black_border_probability": 0.5}
        circles = [sample_lens(mix, rngmod.stream(i, "lens")).image_circle for i in range(200)]
        self.assertTrue(0.3 < sum(c > 0 for c in circles) / 200 < 0.7)               # about half the samples have the black border, half do not
        self.assertTrue(all(0.8 <= c <= 0.9 for c in circles if c > 0))


class StudioLensTests(unittest.TestCase):
    def test_pointer_ray_inverts_the_distorted_projection(self):
        cam = core.make_camera({"distance_cm": 500, "lens_k1": -0.2, "lens_k2": 0.02}, dict(core.DEFAULT_OFFICER))
        self.assertIsNotNone(cam.lens)
        world = np.array([30.0, 60.0, 180.0])
        uv, _ = cam.project(world[None])
        back = core.camera_plane_target(cam, world, uv[0])
        np.testing.assert_allclose(back, world, atol=0.2)

    def test_summary_carries_the_lens_and_the_distorted_horizon(self):
        officer = dict(core.DEFAULT_OFFICER)
        plain = core.camera_summary(core.make_camera({"distance_cm": 500, "pitch_offset_deg": -8}, officer))
        lensed = core.camera_summary(core.make_camera({"distance_cm": 500, "pitch_offset_deg": -8, "lens_k1": -0.2}, officer))
        self.assertIsNone(plain["lens"])
        self.assertEqual(lensed["lens"]["k1"], -0.2)
        self.assertNotAlmostEqual(plain["horizon_v"], lensed["horizon_v"], places=1)

    def test_studio_default_camera_has_no_distortion(self):
        import tempfile as _t
        from tosdg.studio.server import StudioApp
        cam = StudioApp(workspace=Path(_t.mkdtemp())).default_camera()
        self.assertEqual((cam["lens_k1"], cam["lens_k2"], cam["lens_circle"]), (0.0, 0.0, 0.0))
        self.assertEqual((cam["height_cm"], cam["aim_height_cm"], cam["distance_cm"]), (150.0, 150.0, 400.0))
        self.assertIsNone(core.make_camera(cam, dict(core.DEFAULT_OFFICER)).lens)

    def test_auto_border_follows_the_lens(self):
        """With lens_border_auto the black border is computed from k1, k2, focal length and size; the manual value is ignored."""
        fx = 1280 * 3.6 / 6.17
        weak = Lens.auto_circle(-0.1, 0.0, 1280, 720, fx)
        strong = Lens.auto_circle(-0.3, 0.05, 1280, 720, fx)
        self.assertTrue(0.0 < strong < weak < 1.0)
        self.assertEqual(Lens.auto_circle(0.0, 0.0, 1280, 720, fx), 0.0)
        self.assertEqual(Lens.auto_circle(0.2, 0.0, 1280, 720, fx), 0.0)         # pincushion: no border
        wide = Lens.auto_circle(-0.3, 0.05, 1280, 720, 1280 * 2.8 / 6.17)       # shorter focal length (wider view) -> stronger effect
        self.assertLess(wide, strong)
        base = {"distance_cm": 400, "focal_length_mm": 3.6, "lens_k1": -0.3, "lens_k2": 0.05, "lens_circle": 0.5}
        auto = core.make_camera({**base, "lens_border_auto": 1}, dict(core.DEFAULT_OFFICER))
        manual = core.make_camera(base, dict(core.DEFAULT_OFFICER))
        self.assertAlmostEqual(auto.lens.image_circle, strong, places=3)
        self.assertEqual(manual.lens.image_circle, 0.5)
        self.assertAlmostEqual(core.camera_summary(auto)["lens"]["circle"], strong, places=3)

    def test_eval_result_is_strict_json(self):
        """The browser's JSON.parse rejects Infinity / NaN: a lens that never folds must not put them in the summary."""
        from test_studio import default_pose
        for k in ({"lens_k1": -0.13, "lens_k2": 0.035}, {"lens_k1": -0.3, "lens_k2": 0.0}, {"lens_k1": -0.2, "lens_circle": 0.8}):
            ev = core.evaluate(skeleton(), default_pose(), dict(core.DEFAULT_OFFICER), {"distance_cm": 400, "focal_length_mm": 3.6, **k})
            json.dumps(ev, allow_nan=False)

    def test_evaluate_projects_through_the_lens(self):
        from test_studio import default_pose
        sk, officer = skeleton(), dict(core.DEFAULT_OFFICER)
        base = {"distance_cm": 400, "azimuth_deg": 0, "height_cm": 150, "aim_height_cm": 150, "focal_length_mm": 3.6}
        a = core.evaluate(sk, default_pose(), officer, base)
        b = core.evaluate(sk, default_pose(), officer, {**base, "lens_k1": -0.2})
        centre = np.array([640.0, 360.0])
        ua, ub = np.array(a["joints"]["hand_r"]["uv"]), np.array(b["joints"]["hand_r"]["uv"])
        self.assertLess(np.linalg.norm(ub - centre), np.linalg.norm(ua - centre))


class GenerationTests(unittest.TestCase):
    def test_every_camera_type_generates_valid_distorted_samples_with_black_borders(self):
        for pid in ("driver_eye", "motorbike_rider", "dashcam"):
            self.check_camera_type(pid)

    def check_camera_type(self, pid):
        with tempfile.TemporaryDirectory() as td:
            ctx = GenerationContext(make_config(Path(td), backend="proxy", uniform_brightness_range=[1.0, 1.0], num_samples=6, name=pid, camera_profile_weights={pid: 1.0}))
            out = Path(td) / "o"
            man = run(ctx, out, log=lambda *_: None)
            self.assertEqual(man["failures"], [], pid)
            self.assertEqual(validate_dataset(out), [], pid)
            seen = False
            for s in man["samples"]:
                meta = json.loads((out / s["metadata"]).read_text())
                self.assertEqual(meta["schema_version"], "1.8")
                self.assertEqual(meta["camera"]["profile"], pid)
                lens = meta["camera"]["lens"]
                self.assertEqual(lens["model"], "brown_radial_k1k2")
                self.assertLess(lens["k1"], 0.0, pid)                      # every type is a barrel lens
                circ = meta["camera"]["lens"]["image_circle_frac"]
                if circ <= 0:
                    continue
                rgb = np.asarray(Image.open(out / meta["files"]["rgb"]).convert("RGB"))
                mask = np.asarray(Image.open(out / meta["files"]["mask"])) > 0
                ys, xs = np.mgrid[0:180, 0:320]
                far = np.hypot(xs + 0.5 - 160, ys + 0.5 - 90) > circ * np.hypot(320, 180) / 2 + 1.5
                if far.any():
                    seen = True
                    self.assertTrue((rgb[far] == 0).all())
                    self.assertFalse(mask[far].any())
            self.assertTrue(seen, f"{pid}: no sample had a black border")

    def test_same_seed_same_lens(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = GenerationContext(make_config(Path(td), backend="proxy", uniform_brightness_range=[1.0, 1.0], num_samples=2))
            a = run(ctx, Path(td) / "a", log=lambda *_: None)
            b = run(ctx, Path(td) / "b", log=lambda *_: None)
            self.assertEqual([x["sha256"] for x in a["samples"]], [x["sha256"] for x in b["samples"]])


if __name__ == "__main__":
    unittest.main()
