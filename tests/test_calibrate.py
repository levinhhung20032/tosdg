"""The camera of a backplate told from objects of known height in it (calibrate.py, size_reference "auto").

Contents:
  FitTests (5 tests)
  CalibrateOrderTests (5 tests)
  GeometryTests (3 tests)
  GenerationTests (3 tests)
  CliTests (1 tests)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_cli import run_cli
from test_people import generate, library_with_people
from tosdg import calibrate as cal
from tosdg.backgrounds import BackgroundLibrary
from tosdg.camera import Camera

IMG_W, IMG_H, FY = 1920, 1080, 1000.0


def photo_camera(height_cm: float = 140.0, pitch: float = -4.0) -> Camera:
    """A camera with a 1000 px focal length looking along +X."""
    return Camera("photo", (0.0, 0.0, height_cm), pitch, 0.0, 0.0, FY * 6.17 / IMG_W, 6.17, IMG_W, IMG_H)


def seen(camera: Camera, kind: str, x_cm: float, y_cm: float = 0.0, scale: float = 1.0) -> dict:
    """An object of the kind's height (x scale) standing at (x, y), as the photo camera sees it (exact projection)."""
    H = cal.PRIORS[kind][0] * scale
    uv, _ = camera.project(np.array([[x_cm, y_cm, 0.0], [x_cm, y_cm, H]]))
    (u, vf), (_, vh) = uv
    w = 0.3 * (vf - vh)
    return {"kind": kind, "feet": [float(u), float(vf)], "height_px": float(vf - vh), "bbox": [u - w, vh, u + w, vf], "score": 0.9}


def entry(people=(), objects=(), horizon=None, **extra) -> dict:
    e = {"id": "x", "width": IMG_W, "height": IMG_H, "tags": ["day"], "people": {"method": "manual", "found": list(people)},
         "objects": {"method": "manual", "found": list(objects)}, **extra}
    if horizon is not None:
        e["horizon_y_frac"], e["horizon_source"] = horizon, "manual"
    return e


class FitTests(unittest.TestCase):
    def test_people_at_different_distances_give_the_camera(self):
        cam = photo_camera(140.0, -4.0)
        objs = cal.reference_objects(entry([seen(cam, "standing", x, y) for x, y in ((700, -150), (1500, 200), (3000, 50), (6000, -300))]))
        fit, why = cal.fit_ground(objs, IMG_H)
        self.assertEqual(why, "ok")
        self.assertAlmostEqual(fit["camera_height_cm"], 140.0, delta=140.0 * 0.03)        # small-pitch model: within 3 %
        self.assertAlmostEqual(fit["horizon_row"], cam.horizon_v, delta=0.005 * IMG_H)

    def test_people_on_one_row_are_not_enough(self):
        cam = photo_camera()
        objs = cal.reference_objects(entry([seen(cam, "standing", 2000, y) for y in (-300, 0, 300)]))
        fit, why = cal.fit_ground(objs, IMG_H)
        self.assertIsNone(fit)
        self.assertRegex(why, "one row|too unsure|agree")

    def test_a_child_is_left_out(self):
        cam = photo_camera()
        people = [seen(cam, "standing", x) for x in (800, 1600, 3200, 5000)] + [seen(cam, "standing", 1200, 100, scale=0.6)]
        fit, _ = cal.fit_ground(cal.reference_objects(entry(people)), IMG_H)
        self.assertEqual(len(fit["inliers"]), 4)
        self.assertAlmostEqual(fit["camera_height_cm"], 140.0, delta=5.0)

    def test_two_riders_on_one_bike_count_once(self):
        cam = photo_camera()
        a = {**seen(cam, "rider", 1500), "kind": "rider"}
        b = {**a, "height_px": a["height_px"] * 0.95}
        self.assertEqual(len(cal.reference_objects(entry([a, b]))), 1)

    def test_fewer_than_three(self):
        cam = photo_camera()
        fit, why = cal.fit_ground(cal.reference_objects(entry([seen(cam, "standing", x) for x in (800, 3000)])), IMG_H)
        self.assertIsNone(fit)
        self.assertIn("3 needed", why)


class CalibrateOrderTests(unittest.TestCase):
    def setUp(self):
        self.cam = photo_camera(150.0, -2.0)

    def test_three_people_win_over_the_horizon(self):
        c = cal.calibrate(entry([seen(self.cam, "standing", x) for x in (800, 2000, 5000)], horizon=0.2), 155.0)
        self.assertEqual(c["method"], "people")
        self.assertAlmostEqual(c["horizon_y_frac"], self.cam.horizon_v / IMG_H, delta=0.005)

    def test_vehicles_help_when_people_are_too_few(self):
        people = [seen(self.cam, "standing", 900)]
        cars = [seen(self.cam, "car", 1800, 300), seen(self.cam, "bus", 4000, -400), seen(self.cam, "motorbike", 2600, 200)]
        c = cal.calibrate(entry(people, cars, horizon=0.3), 155.0)
        self.assertEqual(c["method"], "objects")
        self.assertAlmostEqual(c["camera_height_cm"], 150.0, delta=8.0)
        self.assertEqual(c["objects_used"], 4)

    def test_a_barrier_marked_by_hand_with_its_height(self):
        b = {**seen(self.cam, "barrier", 2500, scale=1.2), "height_cm": 120.0, "manual": True}
        c = cal.calibrate(entry([seen(self.cam, "standing", x) for x in (900, 5000)], [b], horizon=0.3), 155.0)
        self.assertEqual(c["method"], "objects")
        self.assertAlmostEqual(c["camera_height_cm"], 150.0, delta=8.0)

    def test_too_few_objects_fall_back_to_the_horizon(self):
        true_h = self.cam.horizon_v / IMG_H
        c = cal.calibrate(entry([seen(self.cam, "standing", 1500)], horizon=true_h), 155.0)
        self.assertEqual((c["method"], c["camera_height_from"]), ("horizon", "objects"))   # height measured at that horizon
        self.assertAlmostEqual(c["camera_height_cm"], 150.0, delta=5.0)
        c = cal.calibrate(entry(horizon=0.4), 125.0)
        self.assertEqual((c["method"], c["camera_height_cm"], c["camera_height_from"]), ("horizon", 125.0, "default"))

    def test_nothing_known(self):
        self.assertEqual(cal.calibrate(entry(), 155.0)["method"], "none")


class GeometryTests(unittest.TestCase):
    def test_pitch_puts_the_horizon_on_the_row(self):
        for row in (100.0, 540.0, 900.0):
            cam = photo_camera(130.0, cal.pitch_for_horizon(row, IMG_H / 2, FY))
            self.assertAlmostEqual(cam.horizon_v, row, places=6)

    def test_gap_and_distance_are_exact(self):
        cam = photo_camera(130.0, -6.0)
        for d in (300.0, 1200.0, 5000.0):
            uv, _ = cam.project(np.array([[d, 0.0, 0.0]]))
            g = cal.gap_for_distance(d, 130.0, -6.0, FY)
            self.assertAlmostEqual(uv[0, 1] - cam.horizon_v, g, places=6)
            self.assertAlmostEqual(cal.distance_for_gap(g, 130.0, -6.0, FY), d, places=6)

    def test_feasible_gap_and_free_offsets(self):
        lo, hi = cal.feasible_gap(200.0, 150.0, 183.0, (10.0, 530.0), 60.0)
        self.assertAlmostEqual(lo, 60.0 * 150.0 / 183.0)
        self.assertLessEqual(200.0 + hi, 530.0)
        self.assertGreaterEqual(200.0 + hi - hi * 183.0 / 150.0, 10.0 - 1e-9)           # the head stays in the frame
        self.assertIsNone(cal.feasible_gap(500.0, 150.0, 183.0, (10.0, 530.0), 60.0))  # horizon too low for 60 px
        near = {"bbox": [100.0, 150.0, 140.0, 300.0], "v": 300.0}
        far = {"bbox": [300.0, 150.0, 320.0, 210.0], "v": 210.0}
        cands = np.arange(-50, 1)
        for o, blocks in ((near, True), (far, False)):
            ok = cal.free_offsets(cands, 1.0, 0, [o], 250.0, (100.0, 250.0), (110.0, 130.0) if blocks else (300.0, 320.0),
                                  150.0, 540.0, np.zeros((0, 2)), None, (500, 540), 450, 540)
            self.assertEqual(len(ok) < len(cands), blocks)


class GenerationTests(unittest.TestCase):
    def test_the_officer_has_the_size_of_a_person_at_that_spot(self):
        """A 165 cm person at the officer's ground point is as tall in the picture as the photo's people at that row."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            metas = generate(td, num_samples=4, background_library=str(library_with_people(td)), size_reference="auto",
                             lens_distortion=False, background_ids=["fx_day_clear", "fx_dusk"])
            for m in metas:
                r, c = m["background"]["size_reference"], m["camera"]
                self.assertEqual(r["method"], "people")
                cam = Camera(c["profile"], tuple(c["position"]), c["pitch"], c["yaw"], c["roll"], c["focal_length_mm"],
                             c["sensor_width_mm"], c["width"], c["height"])
                uv, _ = cam.project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 165.0]]))
                person_px = uv[0, 1] - uv[1, 1]
                photo_px = 165.0 * (uv[0, 1] - m["background"]["composition"]["camera_horizon_v"]) / r["camera_height_cm"]
                self.assertAlmostEqual(person_px, photo_px, delta=0.03 * photo_px)

    def test_min_officer_height_with_auto(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            metas = generate(td, num_samples=3, background_library=str(library_with_people(td)), size_reference="auto",
                             min_officer_height_px=90)
            for m in metas:
                b = m["officer"]["bbox"]["xyxy"]
                self.assertGreaterEqual(b[3] - b[1], 90)

    def test_locked_camera_keeps_its_height_and_aligns_the_calibrated_horizon(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ov = {"distance_cm": [900, 900], "height_cm": [120, 120], "azimuth_deg": [0, 0], "aim_height_cm": [110, 110]}
            for m in generate(td, num_samples=2, background_library=str(library_with_people(td)), size_reference="auto",
                              camera_overrides=ov, background_ids=["fx_dusk"]):
                r = m["background"]["size_reference"]
                self.assertTrue(r["camera_locked"])
                self.assertEqual(m["camera"]["position"][2], 120.0)


class CliTests(unittest.TestCase):
    def test_objects_and_calibrate(self):
        with tempfile.TemporaryDirectory() as td:
            lib_dir = library_with_people(Path(td))
            rc, out, _ = run_cli("backgrounds", "objects", lib_dir, "fx_dusk", "--add", "barrier", "10", "200", "60", "240",
                                 "--height-cm", "90")
            self.assertEqual(rc, 0, out)
            o = BackgroundLibrary(lib_dir).by_id["fx_dusk"]["objects"]["found"]
            self.assertEqual((o[0]["kind"], o[0]["height_cm"], o[0]["feet"][1], o[0]["manual"]), ("barrier", 90.0, 240.0, True))
            rc, out, _ = run_cli("backgrounds", "objects", lib_dir, "fx_dusk", "--add", "tree", "1", "2", "3", "4")
            self.assertEqual(rc, 2)
            rc, out, _ = run_cli("backgrounds", "calibrate", lib_dir)
            self.assertEqual(rc, 0, out)
            self.assertRegex(out, r"fx_dusk: people\s+camera\s+165\.0 cm")
            self.assertRegex(out, r"fx_elevated: horizon")


if __name__ == "__main__":
    unittest.main()
