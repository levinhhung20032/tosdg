"""Where the officer stands and which way they face (junction.py, officer_yaw_mode "junction"), roundabouts (podium / baton) and
the officer darkened to the photo (match_backplate_light).

Contents:
  RuleTests (6 tests)
  GenerationTests (5 tests)
  LightTests (2 tests)
  CliTests (1 tests)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import make_config
from test_cli import FAST, run_cli
from test_people import generate, library_with_people
from tosdg import junction as jn
from tosdg.backgrounds import BackgroundLibrary
from tosdg.scene import GenerationContext, match_light

STOP_LEFT = {"instructions": {"front": "NONE", "back": "NONE", "left": "STOP", "right": "NONE"}}
ARMS_OUT = {"instructions": {"front": "STOP", "back": "STOP", "left": "PROCEED", "right": "PROCEED"}}
NONE = {"instructions": {"front": "NONE", "back": "NONE", "left": "NONE", "right": "NONE"}}


def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


class RuleTests(unittest.TestCase):
    def test_cross_and_roundabout_face_a_road(self):
        arms = [10.0, 100.0, -170.0, -80.0]
        self.assertEqual(jn.candidate_yaws("cross", arms), arms)
        self.assertEqual(jn.candidate_yaws("roundabout", arms), arms)

    def test_t_faces_the_stem_and_sees_all_three_roads(self):
        arms = [0.0, 180.0, 90.0]                                 # bar along x, stem to +y
        (y,) = jn.candidate_yaws("T", arms)
        self.assertAlmostEqual(y, 90.0)
        self.assertEqual(jn._in_view(arms, y), 3)
        self.assertEqual(jn.sides_of(arms, y), {"front", "left", "right"})

    def test_y_faces_a_road_or_keeps_two_in_view(self):
        arms = [-150.0, 20.0, 60.0]
        c = jn.candidate_yaws("Y", arms)
        self.assertTrue(set(arms) <= set(c))
        self.assertTrue(all(y in arms or jn._in_view(arms, y) >= 2 for y in c))
        self.assertEqual(jn.candidate_yaws("T", [0.0, 175.0, 95.0]), [95.0])        # marked a little off: still the stem
        self.assertIn(40.0, [round(y, 6) for y in c])                     # between the two branches

    def test_required_sides(self):
        self.assertEqual(jn.required_sides(STOP_LEFT), {"left"})
        self.assertEqual(jn.required_sides(ARMS_OUT), {"front", "left", "right"})     # 4 sides: the back may be a wall
        self.assertEqual(jn.required_sides(NONE), set())

    def test_the_gesture_picks_a_facing_whose_sides_are_roads(self):
        rng = np.random.default_rng(0)
        for _ in range(20):                                       # straight road along x: STOP_LEFT needs a road on the left
            y, r = jn.choose_yaw("straight", [0.0, 180.0], STOP_LEFT, (500.0, 0.0), (0.0, 0.0), (-180, 180), rng, (0.0, 0.0))
            self.assertIn(round(float(wrap(y)) % 180.0), (90,))
            self.assertEqual(r["required_met"], ["left"])

    def test_camera_side_is_preferred(self):
        rng = np.random.default_rng(1)
        arms = [0.0, 90.0, 180.0, -90.0]
        for _ in range(10):                                       # dashcam-like: the camera (at +x) in front of the officer
            y, r = jn.choose_yaw("cross", arms, NONE, (800.0, 0.0), (0.0, 0.0), (-30, 30), rng, (0.0, 0.0))
            self.assertAlmostEqual(float(wrap(y)), 0.0)


def junction_library(td: Path, kind: str = "cross") -> Path:
    """library_with_people with fx_day_clear (480x270, horizon 0.42) given a junction in front of the camera."""
    lib_dir = library_with_people(td)
    lib = BackgroundLibrary(lib_dir)
    arms = {"cross": [-90, 90, 0, 180], "T": [-90, 90, 180], "Y": [-30, 35, 180], "roundabout": [-90, 90, 0, 180]}[kind]
    lib.set_junction("fx_day_clear", {"type": kind, "center": [240, 205], "arms_deg": arms})
    lib.save()
    return lib_dir


class GenerationTests(unittest.TestCase):
    def gen(self, td: Path, kind: str = "cross", n: int = 4, **over):
        return generate(td, num_samples=n, background_library=str(junction_library(td, kind)), size_reference="auto",
                        officer_yaw_mode="junction", background_ids=["fx_day_clear"], lens_distortion=False, **over)

    def test_the_officer_stands_in_the_middle_facing_a_road(self):
        with tempfile.TemporaryDirectory() as td:
            for m in self.gen(Path(td)):
                j = m["background"]["junction"]
                self.assertTrue(j["at_center"])
                self.assertLessEqual(np.hypot(*j["offset_cm"]), jn.JITTER_CM + 1e-6)
                self.assertAlmostEqual(m["officer"]["world_position"][0], j["offset_cm"][0], delta=0.05)
                self.assertLessEqual(min(abs(wrap(j["arms_azimuth_deg"]))), 5.0 + 1e-6)     # facing one road (+- jitter)
                self.assertEqual(j["required_met"], j["required_sides"])
                c = m["camera"]
                self.assertAlmostEqual(c["position"][2], m["background"]["size_reference"]["camera_height_cm"], delta=0.01)

    def test_the_middle_is_seen_where_it_was_marked(self):
        """The junction's middle (world origin) projects on its photo pixel through the composition."""
        from tosdg.camera import Camera
        with tempfile.TemporaryDirectory() as td:
            for m in self.gen(Path(td), n=2):
                c, comp = m["camera"], m["background"]["composition"]
                cam = Camera(c["profile"], tuple(c["position"]), c["pitch"], c["yaw"], c["roll"], c["focal_length_mm"],
                             c["sensor_width_mm"], c["width"], c["height"])
                uv, _ = cam.project(np.zeros((1, 3)))
                s, (ox, oy) = comp["scale"], comp["offset_px"]
                self.assertAlmostEqual(uv[0, 0], 240 * s + ox, delta=1.0)
                self.assertAlmostEqual(uv[0, 1], 205 * s + oy, delta=1.0)

    def test_t_junction_keeps_the_three_roads_in_view(self):
        with tempfile.TemporaryDirectory() as td:
            for m in self.gen(Path(td), "T"):
                self.assertEqual(m["background"]["junction"]["in_view"], 3)

    def test_roundabout_baton_is_drawn(self):
        with tempfile.TemporaryDirectory() as td:
            metas = self.gen(Path(td), "roundabout", n=3, roundabout={"baton_probability": 1.0, "podium_probability": 0.0})
            for m in metas:
                self.assertIsNotNone(m["officer"].get("baton"))
            (Path(td) / "b").mkdir()
            metas = self.gen(Path(td) / "b", "roundabout", n=3, roundabout={"baton_probability": 0.0, "podium_probability": 0.0})
            for m in metas:
                self.assertFalse((m["officer"].get("baton") or {}).get("held", False))

    def test_junction_mode_needs_auto(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "junction"):
                GenerationContext(make_config(Path(td), schema_version="1.6", officer_yaw_mode="junction", **FAST))


class LightTests(unittest.TestCase):
    LIGHT = {"exposure_ev": 0.1, "sun_intensity": 1.0, "ambient_intensity": 0.4, "color_temperature_k": 6000.0, "sun_color_rgb": [1, 1, 1]}

    def test_a_dark_photo_darkens_the_officer_a_bright_one_does_not(self):
        uv = np.array([[40.0, 20.0], [60.0, 80.0]])
        dark = match_light(self.LIGHT, Image.new("RGB", (100, 100), (20, 20, 25)), uv, {"tags": ["night"]})
        self.assertAlmostEqual(dark["backplate_exposure_ev"], -2.0)          # darkened, but never below LIGHT_MIN_EV
        self.assertAlmostEqual(dark["exposure_ev"], 0.1 + dark["backplate_exposure_ev"])
        self.assertLess(dark["sun_intensity"], 0.5)
        self.assertAlmostEqual(dark["ambient_intensity"], 0.4 * 0.6)
        dim = match_light(self.LIGHT, Image.new("RGB", (100, 100), (10, 10, 10)).copy(), uv, {"tags": []})
        self.assertLess(dim["backplate_exposure_ev"], dark["backplate_exposure_ev"] + 1e-9)          # darker photo, darker officer
        self.assertLessEqual(dark["color_temperature_k"], 3400.0)
        day = match_light(self.LIGHT, Image.new("RGB", (100, 100), (200, 200, 200)), uv, {"tags": ["day"]})
        self.assertEqual((day["backplate_exposure_ev"], day["exposure_ev"], day["sun_intensity"]), (0.0, 0.1, 1.0))

    def test_generation_records_it(self):
        with tempfile.TemporaryDirectory() as td:
            for m in generate(Path(td), num_samples=2, match_backplate_light=True):
                self.assertIn("backplate_exposure_ev", m["lighting"])


class CliTests(unittest.TestCase):
    def test_junction_set_and_clear(self):
        with tempfile.TemporaryDirectory() as td:
            lib_dir = library_with_people(Path(td))
            rc, out, _ = run_cli("backgrounds", "junction", lib_dir, "fx_dusk", "--type", "T", "--frac", "--center", "0.5", "0.8",
                                 "--road", "-90", "--road", "90", "--road", "180")
            self.assertEqual(rc, 0, out)
            j = BackgroundLibrary(lib_dir).by_id["fx_dusk"]["junction"]
            self.assertEqual((j["type"], j["center"], j["arms_deg"]), ("T", [240.0, 216.0], [-90.0, 90.0, 180.0]))
            rc, _, _ = run_cli("backgrounds", "junction", lib_dir, "fx_dusk", "--type", "T", "--center", "1", "1", "--road", "0")
            self.assertEqual(rc, 2)                                       # a T needs three roads
            rc, out, _ = run_cli("backgrounds", "junction", lib_dir, "fx_dusk", "--clear")
            self.assertNotIn("junction", BackgroundLibrary(lib_dir).by_id["fx_dusk"])


if __name__ == "__main__":
    unittest.main()
