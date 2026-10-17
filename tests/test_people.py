"""Reference people of a photo (people.py) and the camera told from them (size_reference "auto"), the officer facing along /
across the road (officer_yaw_mode "road"), the Studio random-generate locks (camera_overrides, background_ids) and the removed CCTV camera.

Contents:
  StandingTests (1 tests)
  LibraryTests (2 tests)
  GenerationTests (5 tests)
  OcclusionTests (3 tests)
  BackgroundSetTests (1 tests)
  StudioRandomTests (2 tests)
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import ROOT, fixture_library, make_config
from test_cli import FAST, run_cli
from tosdg.backgrounds import BackgroundLibrary
from tosdg.generate import run
from tosdg.people import _standing
from tosdg.scene import GenerationContext
from tosdg.schema import load_json


def keypoints(ank_y: float, knee_y: float = 300.0, conf: float = 0.9) -> np.ndarray:
    kp = np.zeros((17, 3))
    for i, (x, y) in {5: (40, 100), 6: (60, 100), 11: (42, 200), 12: (58, 200), 13: (42, knee_y), 14: (58, knee_y),
                      15: (42, ank_y), 16: (58, ank_y)}.items():
        kp[i] = (x, y, conf)
    return kp


class StandingTests(unittest.TestCase):
    def test_standing_needs_long_straight_legs_and_seen_ankles(self):
        self.assertEqual(_standing(keypoints(400)), (True, 400.0))
        self.assertFalse(_standing(keypoints(250, knee_y=230))[0])        # sitting: legs short against the torso
        self.assertFalse(_standing(keypoints(400, conf=0.2))[0])          # nothing seen well enough

def library_without_people(td: Path) -> Path:
    """The fixture library copied, every photo searched for people and none found: size_reference people / auto never
    run the detector on it (and never touch the shared fixture library)."""
    lib_dir = td / "lib0"
    shutil.copytree(fixture_library(), lib_dir)
    lib = BackgroundLibrary(lib_dir)
    for b in lib.index["backgrounds"]:
        lib.set_people(b["id"], [])
        lib.set_objects(b["id"], [])
    lib.save()
    return lib_dir


def library_with_people(td: Path) -> Path:
    """The fixture library, each ground-level photo with three standing people on the road at different distances."""
    lib_dir = td / "lib"
    shutil.copytree(fixture_library(), lib_dir)
    lib = BackgroundLibrary(lib_dir)
    for b in lib.index["backgrounds"]:
        if "ground_level" in b["tags"]:
            W, H = b["width"], b["height"]
            hz, found = H * b["horizon_y_frac"], []
            for u, d in ((0.15, 0.08), (0.5, 0.2), (0.85, 0.45)):            # near, middle and far (height ~ feet row - horizon)
                feet_v = hz + d * (H - hz)
                h = feet_v - hz
                found.append({"kind": "standing", "feet": [W * u, feet_v], "height_px": h,
                              "bbox": [W * u - h / 6, feet_v - h, W * u + h / 6, feet_v], "score": 0.9})
            lib.set_people(b["id"], found)
        else:
            lib.set_people(b["id"], [])
        lib.set_objects(b["id"], [])                                         # searched: no vehicle (auto runs no detector)
    lib.save()
    return lib_dir


class LibraryTests(unittest.TestCase):
    def test_people_are_stored_and_valid(self):
        with tempfile.TemporaryDirectory() as td:
            lib_dir = library_with_people(Path(td))
            lib = BackgroundLibrary(lib_dir)                                # loads against the schema (library 1.2)
            self.assertEqual(lib.index["schema_version"], "1.3")
            self.assertEqual(lib.by_id["fx_dusk"]["people"]["method"], "manual")
            self.assertEqual(lib.integrity_errors(), [])
            rc, out, _ = run_cli("backgrounds", "people", lib_dir, "--id", "fx_dusk", "--clear")
            self.assertEqual(rc, 0, out)
            self.assertNotIn("people", BackgroundLibrary(lib_dir).by_id["fx_dusk"])

    def test_cctv_camera_is_gone(self):
        profiles = load_json(ROOT / "profiles" / "VN" / "camera_profiles.json", "camera_profiles")["profiles"]
        self.assertEqual(sorted(p["id"] for p in profiles), ["dashcam", "driver_eye", "motorbike_rider", "roadside"])


def generate(td: Path, **over) -> list[dict]:
    cfg = make_config(td, schema_version="1.6", max_attempts_per_sample=60, **{**FAST, **over})
    man = run(GenerationContext(cfg), td / "ds", log=lambda *_: None)
    return [json.loads((td / "ds" / s["metadata"]).read_text()) for s in man["samples"]]


class GenerationTests(unittest.TestCase):
    def test_auto_tells_the_camera_from_the_people_not_the_horizon(self):
        """auto (calibrate.py): three people at different distances give the photo's camera height (165 cm here) and horizon,
        even when the stored horizon is wrong; the virtual camera stands there and its horizon is the photo's."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            lib_dir = library_with_people(td)
            idx = json.loads((lib_dir / "index.json").read_text())
            true_h = {}
            for b in idx["backgrounds"]:
                true_h[b["id"]] = b["horizon_y_frac"]
                if "ground_level" in b["tags"]:
                    b["horizon_y_frac"] = 0.25                                # wrong, and "set by hand"
            (lib_dir / "index.json").write_text(json.dumps(idx))
            metas = generate(td, num_samples=4, background_library=str(lib_dir), size_reference="auto", lens_distortion=False)
            self.assertEqual(len(metas), 4)
            for m in metas:
                r = m["background"]["size_reference"]
                self.assertEqual(r["mode"], "calibrated")
                if r["method"] == "people":
                    self.assertAlmostEqual(r["camera_height_cm"], 165.0, delta=0.5)
                    self.assertAlmostEqual(r["horizon_y_frac"], true_h[m["background"]["id"]], delta=0.002)
                else:                                                          # fx_elevated: no people, its horizon
                    self.assertEqual((r["method"], r["camera_height_from"]), ("horizon", "default"))
                self.assertAlmostEqual(m["camera"]["position"][2], r["camera_height_cm"], delta=0.01)
                c = m["background"]["composition"]
                plate_h = round(270 * c["scale"])
                self.assertTrue(c["horizon_aligned"])
                self.assertAlmostEqual(r["horizon_y_frac"] * plate_h + c["offset_px"][1], c["camera_horizon_v"], delta=1.0)

    def test_no_lens_distortion(self):
        with tempfile.TemporaryDirectory() as td:
            for m in generate(Path(td), num_samples=3, lens_distortion=False, camera_profile_weights={"dashcam": 1.0}):
                lens = m["camera"].get("lens") or {}
                self.assertEqual((lens.get("k1", 0.0), lens.get("k2", 0.0), lens.get("image_circle_frac", 0.0)), (0.0, 0.0, 0.0))

    def test_the_officer_faces_along_or_across_the_road(self):
        with tempfile.TemporaryDirectory() as td:
            for m in generate(Path(td), num_samples=6, officer_yaw_mode="road"):
                rel = (m["officer"]["world_yaw_deg"] - m["camera"]["yaw"]) % 90.0
                self.assertLessEqual(min(rel, 90.0 - rel), 5.0 + 1e-6)
                cx, cy = m["camera"]["position"][:2]                         # the azimuth is still the camera's, in the officer's frame
                az = np.degrees(np.arctan2(cy, cx)) - m["officer"]["world_yaw_deg"]
                d = (az - m["camera"]["placement"]["azimuth_deg"] + 180.0) % 360.0 - 180.0
                self.assertLess(abs(d), 1e-3)
                self.assertIn(m["camera"]["placement"]["road_quarter"], (0, 1, 2, 3))

    def test_locked_camera_and_background(self):
        with tempfile.TemporaryDirectory() as td:
            ov = {"distance_cm": [700, 700], "height_cm": [120, 120], "azimuth_deg": [15, 15]}
            for m in generate(Path(td), num_samples=3, camera_overrides=ov, background_ids=["fx_dusk"]):
                self.assertEqual(m["background"]["id"], "fx_dusk")
                p = m["camera"]["placement"]
                self.assertEqual((p["distance_cm"], p["height_cm"], p["azimuth_deg"]), (700, 120, 15))

    def test_unknown_locked_background_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "background_ids"):
                GenerationContext(make_config(Path(td), schema_version="1.6", background_ids=["nope"], **FAST))


class OcclusionTests(unittest.TestCase):
    """Keypoints hidden by the officer's own body: from behind the face is hidden, from the front it is not."""
    def view(self, azimuth: float) -> dict:
        with tempfile.TemporaryDirectory() as td:
            ov = {"azimuth_deg": [azimuth, azimuth], "distance_cm": [600, 600], "height_cm": [150, 150], "aim_height_cm": [110, 110]}
            m = generate(Path(td), num_samples=1, camera_overrides=ov, gesture_weights={"VN_NO_SIGNAL": 1.0},
                         lens_distortion=False)[0]
        c = m["officer"]["keypoints_2d"]["coco17"]
        return dict(zip(c["names"], c["occluded"]))

    def test_front_and_back(self):
        front, back = self.view(0.0), self.view(180.0)
        for k in ("nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            self.assertFalse(front[k], k)
        for k in ("nose", "left_eye", "right_eye"):
            self.assertTrue(back[k], k)

    def test_the_far_side_is_hidden_from_the_side(self):
        right_view = self.view(90.0)                                        # camera on the officer's right
        self.assertTrue(right_view["left_eye"])
        self.assertFalse(right_view["right_eye"])

    def test_export_marks_occluded_points_visibility_1(self):
        from tosdg.export import keypoints_of
        meta = {"officer": {"keypoints_2d": {n: {"uv": [[1.0, 2.0]] * k, "in_frame": [True] * k, "occluded": [True] + [False] * (k - 1)}
                                             for n, k in (("coco17", 17), ("hand21_l", 21), ("hand21_r", 21))}}}
        kp = keypoints_of(meta)
        self.assertEqual((kp[0, 2], kp[1, 2], kp[17, 2]), (1, 2, 1))


class BackgroundSetTests(unittest.TestCase):
    def test_set_makes_the_horizon_reliable(self):
        with tempfile.TemporaryDirectory() as td:
            lib_dir = Path(td) / "lib"
            shutil.copytree(fixture_library(), lib_dir)
            rc, out, _ = run_cli("backgrounds", "set", lib_dir, "fx_dusk", "--horizon", "0.5", "--camera-height", "130", "--road-yaw", "15")
            self.assertEqual(rc, 0, out)
            b = BackgroundLibrary(lib_dir).by_id["fx_dusk"]
            self.assertEqual((b["horizon_y_frac"], b["horizon_source"], b["camera_height_cm"], b["road_yaw_deg"]), (0.5, "manual", 130.0, 15.0))
            self.assertTrue(BackgroundLibrary.horizon_reliable(b))


class StudioRandomTests(unittest.TestCase):
    def setUp(self):
        from tosdg.studio.server import StudioApp
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        self.app = StudioApp(td / "ws", library=library_with_people(td))

    def tearDown(self):
        self._td.cleanup()

    def test_config_from_the_panel(self):
        from tosdg.studio.server import ApiError
        cam = {"distance_cm": 900, "height_cm": 120, "azimuth_deg": 0, "focal_length_mm": 4, "aim_height_cm": 110, "roll_deg": 0,
               "sensor_width_mm": 6.17}
        c = self.app.random_config({"num_samples": 5, "seed": 3, "gesture": "VN_STOP_FRONT", "camera_type": "dashcam",
                                    "camera_lock": cam, "background": "fx_dusk", "officer_turn": "studio", "baton": True,
                                    "podium_probability": 0.15, "min_officer_height_px": 200, "renderer": "cpu"})
        self.assertEqual(c["gesture_weights"]["VN_STOP_FRONT"], 1.0)
        self.assertEqual(sum(c["gesture_weights"].values()), 1.0)
        self.assertEqual(c["camera_profile_weights"], {"dashcam": 1.0, "driver_eye": 0.0, "motorbike_rider": 0.0, "roadside": 0.0})
        self.assertEqual(c["camera_overrides"]["distance_cm"], [900.0, 900.0])
        self.assertEqual(c["background_ids"], ["fx_dusk"])
        self.assertNotIn("officer_yaw_mode", c)                          # "as in this view"
        self.assertNotIn("min_officer_height_px", c)                     # a locked camera keeps its distance
        self.assertEqual((c["size_reference"], c["baton"]["enabled"], c["podium"]["probability"]), ("auto", True, 0.15))
        from tosdg.schema import validate_instance
        validate_instance(c, "generation_config")                       # a valid generation config
        with self.assertRaises(ApiError):
            self.app.random_config({"officer_turn": "studio"})            # needs a locked camera
        with self.assertRaises(ApiError):
            self.app.random_config({"gesture": "VN_NOPE"})

    def test_a_batch_runs_in_the_background(self):
        import time
        r = self.app.api_random_generate({"num_samples": 2, "seed": 1, "resolution": [160, 90], "renderer": "cpu",
                                          "camera_type": "roadside", "min_officer_height_px": 40})
        self.assertTrue(r["running"])
        t0 = time.time()
        while self.app.random_job["running"] and time.time() - t0 < 120:
            time.sleep(0.2)
        st = self.app.api_random_status({})
        self.assertIsNone(st["error"])
        self.assertEqual(len(st["samples"]), 2)
        shutil.rmtree(st["out_dir"], ignore_errors=True)
        Path(st["out_dir"] + ".config.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
