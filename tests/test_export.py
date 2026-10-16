"""Export (COCO keypoints, YOLO-pose, CVAT) and the minimum officer size (min_officer_height_px).

Contents:
  SplitTests (3 tests)
  RleTests (2 tests)
  ExportTests (6 tests)
  MinSizeTests (5 tests)
"""
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import make_config
from test_cli import FAST, run_cli
from tosdg import rng as rngmod
from tosdg.camera import NEAR_SPREAD, _near_enough, sample_camera
from tosdg.export import FLIP_IDX, KEYPOINTS, export, keypoints_of, mask_rle_coco, mask_rle_cvat, split_samples
from tosdg.generate import run
from tosdg.scene import GenerationContext

_DS = None


def dataset() -> Path:
    """One small generated dataset shared by the export tests (proxy backend)."""
    global _DS
    if _DS is None:
        td = Path(tempfile.mkdtemp(prefix="tosdg_export_"))
        run(GenerationContext(make_config(td, num_samples=5, **FAST)), td / "ds", log=lambda *_: None)
        _DS = td / "ds"
    return _DS


def decode(runs, n) -> np.ndarray:
    flat = np.zeros(n, bool)
    pos = 0
    for k, r in enumerate(runs):
        if k % 2:
            flat[pos:pos + r] = True
        pos += r
    assert pos == n
    return flat


class SplitTests(unittest.TestCase):
    def test_split_is_seeded_and_follows_the_ratios(self):
        ids = [f"s{i:03d}" for i in range(100)]
        a = split_samples(ids, (0.7, 0.2, 0.1), 7)
        self.assertEqual(a, split_samples(list(reversed(ids)), (0.7, 0.2, 0.1), 7))     # input order does not matter
        self.assertNotEqual(a, split_samples(ids, (0.7, 0.2, 0.1), 8))
        self.assertEqual([sum(v == s for v in a.values()) for s in ("train", "val", "test")], [70, 20, 10])

    def test_ratios_are_normalised_and_checked(self):
        a = split_samples(["a", "b", "c", "d"], (2, 2, 0), 0)
        self.assertEqual(sorted(a.values()), ["train", "train", "val", "val"])
        for bad in ((0, 0, 0), (1, -1, 1), (1, 1)):
            with self.assertRaises(ValueError):
                split_samples(["a"], bad, 0)

    def test_flip_index_swaps_left_and_right(self):
        self.assertEqual(len(FLIP_IDX), len(KEYPOINTS))
        for i, j in enumerate(FLIP_IDX):
            self.assertEqual(FLIP_IDX[j], i)
            a, b = KEYPOINTS[i], KEYPOINTS[j]
            self.assertEqual(a.replace("left", "X").replace("right", "left").replace("X", "right"), b)


class RleTests(unittest.TestCase):
    def test_coco_rle_is_column_major_starting_with_background(self):
        m = np.zeros((5, 7), bool)
        m[1:4, 2:5] = True
        m[0, 0] = True                                       # starts with foreground: a zero-length background run first
        runs = mask_rle_coco(m)
        self.assertEqual(runs[0], 0)
        self.assertTrue((decode(runs, m.size).reshape(7, 5).T == m).all())

    def test_cvat_rle_is_row_major_inside_the_box(self):
        m = np.zeros((6, 8), bool)
        m[2:5, 3:6] = True
        m[2, 3] = False
        runs = mask_rle_cvat(m, [3, 2, 6, 5])
        self.assertTrue((decode(runs, 9).reshape(3, 3) == m[2:5, 3:6]).all())


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.metas = {m["sample_id"]: m for m in (json.loads(p.read_text(encoding="utf-8"))
                                                   for p in dataset().glob("*.json") if p.name != "manifest.json")}

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def mask(self, sid):
        return np.asarray(Image.open(dataset() / f"{sid}.mask.png")) > 0

    def test_coco(self):
        r = export(dataset(), self.td / "coco", "coco", (0.6, 0.2, 0.2), seed=3)
        self.assertEqual(sum(r["counts"].values()), 5)
        seen = 0
        for f in (self.td / "coco" / "annotations").glob("person_keypoints_*.json"):
            doc = json.loads(f.read_text())
            split = f.stem.rsplit("_", 1)[1]
            self.assertEqual(len(doc["categories"][0]["keypoints"]), 59)
            for im, an in zip(doc["images"], doc["annotations"]):
                sid = Path(im["file_name"]).stem
                m = self.metas[sid]
                self.assertTrue((self.td / "coco" / "images" / split / im["file_name"]).exists())
                h, w = an["segmentation"]["size"]
                self.assertTrue((decode(an["segmentation"]["counts"], h * w).reshape(w, h).T == self.mask(sid)).all())
                self.assertEqual(an["bbox"], [float(v) for v in m["officer"]["bbox"]["xywh"]])
                self.assertEqual(an["gesture"], m["officer"]["gesture"]["id"])
                kp = np.array(an["keypoints"]).reshape(-1, 3)
                self.assertTrue(np.allclose(kp, keypoints_of(m), atol=1e-3))
                seen += 1
        self.assertEqual(seen, 5)

    def test_yolo(self):
        export(dataset(), self.td / "yolo", "yolo", seed=1, yolo_classes="gesture")
        labels = list((self.td / "yolo" / "labels").rglob("*.txt"))
        self.assertEqual(len(labels), 5)
        for f in labels:
            v = f.read_text().split()
            m = self.metas[f.stem]
            W, H = m["image"]["width"], m["image"]["height"]
            self.assertEqual(len(v), 5 + 59 * 3)
            x0, y0, x1, y1 = m["officer"]["bbox"]["xyxy"]
            self.assertAlmostEqual(float(v[1]) * W, (x0 + x1) / 2, places=2)
            self.assertAlmostEqual(float(v[4]) * H, y1 - y0, places=2)
            kp = np.array(v[5:], float).reshape(-1, 3) * [W, H, 1]
            self.assertTrue(np.allclose(kp, keypoints_of(m), atol=0.01))
            self.assertTrue((f.parent.parent.parent / "images" / f.parent.name / (f.stem + ".png")).exists())
        yaml = (self.td / "yolo" / "data.yaml").read_text()
        self.assertIn("kpt_shape: [59, 3]", yaml)
        self.assertIn("VN_STOP_LEFT", yaml)

    def test_cvat(self):
        export(dataset(), self.td / "cvat", "cvat")
        root = ET.parse(self.td / "cvat" / "annotations.xml").getroot()
        images = root.findall("image")
        self.assertEqual(len(images), 5)
        for im in images:
            sid = Path(im.get("name")).stem
            mk = im.find("mask")
            x0, y0, w, h = (int(mk.get(k)) for k in ("left", "top", "width", "height"))
            runs = [int(x) for x in mk.get("rle").split(",")]
            self.assertTrue((decode(runs, w * h).reshape(h, w) == self.mask(sid)[y0:y0 + h, x0:x0 + w]).all())
            self.assertEqual(im.find("box/attribute").text, self.metas[sid]["officer"]["gesture"]["id"])
            sk = im.findall("skeleton")
            self.assertEqual([len(s.findall("points")) for s in sk], [17, 21, 21])
        labels = json.loads((self.td / "cvat" / "cvat_labels.json").read_text())
        self.assertEqual([l["name"] for l in labels], ["traffic_officer", "officer_body", "left_hand", "right_hand"])

    def test_labels_csv(self):
        import csv
        export(dataset(), self.td / "c", "coco", seed=3)
        with (self.td / "c" / "labels.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        for r in rows:
            m = self.metas[Path(r["image"]).stem]
            self.assertTrue((self.td / "c" / r["image"]).exists())
            self.assertEqual(r["gesture_id"], m["officer"]["gesture"]["id"])
            self.assertEqual(r["signalling"], "yes" if m["officer"]["gesture"]["signalling"] else "no")
            self.assertTrue(r["hieu_lenh"])
            self.assertEqual(r["observer_side"], m["traffic_control"]["approach_side"] or "")

    def test_refuses_a_non_empty_output_or_an_invalid_dataset(self):
        (self.td / "busy").mkdir()
        (self.td / "busy" / "x.txt").write_text("x")
        with self.assertRaisesRegex(ValueError, "not empty"):
            export(dataset(), self.td / "busy", "coco")
        bad = self.td / "bad"
        shutil.copytree(dataset(), bad)
        next(bad.glob("*.mask.png")).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "does not pass 'tosdg validate'"):
            export(bad, self.td / "out", "coco")

    def test_cli(self):
        rc, out, _ = run_cli("export", dataset(), "--format", "yolo", "--out", self.td / "y", "--split", "1,0,0")
        self.assertEqual(rc, 0, out)
        self.assertIn("train 5, val 0, test 0", out)
        rc, _, err = run_cli("export", dataset(), "--format", "coco", "--out", self.td / "c", "--split", "1,x,0")
        self.assertEqual(rc, 2)


class MinSizeTests(unittest.TestCase):
    PROFILE = {"id": "t", "distance_cm": [800, 2500], "height_cm": [120, 120], "azimuth_deg": [0, 0],
               "focal_length_mm": [4.0, 4.0], "sensor_width_mm": 6.0}

    def test_off_means_the_same_draws(self):
        a, pa = sample_camera(self.PROFILE, rngmod.stream(5, "camera"), [0, 0, 0], 0, 640, 360)
        b, pb = sample_camera(self.PROFILE, rngmod.stream(5, "camera"), [0, 0, 0], 0, 640, 360, None)
        self.assertEqual((pa, a.position), (pb, b.position))

    def test_distance_is_capped_and_keeps_its_place_in_the_range(self):
        fx = 4.0 * 640 / 6.0
        cap = fx * 180 / 100                                          # 100 px tall at 768 cm
        self.assertEqual(_near_enough(1000, [800, 2500], fx, 180, 20), 1000)    # 20 px even at 25 m: unchanged
        self.assertAlmostEqual(_near_enough(800, [800, 2500], fx, 180, 100), NEAR_SPREAD * cap)  # cap < profile minimum: closer
        self.assertAlmostEqual(_near_enough(2500, [800, 2500], fx, 180, 100), cap)
        self.assertAlmostEqual(_near_enough(1650, [100, 3200], fx, 180, 100), 100 + 0.5 * (cap - 100))

    def test_generated_officers_are_at_least_that_tall(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            from test_people import library_with_people
            cfg = make_config(td, num_samples=6, schema_version="1.6", size_reference="auto", min_officer_height_px=90,
                              background_library=str(library_with_people(td)), **FAST)
            man = run(GenerationContext(cfg), td / "ds", log=lambda *_: None)
            self.assertEqual(len(man["samples"]), 6)
            for s in man["samples"]:
                b = json.loads((td / "ds" / s["metadata"]).read_text())["officer"]["bbox"]["xyxy"]
                self.assertGreaterEqual(b[3] - b[1], 90)

    def test_needs_sizing_from_people(self):
        """A bigger officer alone in front of an unchanged photo looks too big against it (D40): refused."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "size_reference"):
                GenerationContext(make_config(Path(td), min_officer_height_px=90, **FAST))

    def test_taller_than_the_image_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "taller than the image"):
                GenerationContext(make_config(Path(td), size_reference="auto", min_officer_height_px=500, **FAST))     # image is 180 px tall


if __name__ == "__main__":
    unittest.main()
