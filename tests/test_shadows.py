"""The officer's cast shadow (shadows.py): geometry, the sun read from a photo, generation, library and Studio."""
import base64
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from helpers import ROOT, fixture_library, make_config

from tosdg import shadows
from tosdg.backgrounds import BackgroundLibrary
from tosdg.camera import Camera
from tosdg.generate import run
from tosdg.scene import GenerationContext
from tosdg.validate import validate_dataset


def street_camera(width=640, height=360, yaw=180.0, pitch=-8.0):
    """A camera 800 cm in front of the origin (on +x), 150 cm high, looking back at it."""
    return Camera("test", (800.0, 0.0, 150.0), pitch, yaw, 0.0, 4.0, 6.17, width, height)


def pole(height=200.0, half=6.0, n=12):
    """A thin vertical prism on the origin: (vertices, triangles)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([half * np.cos(a), half * np.sin(a)], axis=1)
    pos = np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, height)]])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[i, j, n + j], [i, n + j, n + i]]
    tris += [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    tris += [[0, i + 1, i] for i in range(1, n - 1)]
    return pos, np.array(tris, np.int64)


class SunMath(unittest.TestCase):
    def test_vector_and_sun_are_inverse(self):
        for yaw in (-170.0, 0.0, 33.0):
            for d, r in ((0.0, 1.0), (90.0, 0.5), (-135.0, 2.0)):
                az, el = shadows.sun_from_vector(yaw, d, r)
                d2, r2 = shadows.vector_of_sun(yaw, az, el)
                self.assertAlmostEqual((d2 - d + 180) % 360 - 180, 0.0, places=6)
                self.assertAlmostEqual(r2, r, places=6)

    def test_direction_is_relative_to_the_view(self):
        az, _ = shadows.sun_from_vector(30.0, 0.0, 1.0)        # shadow away from the camera = the sun behind it
        self.assertAlmostEqual(az, -150.0)
        self.assertAlmostEqual(shadows.sun_from_vector(0.0, 0.0, 1.0)[1], 45.0)          # as long as tall: 45 degrees

    def test_sun_from_photo_recovers_the_sun_through_the_camera(self):
        cam = street_camera(1280, 720)
        az, el = 50.0, 35.0
        d = np.array([np.cos(np.radians(az + 180)), np.sin(np.radians(az + 180))])
        vecs = []
        for g in ((0.0, 0.0), (300.0, 250.0), (-200.0, -300.0)):
            G = np.array([g[0], g[1], 0.0])
            tip = G + np.r_[d * 165.0 / np.tan(np.radians(el)), 0.0]
            uv, _ = cam.project_pinhole(np.array([G, tip]))
            vecs.append({"kind": "standing", "feet": uv[0].tolist(), "tip": uv[1].tolist(), "height_cm": 165.0, "score": 1.0})
        est = shadows.sun_from_photo({"status": "found", "found": vecs}, cam, {"scale": 1.0, "offset_px": [0, 0]})
        self.assertAlmostEqual(est["azimuth_deg"], az, delta=0.5)
        self.assertAlmostEqual(est["elevation_deg"], el, delta=0.5)
        self.assertEqual((est["vectors_used"], est["source"]), (3, "found"))
        # the photo is scaled / cropped into the canvas: the same vectors in photo pixels give the same sun
        comp = {"scale": 2.0, "offset_px": [-100, -40]}
        photo = [{**v, "feet": [(v["feet"][0] + 100) / 2, (v["feet"][1] + 40) / 2], "tip": [(v["tip"][0] + 100) / 2, (v["tip"][1] + 40) / 2]} for v in vecs]
        est2 = shadows.sun_from_photo({"status": "found", "found": photo}, cam, comp)
        self.assertAlmostEqual(est2["azimuth_deg"], az, delta=0.5)
        # marked by hand wins; nothing usable -> None
        manual = [{**vecs[0], "feet": vecs[0]["tip"], "tip": vecs[0]["feet"]}]           # the opposite way
        est3 = shadows.sun_from_photo({"status": "found", "found": vecs, "manual": manual}, cam, {"scale": 1.0, "offset_px": [0, 0]})
        self.assertEqual(est3["source"], "manual")
        self.assertAlmostEqual((est3["azimuth_deg"] - az - 180 + 180) % 360 - 180, 0.0, delta=1.0)
        self.assertIsNone(shadows.sun_from_photo({"status": "none", "found": []}, cam, comp))
        self.assertIsNone(shadows.sun_from_photo(None, cam, comp))


class CastShadow(unittest.TestCase):
    def test_a_pole_casts_a_shadow_as_long_as_the_sun_says(self):
        cam = street_camera()
        pos, tris = pole(200.0)
        az, el = 90.0, 45.0                               # sun on +y, 45 degrees: the shadow is 200 cm long towards -y
        cov = shadows.cast(pos, tris, shadows.sun_direction(az, el), cam, softness=0.0, contact=0.0)
        uv, _ = cam.project_pinhole(np.array([[0, -100.0, 0], [0, -180.0, 0], [0, -240.0, 0], [0, 100.0, 0]]))
        at = lambda p: cov[int(p[1]), int(p[0])]
        self.assertGreater(at(uv[0]), 0.5)                # along the shadow
        self.assertGreater(at(uv[1]), 0.5)
        self.assertLess(at(uv[2]), 0.1)                   # past its tip (200 cm)
        self.assertLess(at(uv[3]), 0.05)                  # on the sunny side
        low = shadows.cast(pos, tris, shadows.sun_direction(az, 25.0), cam, softness=0.0, contact=0.0)
        self.assertGreater(low[int(uv[2][1]), int(uv[2][0])], 0.5)   # a lower sun: longer

    def test_contact_only_without_sun_and_apply_darkens_only_the_shadow(self):
        cam = street_camera()
        pos, tris = pole(200.0, half=15.0)
        cov = shadows.cast(pos, tris, shadows.sun_direction(90.0, 45.0), cam, contact=0.6, sun_on=False)
        uv, _ = cam.project_pinhole(np.array([[0, 0, 0], [0, -150.0, 0]]))
        self.assertGreater(cov[int(uv[0][1]), int(uv[0][0])], 0.3)        # under the pole
        self.assertLess(cov[int(uv[1][1]), int(uv[1][0])], 0.02)          # no cast shadow
        self.assertLessEqual(cov.max(), 0.6 + 1e-6)
        plate = Image.new("RGB", (cam.width, cam.height), (200, 200, 200))
        out = np.asarray(shadows.apply(plate, cov, 0.5), float)
        self.assertEqual(out[0, 0].tolist(), [200.0, 200.0, 200.0])
        self.assertLess(out[int(uv[0][1]), int(uv[0][0]), 0], 200.0)


def synthetic_street(sunny: bool, direction_deg=60.0, elevation=35.0):
    """A 1280x720 street photo seen by the finder's nominal camera (focal = width, 150 cm high, horizon row 300) with five
    people standing on it; sunny: a blue sky and each person's shadow strip on the road for the given sun."""
    W, H, v0, hc, f, cx = 1280, 720, 300.0, 150.0, 1280.0, 640.0
    im = Image.new("RGB", (W, H), (110, 160, 230) if sunny else (205, 205, 210))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(v0), W, H], fill=(150, 150, 150))
    refs = []
    a = np.radians(direction_deg)
    D, N = np.array([np.sin(a), np.cos(a)]), np.array([np.cos(a), -np.sin(a)])
    proj = lambda P: (cx + f * P[0] / P[1], v0 + f * hc / P[1])
    for x, z in ((-300, 900), (250, 1100), (-50, 1500), (400, 700), (-450, 1300)):
        G = np.array([x, z], float)
        fu, fv = proj(G)
        hpx = f * 165.0 / z
        refs.append({"kind": "standing", "feet": [fu, fv], "bbox": [fu - 0.15 * hpx, fv - hpx, fu + 0.15 * hpx, fv], "h": hpx, "H": 165.0})
        if sunny:
            L = 165.0 / np.tan(np.radians(elevation))
            quad = [proj(G + D * t + N * s) for t, s in ((0, -22), (L, -22), (L, 22), (0, 22))]
            d.polygon(quad, fill=(70, 70, 75))
    return im, refs, v0, hc


class FindPhotoShadows(unittest.TestCase):
    def test_a_sunny_photo_gives_the_direction_of_its_shadows(self):
        im, refs, v0, hc = synthetic_street(True, 60.0, 35.0)
        res = shadows.find_photo_shadows(im, refs, None, v0, hc)
        self.assertEqual((res["status"], res["weather"]), ("found", "sunny"), res["reason"])
        self.assertGreaterEqual(len(res["found"]), 3)
        self.assertLess(abs((res["sun_nominal"]["direction_deg"] - 60.0 + 180) % 360 - 180), 10.0)
        self.assertLess(abs(res["sun_nominal"]["elevation_deg"] - 35.0), 10.0)

    def test_overcast_dusk_night_and_no_people(self):
        im, refs, v0, hc = synthetic_street(False)
        res = shadows.find_photo_shadows(im, refs, None, v0, hc)
        self.assertEqual((res["status"], res["weather"]), ("none", "overcast"))
        sunny, refs2, *_ = synthetic_street(True)
        self.assertEqual(shadows.find_photo_shadows(sunny, refs2, None, v0, hc, tags=["night"])["weather"], "night")
        self.assertEqual(shadows.find_photo_shadows(sunny, refs2, None, v0, hc, tags=["dusk"])["weather"], "dusk")
        none = shadows.find_photo_shadows(sunny, [], None, v0, hc)
        self.assertEqual((none["status"], none["weather"]), ("none", "sunny"))      # sunny, no direction: config unknown_sun

    def test_a_uniformly_dark_road_is_not_a_shadow(self):
        im, refs, v0, hc = synthetic_street(True)
        d = ImageDraw.Draw(im)
        d.rectangle([0, int(v0), 1280, 720], fill=(70, 70, 75))                   # no strips: all dark
        self.assertEqual(shadows.find_photo_shadows(im, refs, None, v0, hc)["status"], "none")


class Library(unittest.TestCase):
    def test_mark_detect_and_clear(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(fixture_library(), Path(td) / "lib")
            lib = BackgroundLibrary(Path(td) / "lib")
            bid = sorted(lib.by_id)[0]
            e = lib.set_shadows(bid)
            self.assertIn(e["shadows"]["status"], ("found", "none"))
            lib.mark_shadow(bid, (10, 200), (60, 210), 165.0)
            lib.save()
            lib2 = BackgroundLibrary(Path(td) / "lib")
            self.assertEqual(len(lib2.by_id[bid]["shadows"]["manual"]), 1)
            lib2.set_shadows(bid)                                       # read again: the marks stay
            self.assertEqual(len(lib2.by_id[bid]["shadows"]["manual"]), 1)
            with self.assertRaises(ValueError):
                lib2.mark_shadow(bid, (-5, 10), (20, 30))
            lib2.clear_shadows(bid, manual_only=True)
            self.assertNotIn("manual", lib2.by_id[bid]["shadows"])
            lib2.clear_shadows(bid)
            self.assertNotIn("shadows", lib2.by_id[bid])
            lib2.save()


class Generation(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.lib = self.td / "lib"
        shutil.copytree(fixture_library(), self.lib)

    def tearDown(self):
        shutil.rmtree(self.td)

    def cfg(self, **shadow):
        over = {"num_samples": 2, "background_library": str(self.lib)}
        if shadow:
            over["shadow"] = shadow
        return make_config(self.td, **over)

    def test_shadow_is_drawn_recorded_and_reproduced(self):
        plain = run(GenerationContext(self.cfg()), self.td / "plain", log=lambda *_: None)
        ctx = GenerationContext(self.cfg(enabled=True, source="random", strength=[0.7, 0.7], softness=[0.1, 0.1]))
        out = self.td / "sh"
        man = run(ctx, out, log=lambda *_: None)
        self.assertEqual(validate_dataset(out), [])
        for s, p in zip(man["samples"], plain["samples"]):
            meta = json.loads((out / s["metadata"]).read_text())
            sh = meta["lighting"]["shadow"]
            self.assertEqual((sh["enabled"], sh["sun"], sh["source"], sh["strength"]), (True, True, "random", 0.7))
            self.assertAlmostEqual(meta["lighting"]["sun_azimuth_deg"], sh["sun_azimuth_deg"], places=3)   # the officer is lit by that sun
            pmeta = json.loads((self.td / "plain" / p["metadata"]).read_text())
            self.assertNotIn("shadow", pmeta["lighting"])
            self.assertEqual(meta["officer"]["gesture"], pmeta["officer"]["gesture"])       # a new stream: the other draws are the same
            self.assertEqual(meta["officer"]["bbox"]["xyxy"], pmeta["officer"]["bbox"]["xyxy"])  # the shadow is not the officer
        again = run(GenerationContext(self.cfg(enabled=True, source="random", strength=[0.7, 0.7], softness=[0.1, 0.1])), self.td / "sh2",
                    log=lambda *_: None)
        self.assertEqual([s["sha256"] for s in again["samples"]], [s["sha256"] for s in man["samples"]])

    def test_photo_source_reads_the_library_and_follows_the_weather(self):
        ctx = GenerationContext(self.cfg(enabled=True, source="photo", no_sun="contact", unknown_sun="off"))
        self.assertTrue(all("shadows" in b for b in BackgroundLibrary(self.lib).index["backgrounds"]))     # read and stored once
        man = run(ctx, self.td / "ph", log=lambda *_: None)
        for s in man["samples"]:
            sh = json.loads((self.td / "ph" / s["metadata"]).read_text())["lighting"]["shadow"]
            self.assertIn(sh["source"], ("contact", "off", "photo-found", "photo-manual"))
            if sh["source"] == "contact":
                self.assertFalse(sh["sun"])

    def test_config_errors(self):
        with self.assertRaisesRegex(ValueError, "mesh"):
            GenerationContext(make_config(self.td, backend="proxy", background_library=str(self.lib), shadow={"enabled": True}))
        with self.assertRaisesRegex(ValueError, "min <= max"):
            GenerationContext(self.cfg(enabled=True, strength=[0.8, 0.2]))
        with self.assertRaises(Exception):
            GenerationContext(self.cfg(enabled=True, source="sky"))       # schema


class Studio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tosdg.studio.server import StudioApp
        cls._td = tempfile.TemporaryDirectory()
        lib = Path(cls._td.name) / "lib"
        shutil.copytree(fixture_library(), lib)
        cls.app = StudioApp(Path(cls._td.name) / "ws", library=lib)
        cls.bg = sorted(BackgroundLibrary(lib).by_id)[0]

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def base(self, **shadow):
        return {"pose": self.app.default_pose(), "camera": {"width": 320, "height": 180, "distance_cm": 380}, "bg_id": self.bg,
                "shadow": {"on": True, "mode": "custom", "direction_deg": 90.0, "length_ratio": 1.0, **shadow}}

    def test_layer_render_and_off(self):
        r = self.app.api_shadow(self.base())
        self.assertEqual((r["shadow"]["source"], r["shadow"]["direction_deg"], r["shadow"]["length_ratio"]), ("custom", 90.0, 1.0))
        a = np.asarray(Image.open(io.BytesIO(base64.b64decode(r["image"].split(",", 1)[1]))))[..., 3]
        self.assertGreater(a.max(), 60)
        self.assertGreater(r["arrow"][1][0], r["arrow"][0][0])          # 90 degrees: to the right in the picture
        off = self.app.api_shadow({**self.base(), "shadow": {"on": False}})
        self.assertIsNone(off["image"])
        img = lambda d: np.asarray(Image.open(io.BytesIO(base64.b64decode(d["image"].split(",", 1)[1]))).convert("L"), float)
        with_sh, without = self.app.api_render(self.base()), self.app.api_render({**self.base(), "shadow": {"on": False}})
        self.assertEqual(with_sh["bbox"]["xyxy"], without["bbox"]["xyxy"])
        self.assertLess(img(with_sh).sum(), img(without).sum())          # the photo is darker by the shadow
        with self.assertRaises(Exception):
            self.app.api_shadow(self.base(strength=3.0))

    def test_mark_maps_the_view_back_to_the_photo(self):
        from tosdg.studio import core
        body = self.base()
        officer, cam_params = self.app._scene_args(body)
        cam = core.make_camera(cam_params, officer)
        bg = self.app.ws.get_background(self.bg)
        _, comp = self.app._plate_geometry(bg, cam)
        s, (ox, oy) = comp["scale"], comp["offset_px"]
        feet, tip = (200.0, 230.0), (260.0, 250.0)                         # photo pixels, on the road (below the horizon)
        view = lambda p: [p[0] * s + ox, p[1] * s + oy]
        r = self.app.api_shadow_mark({**body, "feet": view(feet), "tip": view(tip)})
        self.assertAlmostEqual(r["photo_feet"][0], feet[0], delta=0.2)
        self.assertAlmostEqual(r["photo_tip"][1], tip[1], delta=0.2)
        self.assertEqual(len(r["background"]["shadows"]["manual"]), 1)
        sh = self.app.api_shadow({**body, "shadow": {"on": True, "mode": "photo"}})["shadow"]
        self.assertEqual(sh["source"], "photo-manual")
        self.assertNotIn("manual", self.app.api_shadow_clear({"bg_id": self.bg})["background"]["shadows"])

    def test_shadow_status_lists_the_photos_without_a_direction(self):
        lib = BackgroundLibrary(self.app.ws.lib_dir)
        for i in lib.by_id:                                          # forget what was read: the status reads them again
            lib.clear_shadows(i)
        lib.save()
        r = self.app.api_shadow_status({"background": "random"})
        ids = {p["id"] for p in r["photos"]}
        self.assertEqual(ids, {b["id"] for b in self.app.ws.list_backgrounds()})
        self.assertTrue(all("shadows" in self.app.ws.get_background(i) for i in ids))   # read and stored
        self.assertIn(self.bg, r["missing"])                          # fixture photos: no people, no direction
        body = self.base()
        officer, cam_params = self.app._scene_args(body)
        from tosdg.studio import core
        cam = core.make_camera(cam_params, officer)
        _, comp = self.app._plate_geometry(self.app.ws.get_background(self.bg), cam)
        s, (ox, oy) = comp["scale"], comp["offset_px"]
        self.app.api_shadow_mark({**body, "feet": [200 * s + ox, 230 * s + oy], "tip": [260 * s + ox, 250 * s + oy]})
        one = self.app.api_shadow_status({"background": self.bg})
        self.assertEqual((one["photos"][0]["direction"], one["missing"]), ("manual", []))
        self.app.api_shadow_clear({"bg_id": self.bg})
        with self.assertRaises(Exception):
            self.app.api_shadow_status({"background": "no-such-photo"})

    def test_random_config_shadow(self):
        cfg = self.app.random_config({"shadow": "photo", "officer_turn": "road"})
        self.assertEqual((cfg["shadow"], cfg["schema_version"]), ({"enabled": True, "source": "photo"}, "1.7"))
        self.assertNotIn("shadow", self.app.random_config({"officer_turn": "road"}))
        marks = self.app.random_config({"shadow": "marks", "officer_turn": "road"})["shadow"]
        self.assertEqual(marks, {"enabled": True, "source": "photo", "unknown_sun": "random", "no_sun": "random"})
        self.assertEqual(self.app.random_config({"shadow": "random", "officer_turn": "road"})["shadow"]["source"], "random")
        with self.assertRaises(Exception):
            self.app.random_config({"shadow": "sometimes", "officer_turn": "road"})


if __name__ == "__main__":
    unittest.main()
