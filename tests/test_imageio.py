"""Image import (PNG / JPG / JPEG / WebP / BMP / GIF / TIFF -> stored as PNG or JPG) and export (rgb as PNG or JPG).

Contents:
  ImportTests (7 tests)
  ExportTests (2 tests)
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from helpers import make_config  # noqa: E402
from tosdg import imageio  # noqa: E402
from tosdg.backgrounds import BackgroundLibrary  # noqa: E402
from tosdg.generate import run  # noqa: E402
from tosdg.scene import GenerationContext  # noqa: E402
from tosdg.studio.workspace import WorkspaceError, prepare_image  # noqa: E402
from tosdg.validate import validate_dataset  # noqa: E402


def encode(fmt: str, mode: str = "RGB", size=(40, 30), **kw) -> bytes:
    b = io.BytesIO()
    color = (200, 30, 30, 255) if mode == "RGBA" else (200, 30, 30)
    Image.new(mode, size, color).save(b, fmt, **kw)
    return b.getvalue()


class ImportTests(unittest.TestCase):
    def test_png_stays_png_jpeg_stays_jpg(self):
        for fmt, ext in (("PNG", ".png"), ("JPEG", ".jpg")):
            data, got, w, h = prepare_image(encode(fmt))
            self.assertEqual((got, w, h), (ext, 40, 30))
            self.assertEqual(Image.open(io.BytesIO(data)).format, fmt)

    def test_webp_bmp_gif_tiff_are_stored_as_jpg(self):
        for fmt in ("WEBP", "BMP", "GIF", "TIFF"):
            data, ext, w, h = prepare_image(encode(fmt))
            self.assertEqual((ext, w, h), (".jpg", 40, 30), fmt)
            self.assertEqual(Image.open(io.BytesIO(data)).format, "JPEG", fmt)

    def test_transparent_webp_is_stored_as_png(self):
        b = io.BytesIO()
        Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(b, "WEBP")
        data, ext, _, _ = prepare_image(b.getvalue())
        self.assertEqual(ext, ".png")
        self.assertEqual(Image.open(io.BytesIO(data)).mode, "RGBA")

    def test_opaque_rgba_webp_is_stored_as_jpg(self):
        _, ext, _, _ = prepare_image(encode("WEBP", "RGBA"))
        self.assertEqual(ext, ".jpg")

    def test_animated_gif_keeps_first_frame(self):
        frames = [Image.new("RGB", (16, 16), c) for c in ((250, 0, 0), (0, 0, 250))]
        b = io.BytesIO()
        frames[0].save(b, "GIF", save_all=True, append_images=frames[1:])
        data, ext, _, _ = prepare_image(b.getvalue())
        px = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).mean(axis=(0, 1))
        self.assertGreater(px[0], px[2] + 100)

    def test_rejects_unsupported_and_garbage(self):
        b = io.BytesIO()
        Image.new("RGB", (8, 8)).save(b, "PPM")
        for bad in (b.getvalue(), b"not an image", b""):
            with self.assertRaises(WorkspaceError):
                prepare_image(bad)

    def test_background_library_converts_non_png_jpeg(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "a.webp").write_bytes(encode("WEBP"))
            (td / "b.jpeg").write_bytes(encode("JPEG"))
            (td / "c.png").write_bytes(encode("PNG"))
            lib = BackgroundLibrary(td / "lib")
            files = {Path(n).stem: lib.add(td / n, ["t"])["file"] for n in ("a.webp", "b.jpeg", "c.png")}
            self.assertEqual({k: Path(v).suffix for k, v in files.items()}, {"a": ".jpg", "b": ".jpg", "c": ".png"})
            for v in files.values():
                Image.open(lib.root / v).verify()


class ExportTests(unittest.TestCase):
    def test_save_rgb_png_and_jpg(self):
        rgb = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            for fmt, pil in (("png", "PNG"), ("jpg", "JPEG")):
                p = Path(td) / f"x.{fmt}"
                imageio.save_rgb(rgb, p, fmt)
                self.assertEqual(Image.open(p).format, pil)
            self.assertTrue(np.array_equal(np.asarray(Image.open(Path(td) / "x.png")), rgb))

    def test_generate_jpg_dataset_validates_and_reproduces(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = make_config(td, num_samples=2, image_format="jpg")
            ctx = GenerationContext(cfg)
            out = td / "out"
            run(ctx, out, log=lambda *_: None)
            self.assertEqual(validate_dataset(out), [])
            meta = json.loads(next(out.glob("*_000000.json")).read_text())
            self.assertEqual((meta["image"]["format"], meta["files"]["rgb"].rsplit(".", 2)[-2:]), ("jpg", ["rgb", "jpg"]))
            self.assertEqual(meta["files"]["mask"].rsplit(".", 1)[-1], "png")
            self.assertEqual(Image.open(out / meta["files"]["rgb"]).format, "JPEG")
            first = (out / meta["files"]["rgb"]).read_bytes()
            run(ctx, td / "out2", log=lambda *_: None)
            self.assertEqual((td / "out2" / meta["files"]["rgb"]).read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
