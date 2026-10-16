"""Image import and export formats.

Import accepts PNG, JPEG (.jpg / .jpeg), WebP, BMP, GIF and TIFF. Storage is normalised: PNG stays PNG, JPEG stays JPEG (.jpg),
everything else (WebP, BMP, GIF, TIFF) is converted to JPG, unless it has real transparency, which JPG cannot hold: that one is
stored as PNG. Camera metadata is dropped and the EXIF orientation applied. Animated files keep their first frame.

Export (generated datasets): the rgb image is written as PNG (lossless, default) or JPG; masks are always PNG.

Contents:
  constants: ACCEPTED_FORMATS, EXPORT_FORMATS, JPEG_QUALITY
  accepted_names — The accepted formats as text for error messages
  has_alpha — True if the image has transparent pixels
  normalise — Open image bytes -> (stored bytes, extension, width, height)
  save_rgb — Write an (H, W, 3) uint8 array as PNG or JPG
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ACCEPTED_FORMATS = {"PNG", "JPEG", "MPO", "WEBP", "BMP", "GIF", "TIFF"}       # PIL format names (MPO = a JPEG from some cameras)
EXPORT_FORMATS = {"png": "PNG", "jpg": "JPEG"}
JPEG_QUALITY = 95


def accepted_names() -> str:
    return "PNG, JPG/JPEG, WebP, BMP, GIF or TIFF"


def has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        return img.getchannel("A").getextrema()[0] < 255
    return img.mode == "P" and "transparency" in img.info


def normalise(data: bytes, quality: int = 93) -> tuple[bytes, str, int, int]:
    """Returns (stored bytes, '.png' | '.jpg', width, height). Raises ValueError for unreadable or unsupported data."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in ACCEPTED_FORMATS:
                raise ValueError(f"unsupported image format {fmt}; use {accepted_names()}")
            im.seek(0)                                                    # animated GIF / WebP / multi-page TIFF: first frame
            im.load()
            img = ImageOps.exif_transpose(im).copy()
    except ValueError:
        raise
    except Exception as e:                                                # PIL raises many types for bad input
        raise ValueError(f"not a readable image: {e}") from None
    if fmt == "PNG" or (fmt not in ("JPEG", "MPO") and has_alpha(img)):
        out_fmt, ext = "PNG", ".png"
    else:
        out_fmt, ext = "JPEG", ".jpg"
    if out_fmt == "JPEG":
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        img = img.convert("RGBA" if has_alpha(img) else "RGB")
    buf = io.BytesIO()
    img.save(buf, out_fmt, **({} if out_fmt == "PNG" else {"quality": quality}))
    return buf.getvalue(), ext, img.width, img.height


def save_rgb(rgb: np.ndarray, path: str | Path, image_format: str = "png") -> None:
    fmt = EXPORT_FORMATS[image_format]
    Image.fromarray(rgb).save(path, fmt, **({"optimize": False} if fmt == "PNG" else {"quality": JPEG_QUALITY, "subsampling": 0}))
