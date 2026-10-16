"""Synthetic fixture backplates for tests and smoke runs (no external images needed).

These are flat-shaded road scenes with a known horizon row. They exercise the
backplate pipeline; real photo backplates replace them via `tosdg backgrounds import`.

Contents:
  constants: FIXTURES
  make_backplate
  build_fixture_library
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .backgrounds import BackgroundLibrary
from .ground import mask_below_horizon

FIXTURES = [
    # id, horizon_frac, sky top, sky bottom, ground, tags
    ("fx_day_clear", 0.42, (90, 140, 210), (190, 210, 235), (95, 95, 98), ["day", "urban", "ground_level"]),
    ("fx_day_overcast", 0.45, (150, 155, 165), (200, 202, 208), (88, 90, 92), ["day", "overcast", "ground_level"]),
    ("fx_dusk", 0.40, (60, 50, 90), (230, 150, 100), (70, 68, 72), ["dusk", "urban", "ground_level"]),
    ("fx_elevated", 0.12, (120, 160, 210), (180, 200, 225), (100, 100, 104), ["day", "elevated"]),
]


def make_backplate(w: int, h: int, horizon: float, sky0, sky1, ground, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    hy = int(round(horizon * h))
    img = np.zeros((h, w, 3), np.float32)
    t = np.linspace(0, 1, max(hy, 1))[:, None]
    img[:hy] = (np.array(sky0) * (1 - t) + np.array(sky1) * t)[:, None, :]
    img[hy:] = ground
    img += rng.normal(0, 4, img.shape)  # sensor-like noise so images are not flat
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    vx = w // 2  # vanishing point on the horizon
    for frac in (-0.9, -0.3, 0.3, 0.9):  # lane lines converging to the vanishing point
        d.line([(vx, hy), (vx + frac * w, h)], fill=(225, 225, 215), width=max(2, w // 300))
    for i in range(6):  # buildings on the horizon
        x0 = int(rng.uniform(0, w))
        bw, bh = int(rng.uniform(w * 0.04, w * 0.12)), int(rng.uniform(h * 0.05, h * 0.25))
        c = tuple(int(x) for x in rng.uniform(60, 160, 3))
        d.rectangle([x0, hy - bh, x0 + bw, hy], fill=c)
    return im


def build_fixture_library(lib_dir: str | Path, width: int = 1280, height: int = 720) -> BackgroundLibrary:
    lib_dir = Path(lib_dir)
    tmp = lib_dir / "_src"
    tmp.mkdir(parents=True, exist_ok=True)
    lib = BackgroundLibrary(lib_dir)
    for k, (bid, hf, s0, s1, g, tags) in enumerate(FIXTURES):
        if bid in lib.by_id:
            if "ground" not in lib.by_id[bid]:          # libraries built before ground masks (index 1.0) get theirs
                lib.set_ground(bid, mask_below_horizon(lib.by_id[bid]["width"], lib.by_id[bid]["height"], hf), method="horizon")
            continue
        p = tmp / f"{bid}.png"
        make_backplate(width, height, hf, s0, s1, g, seed=k).save(p)
        lib.add(p, tags=tags + ["fixture"], category="fixture", bg_id=bid, horizon_y_frac=hf)
        lib.set_ground(bid, mask_below_horizon(width, height, hf), method="horizon")   # exact: the ground is everything below the horizon
        p.unlink()
    tmp.rmdir()
    lib.save()
    return lib
