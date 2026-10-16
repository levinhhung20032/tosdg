"""Automatic horizon estimate for an imported backplate (heuristic, 2D, roll ignored).

The horizon is taken as the row where the picture changes most from "above" to "below" (sky -> buildings / ground):
the colour of a band of rows above the row is compared with the band below it, over a downscaled copy. A weak
peak means the image has no clear sky/ground split, and then nothing is returned: the user sets it by hand.
The result is only a starting value; the Studio horizon marker and `--horizon` override it.

Contents:
  constants: WORK_HEIGHT, MIN_FRAC, MAX_FRAC, MIN_CONTRAST
  detect_horizon — Horizon row as a fraction of the image height, or None…
"""
from __future__ import annotations

import numpy as np
from PIL import Image

WORK_HEIGHT = 200
MIN_FRAC, MAX_FRAC = 0.1, 0.9          # horizon search range (fraction of image height)
MIN_CONTRAST = 14.0                    # smallest above/below colour difference (0..255) accepted as a real split


def detect_horizon(img: Image.Image) -> float | None:
    """Horizon row as a fraction of the image height in (0, 1), or None if there is no clear sky/ground split."""
    w = max(8, round(img.width * WORK_HEIGHT / img.height)) if img.height > WORK_HEIGHT else img.width
    h = min(img.height, WORK_HEIGHT)
    if h < 20:
        return None
    a = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float64)
    a = a[:, w // 10: w - w // 10]                              # ignore the borders
    rows = a.mean(axis=1)                                        # (h, 3) mean colour of each row
    k = max(3, h // 10)
    lo, hi = int(MIN_FRAC * h), int(MAX_FRAC * h)
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(rows, axis=0)])
    best_y, best_s = None, 0.0
    for y in range(max(lo, k), min(hi, h - k) + 1):
        above = (cs[y] - cs[y - k]) / k
        below = (cs[y + k] - cs[y]) / k
        s = float(np.linalg.norm(below - above)) * (1.0 - 0.3 * abs(y / h - 0.5) / 0.4)    # mild preference for the middle
        if s > best_s:
            best_y, best_s = y, s
    if best_y is None or best_s < MIN_CONTRAST:
        return None
    return round(best_y / h, 4)
