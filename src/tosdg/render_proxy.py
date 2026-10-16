"""Proxy render backend (TEST BACKEND - not the production renderer).

Draws the officer as shaded capsules over the composed backplate and produces
an instance mask from exactly the same geometry. It exists so the full
pipeline (sampling -> render -> ground truth -> validation) can be exercised
end-to-end without Unreal, and as a reference for checking the Unreal backend's
projection/bbox. Images are deliberately simple.

Contents:
  constants: SUPERSAMPLE
  render — Returns (rgb uint8 HxWx3, mask bool HxW)
  (+ 1 private helper)
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .skeleton import FKResult, Skeleton

SUPERSAMPLE = 2


def _capsule_polygon(a: np.ndarray, b: np.ndarray, ra: float, rb: float) -> list[tuple[float, float]]:
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return []
    perp = np.array([-d[1], d[0]]) / n
    return [tuple(a + perp * ra), tuple(b + perp * rb), tuple(b - perp * rb), tuple(a - perp * ra)]


def render(skeleton: Skeleton, fk_world: FKResult, camera: Camera, backplate: Image.Image,
           colors: dict[str, list[int]], lighting: dict, extra_capsules: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb uint8 HxWx3, mask bool HxW). `extra_capsules` = [(a, b, radius, colour region)] props in world coordinates (a baton)."""
    W, H, S = camera.width, camera.height, SUPERSAMPLE
    cam_pos = np.asarray(camera.position)
    sun = np.asarray(lighting["sun_direction"])
    sun_rgb = np.asarray(lighting["sun_color_rgb"])
    amb = lighting["ambient_intensity"]

    items = []
    parts = [(fk_world.positions[s["from"]], fk_world.positions[s["to"]], s["radius"], s["region"]) for s in skeleton.segments.values()]
    parts += [(np.asarray(a), np.asarray(b), rad, region) for a, b, rad, region in (extra_capsules or [])]
    for p0, p1, radius, region in parts:
        uv, depth = camera.project(np.stack([p0, p1]))
        if np.any(depth <= 1.0) or np.any(~np.isfinite(uv)):
            continue  # behind/at the camera: subject validation rejects such samples
        r_px = radius * camera.fx / depth
        # shade at the silhouette centre: normal = view direction orthogonalised to the axis
        mid = (p0 + p1) / 2
        axis = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-6)
        view = cam_pos - mid
        nrm = view - axis * (view @ axis)
        nrm = nrm / max(np.linalg.norm(nrm), 1e-6)
        diffuse = max(0.0, float(nrm @ sun)) * lighting["sun_intensity"]
        base = np.asarray(colors[region], float) / 255.0
        col = base * (amb + diffuse * sun_rgb)
        items.append((float(depth.mean()), uv * S, r_px * S, col))

    img = backplate.convert("RGB").resize((W * S, H * S), Image.BILINEAR)
    mask = Image.new("L", (W * S, H * S), 0)
    d_img, d_mask = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    for _, uv, r, col in sorted(items, key=lambda t: -t[0]):  # painter's algorithm, far first
        fill = tuple(int(np.clip(c * 255, 0, 255)) for c in col)
        poly = _capsule_polygon(uv[0], uv[1], r[0], r[1])
        for d, f in ((d_img, fill), (d_mask, 255)):
            if poly:
                d.polygon(poly, fill=f)
            for (x, y), rr in zip(uv, r):
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=f)

    rgb = np.asarray(img.resize((W, H), Image.BOX), dtype=np.float32) / 255.0
    rgb = np.clip(rgb * 2.0 ** lighting["exposure_ev"], 0, 1)
    m = np.asarray(mask.resize((W, H), Image.BOX)) >= 128
    return (rgb * 255 + 0.5).astype(np.uint8), m
