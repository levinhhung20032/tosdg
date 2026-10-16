"""Lighting parameters (backend-agnostic description of a simple sun + sky setup).

Contents:
  constants: DEFAULT_LIGHTING
  kelvin_to_rgb — Approximate blackbody colour (Tanner Helland fit),…
  sample_lighting
"""
from __future__ import annotations

import numpy as np

from . import rng as rngmod

DEFAULT_LIGHTING = {
    "sun_elevation_deg": [15, 75], "sun_azimuth_deg": [-180, 180], "sun_intensity": [0.6, 1.1],
    "ambient_intensity": [0.25, 0.5], "color_temperature_k": [4500, 7000], "exposure_ev": [-0.5, 0.5],
}


def kelvin_to_rgb(k: float) -> list[float]:
    """Approximate blackbody colour (Tanner Helland fit), normalised so max channel = 1."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    c = np.clip([r, g, b], 0, 255) / 255.0
    return [float(x) for x in c / c.max()]


def sample_lighting(ranges: dict, rng: np.random.Generator) -> dict:
    r = {**DEFAULT_LIGHTING, **(ranges or {})}
    out = {k: rngmod.uniform(rng, r[k]) for k in sorted(DEFAULT_LIGHTING)}
    el, az = np.radians(out["sun_elevation_deg"]), np.radians(out["sun_azimuth_deg"])
    # direction *towards* the sun, UE world frame
    out["sun_direction"] = [float(np.cos(el) * np.cos(az)), float(np.cos(el) * np.sin(az)), float(np.sin(el))]
    out["sun_color_rgb"] = kelvin_to_rgb(out["color_temperature_k"])
    return out
