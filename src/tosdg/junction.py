"""Where the officer stands and which way they face, from the road layout of the backplate (officer_yaw_mode "junction").

A traffic officer directs the traffic of the roads that meet where they stand, and a gesture means something for the
traffic on given sides of the officer (gestures.json "instructions": front / back / left / right). So:
  - at a junction the officer stands in its middle (a small offset), on a T / cross / Y junction or at a roundabout;
  - the facing follows the junction type (user rules, D43):
      cross (ngã tư)      face one of the roads;
      T (ngã ba chữ T)     the view covers all three roads: face the stem (the bar roads are then left and right);
      Y (ngã ba chữ Y)     face one road, or so that at least two roads are in view;
      roundabout           face one of the roads; the officer may stand on the podium and hold the baton (randomised);
      straight road        along or across the road (no junction in the photo);
  - among the facings the type allows, those where the sides the gesture talks to (instruction != NONE) are roads are
    kept (for a gesture to three or four sides, the officer's back may be a wall: back is not required); then those that
    show the officer from the side the camera profile asks for (azimuth range), if any.

Layout of a photo, set by hand (`tosdg backgrounds junction`), library 1.2 key "junction":
    {"type": "cross" | "T" | "Y" | "roundabout", "center": [u, v], "arms_deg": [a, ...]}
center = the ground point where the officer stands (photo pixels); arms_deg = the direction of each road leaving the
junction, in degrees from the photo's view direction (0 = away from the camera, 90 = to the right, 180 = towards the
camera, -90 = to the left; as road_yaw_deg). Directions, not ground points: measuring them on the photo's ground plane
needs the photo's focal length (unknown) and is very sensitive to the horizon for far points (D43).

Contents:
  constants: TYPES, HALF_FOV_DEG, SIDE_DEG, JITTER_CM, ROUNDABOUT_DEFAULTS
  validate — Check a junction annotation against the photo size
  ground_point — Ground point (z = 0) seen at a pixel of a pinhole camera (the junction's middle)
  required_sides — Sides of the officer a gesture talks to that must be roads
  sides_of — The officer's sides that have a road, for a facing
  candidate_yaws — Facings a junction type allows
  choose_yaw — Facing for a gesture among the allowed ones
"""
from __future__ import annotations

import numpy as np

TYPES = ("cross", "T", "Y", "roundabout")
HALF_FOV_DEG = 110.0          # an officer's field of view (eyes + a little head turn): +-110 degrees
SIDE_DEG = {"front": 0.0, "right": 90.0, "back": 180.0, "left": -90.0}   # azimuth in the officer's frame (+ = right)
JITTER_CM = 100.0             # the officer stands within this distance of the junction's middle
ROUNDABOUT_DEFAULTS = {"podium_probability": 0.5, "baton_probability": 0.5}


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def validate(j: dict, width: int, height: int) -> None:
    """Raises ValueError when a junction annotation is not usable for a photo of that size."""
    if j.get("type") not in TYPES:
        raise ValueError(f"junction type must be one of {list(TYPES)}, got {j.get('type')!r}")
    p = j.get("center")
    if not (isinstance(p, (list, tuple)) and len(p) == 2 and 0 <= p[0] <= width and 0 <= p[1] <= height):
        raise ValueError(f"junction center {p} is not inside the {width}x{height} photo")
    arms = list(j.get("arms_deg") or [])
    if not all(isinstance(a, (int, float)) and -180 <= a <= 180 for a in arms):
        raise ValueError(f"junction road directions must be degrees in [-180, 180], got {arms}")
    need = {"cross": 4, "T": 3, "Y": 3, "roundabout": 3}[j["type"]]
    if len(arms) < need:
        raise ValueError(f"a {j['type']} junction needs at least {need} roads, got {len(arms)}")


def ground_point(cam, uv) -> np.ndarray | None:
    """World point on the ground (z = 0) seen at pixel uv of a pinhole camera, or None at / above the horizon."""
    u, v = float(uv[0]), float(uv[1])
    d = cam.R @ np.array([1.0, (u - cam.cx) / cam.fx, -(v - cam.cy) / cam.fy])
    z0 = float(cam.position[2])
    if d[2] >= -1e-9 or z0 <= 0:
        return None
    return np.asarray(cam.position, float) + d * (z0 / -d[2])


def required_sides(gesture: dict) -> set[str]:
    """Sides the gesture talks to (instruction other than NONE); for three or four sides the back is left out (an officer
    on a T junction facing the stem has a wall behind them and still stops / lets go the others)."""
    s = {k for k, v in gesture.get("instructions", {}).items() if v != "NONE"}
    return s - {"back"} if len(s) >= 3 else s


def sides_of(arms_deg, yaw: float) -> set[str]:
    """The officer's sides (front / right / back / left, 90-degree sectors) that have a road, facing world yaw `yaw`."""
    out = set()
    for a in _wrap(np.asarray(arms_deg, float) - yaw):
        out.add(min(SIDE_DEG, key=lambda s: abs(_wrap(a - SIDE_DEG[s]))))
    return out


def _in_view(arms_deg, yaw: float) -> int:
    return int((np.abs(_wrap(np.asarray(arms_deg, float) - yaw)) <= HALF_FOV_DEG).sum())


def candidate_yaws(kind: str, arms_deg) -> list[float]:
    """World yaws (degrees) the junction type allows: see the module doc. `kind` "straight": arms are the road both ways."""
    arms = [float(a) for a in arms_deg]
    if kind in ("cross", "roundabout"):
        return arms
    if kind == "straight":
        return [float(_wrap(arms[0] + 90.0 * q)) for q in range(4)]
    if kind == "T":
        # the stem: the road with no road opposite it; facing it, the bar's two roads are left and right, all three in view
        # (with marked points a few degrees off, the bar may measure a little past +-90 degrees: still the stem)
        opp = [min(abs(float(_wrap(a + 180.0 - b))) for k, b in enumerate(arms) if k != i) for i, a in enumerate(arms)]
        return [arms[int(np.argmax(opp))]]
    if kind == "Y":
        out = list(arms)
        s = sorted(arms)
        for a, b in zip(s, s[1:] + [s[0] + 360.0]):                # bisector of each pair of neighbouring roads
            if b - a <= 2 * HALF_FOV_DEG:
                y = float(_wrap((a + b) / 2.0))
                if _in_view(arms, y) >= 2:
                    out.append(y)
        return out
    raise ValueError(f"unknown junction type {kind!r}")


def choose_yaw(kind: str, arms_deg, gesture: dict, camera_xy, officer_xy, azimuth_range, rng: np.random.Generator,
               jitter: tuple[float, float]) -> tuple[float, dict]:
    """(world yaw, report): a facing the junction allows, where the most of the gesture's required sides are roads, and
    (when one is) where the camera is in the profile's azimuth range; then a uniform jitter."""
    cands = candidate_yaws(kind, arms_deg)
    req = required_sides(gesture)
    score = [len(req & sides_of(arms_deg, y)) for y in cands]
    best = [y for y, s in zip(cands, score) if s == max(score)]
    cam_dir = float(np.degrees(np.arctan2(camera_xy[1] - officer_xy[1], camera_xy[0] - officer_xy[0])))
    lo, hi = (float(x) for x in azimuth_range)
    seen_ok = [y for y in best if lo <= float(_wrap(cam_dir - y)) <= hi]
    pool = seen_ok or best
    y0 = pool[int(rng.integers(len(pool)))]
    yaw = float(_wrap(y0 + rng.uniform(*jitter)))
    covered = sides_of(arms_deg, y0)
    return yaw, {"type": kind, "arms_azimuth_deg": [round(float(a), 2) for a in _wrap(np.asarray(arms_deg) - yaw)],
                 "road_sides": sorted(covered), "required_sides": sorted(req), "required_met": sorted(req & covered),
                 "in_view": _in_view(arms_deg, yaw)}
