"""The camera that took a backplate, from objects of known height in it: camera height and horizon row (=> pitch).

A 2D photo has no depth, and its horizon is often unsure (the automatic detector finds the roof line, D41) while the
camera height is a guess (155 cm), so sizing the officer from the horizon alone puts them out of scale. People (and
vehicles) standing on the ground tell both. On flat ground, with a camera at height h_c whose horizon is row v0, an
object of real height H whose ground point is on row v is (Hoiem et al. 2006, "Putting objects in perspective")
        height_px = H * (v - v0) / h_c          i.e.   height_px / H = a * v - b,   a = 1 / h_c,  b = v0 / h_c
exact for a level camera, and a good approximation for the small pitches of street photos. Each object gives one linear
equation in (a, b); objects at different rows (distances) separate the two. Weighted least squares over the consensus
set of all object pairs (robust to a child, a sitting person, a box around two cars), then the fit must be well
conditioned: enough spread in rows, small standard errors.

Order (`calibrate`):
  1. people: at least MIN_OBJECTS standing people / riders, all assumed PRIORS height (165 cm standing);
  2. objects: people + vehicles (car, bus, truck, motorbike, bicycle) + barriers marked by hand, at least MIN_OBJECTS;
  3. horizon: the photo's horizon row (set by hand or detected), camera height from the found objects at that horizon
     (median), else camera_height_cm / the default of the tags;
  4. none: no horizon either.

At generation (scene.py, size_reference "auto") the virtual camera is put at that height and pitched so that its horizon
is the photo's horizon row in the composed frame (`pitch_for_horizon`); the officer's distance is chosen so that they
are in the frame and big enough (`feasible_gap`), and their horizontal spot keeps them on the road and off any object
nearer to the camera (`free_offsets`).

Contents:
  constants: PRIORS, MIN_OBJECTS, INLIER_Z, MIN_ROW_GAP, MAX_HORIZON_SE, MAX_HEIGHT_REL_SE, CAMERA_HEIGHT_RANGE,
             PIXEL_NOISE, SIDE_MARGIN, NEAR_TOLERANCE
  reference_objects — Objects of a library entry usable for calibration (riders on one bike merged)
  fit_ground — Camera height and horizon row from objects, or None with the reason
  calibrate — How the camera of a background is known: people, objects, horizon or none
  pitch_for_horizon — UE pitch that puts the horizon on a canvas row
  gap_for_distance / distance_for_gap — Feet row below the horizon <-> horizontal distance
  feasible_gap — Range of feet rows below the horizon where the officer fits the frame and is big enough
  free_offsets — Horizontal crop offsets that keep the officer on the ground and off nearer objects
"""
from __future__ import annotations

import numpy as np

# kind: (real height cm, relative spread). People: the user's assumption (adults ~165 cm); a seated rider measured from the
# head to the bottom of the wheels ~0.9 of that (people.RIDER_TO_STANDING). Vehicles: typical Vietnamese street traffic.
PRIORS = {
    "standing": (165.0, 0.06), "rider": (150.0, 0.08),
    "car": (150.0, 0.12), "bus": (310.0, 0.10), "truck": (300.0, 0.25),
    "motorbike": (110.0, 0.12), "bicycle": (100.0, 0.10), "barrier": (100.0, 0.20),
}
PEOPLE = ("standing", "rider")
MIN_OBJECTS = 3                     # inliers needed for a fit (2 fix a line; the third checks it)
INLIER_Z = 2.5                      # residual / expected spread above which an object is an outlier
MIN_ROW_GAP = 0.02                  # two objects closer than this (x photo height) in rows do not fix a line
MAX_HORIZON_SE = 0.04               # standard error of the horizon row, x photo height
MAX_HEIGHT_REL_SE = 0.2             # standard error of the camera height / camera height
CAMERA_HEIGHT_RANGE = (50.0, 1000.0)
PIXEL_NOISE = 2.0                   # px, box edge error of the detector
SIDE_MARGIN = 0.1                   # free space kept between the officer and a nearer object, x the officer's height
NEAR_TOLERANCE = 0.02               # an object whose feet are above the officer's by less than this x frame height is as near


def reference_objects(entry: dict) -> list[dict]:
    """People ("people") and objects ("objects") of a library entry with a known kind, as
    {"kind", "v", "h", "H", "sigma", "bbox", "feet"}; two riders on one bike (same wheels) count once (the taller)."""
    out = []
    for o in [*((entry.get("people") or {}).get("found") or []), *((entry.get("objects") or {}).get("found") or [])]:
        if o["kind"] not in PRIORS:
            continue
        H, rel = PRIORS[o["kind"]]
        H = float(o.get("height_cm", H))
        out.append({"kind": o["kind"], "v": float(o["feet"][1]), "h": float(o["height_px"]), "H": H, "sigma": rel,
                    "bbox": [float(x) for x in o["bbox"]], "feet": [float(x) for x in o["feet"]]})
    tol = 0.01 * float(entry.get("height", 1000))
    keep: list[dict] = []
    for o in sorted(out, key=lambda o: -o["h"]):
        if o["kind"] == "rider" and any(k["kind"] == "rider" and abs(k["v"] - o["v"]) <= tol
                                        and abs(k["feet"][0] - o["feet"][0]) <= tol for k in keep):
            continue
        keep.append(o)
    return sorted(keep, key=lambda o: (o["v"], o["feet"][0]))


def _wls(v, y, w):
    A = np.stack([v, -np.ones_like(v)], axis=1)
    AtW = A.T * w
    M = AtW @ A
    if abs(np.linalg.det(M)) < 1e-18:
        return None
    a, b = np.linalg.solve(M, AtW @ y)
    return float(a), float(b), M


def fit_ground(objs: list[dict], img_h: float) -> tuple[dict | None, str]:
    """(fit, reason). fit = {"camera_height_cm", "horizon_row", "height_se_cm", "horizon_se_px", "inliers": [index], "rms_z"}
    with the horizon in photo pixels; None (with the reason) when there are fewer than MIN_OBJECTS consistent objects or
    the fit is too unsure."""
    n = len(objs)
    if n < MIN_OBJECTS:
        return None, f"{n} object(s), {MIN_OBJECTS} needed"
    v = np.array([o["v"] for o in objs])
    y = np.array([o["h"] / o["H"] for o in objs])
    sig = np.hypot(np.array([o["sigma"] for o in objs]) * y, PIXEL_NOISE / np.array([o["H"] for o in objs]))
    w = 1.0 / sig ** 2

    def z_of(a, b):
        return (y - (a * v - b)) / sig

    best, best_key = np.ones(n, bool), None
    for i in range(n):                                           # consensus of the line through every usable pair
        for j in range(i + 1, n):
            if abs(v[i] - v[j]) < MIN_ROW_GAP * img_h:
                continue
            a = (y[i] - y[j]) / (v[i] - v[j])
            if a <= 0:
                continue
            b = a * v[i] - y[i]
            z = np.abs(z_of(a, b))
            inl = z < INLIER_Z
            key = (int(inl.sum()), -float(np.minimum(z, INLIER_Z) @ np.minimum(z, INLIER_Z)))
            if best_key is None or key > best_key:
                best, best_key = inl, key
    for _ in range(3):                                           # refine on the inliers, then take the inliers of the refined line
        if best.sum() < MIN_OBJECTS:
            return None, f"only {int(best.sum())} of {n} objects agree on one ground plane"
        r = _wls(v[best], y[best], w[best])
        if r is None:
            return None, "objects on one row: the distance cannot be told"
        a, b, M = r
        new = np.abs(z_of(a, b)) < INLIER_Z
        if (new == best).all():
            break
        best = new
    if best.sum() < MIN_OBJECTS:
        return None, f"only {int(best.sum())} of {n} objects agree on one ground plane"
    if a <= 0:
        return None, "nearer objects are not taller: no ground plane"
    k = int(best.sum())
    zin = z_of(a, b)[best]
    s2 = max(1.0, float(zin @ zin) / max(1, k - 2))                  # residuals larger than the priors widen the errors
    cov = np.linalg.inv(M) * s2
    hc, v0 = 1.0 / a, b / a
    se_hc = float(np.sqrt(cov[0, 0])) / a ** 2
    J = np.array([-b / a ** 2, 1.0 / a])
    se_v0 = float(np.sqrt(J @ cov @ J))
    fit = {"camera_height_cm": hc, "horizon_row": v0, "height_se_cm": se_hc, "horizon_se_px": se_v0,
           "inliers": [int(i) for i in np.flatnonzero(best)], "rms_z": float(np.sqrt(np.mean(zin ** 2)))}
    if not CAMERA_HEIGHT_RANGE[0] <= hc <= CAMERA_HEIGHT_RANGE[1]:
        return None, f"camera height {hc:.0f} cm is implausible"
    if se_v0 > MAX_HORIZON_SE * img_h or se_hc > MAX_HEIGHT_REL_SE * hc:
        return None, (f"too unsure (horizon +-{se_v0 / img_h:.1%} of the height, camera height +-{se_hc:.0f} cm): "
                      "the objects are at too similar distances")
    if (v[best] <= v0).any():
        return None, "an object stands above the horizon"
    return fit, "ok"


def calibrate(entry: dict, default_camera_height_cm: float) -> dict:
    """{"method": "people" | "objects" | "horizon" | "none", "camera_height_cm", "horizon_y_frac", "objects_used",
    "candidates", "reason", ...}: the camera of a library entry (horizon as a fraction of the photo height, may be outside
    0..1). default_camera_height_cm: the horizon fallback without objects (camera_height_cm of the entry or of its tags)."""
    H = float(entry["height"])
    objs = reference_objects(entry)
    people = [o for o in objs if o["kind"] in PEOPLE]
    reasons = []
    for method, group in (("people", people), ("objects", objs)):
        if method == "objects" and len(group) == len(people) and reasons:
            reasons.append("objects: no vehicle / barrier besides the people")
            continue
        fit, why = fit_ground(group, H)
        if fit is not None:
            return {"method": method, "camera_height_cm": round(fit["camera_height_cm"], 2),
                    "horizon_y_frac": round(fit["horizon_row"] / H, 5), "objects_used": len(fit["inliers"]),
                    "candidates": len(group), "height_se_cm": round(fit["height_se_cm"], 2),
                    "horizon_se_frac": round(fit["horizon_se_px"] / H, 5),
                    "kinds": sorted({group[i]["kind"] for i in fit["inliers"]}), "reason": "ok"}
        reasons.append(f"{method}: {why}")
    hf = entry.get("horizon_y_frac")
    if hf is None:
        return {"method": "none", "objects_used": 0, "candidates": len(objs), "reason": "; ".join(reasons + ["no horizon"])}
    v0 = hf * H
    below = [o for o in objs if o["v"] > v0 + 0.02 * H]
    if below and not entry.get("camera_height_cm"):
        est = np.array([o["H"] * (o["v"] - v0) / o["h"] for o in below])
        hc = float(np.median(est))
        if CAMERA_HEIGHT_RANGE[0] <= hc <= CAMERA_HEIGHT_RANGE[1]:
            return {"method": "horizon", "camera_height_cm": round(hc, 2), "horizon_y_frac": float(hf),
                    "objects_used": len(below), "candidates": len(objs), "camera_height_from": "objects",
                    "reason": "; ".join(reasons)}
    return {"method": "horizon", "camera_height_cm": float(default_camera_height_cm), "horizon_y_frac": float(hf),
            "objects_used": 0, "candidates": len(objs),
            "camera_height_from": "manual" if entry.get("camera_height_cm") else "default", "reason": "; ".join(reasons)}


def pitch_for_horizon(horizon_row: float, cy: float, fy: float) -> float:
    """UE pitch (degrees, + = up) of a pinhole camera whose horizon is on that row (camera.horizon_v inverted, roll ignored)."""
    return float(np.degrees(np.arctan((horizon_row - cy) / fy)))


def gap_for_distance(d: float, cam_h: float, pitch_deg: float, fy: float) -> float:
    """Rows between the horizon and a ground point at horizontal distance d (straight ahead) of a camera cam_h high."""
    t = -np.radians(pitch_deg)                                       # downward tilt
    return float(fy * cam_h / (np.cos(t) * (d * np.cos(t) + cam_h * np.sin(t))))


def distance_for_gap(g: float, cam_h: float, pitch_deg: float, fy: float) -> float:
    t = -np.radians(pitch_deg)
    return float((fy * cam_h / (g * np.cos(t)) - cam_h * np.sin(t)) / np.cos(t))


def feasible_gap(v0: float, cam_h: float, top_cm: float, band: tuple[float, float], min_px: float) -> tuple[float, float] | None:
    """[lo, hi] of g = feet row - horizon row (canvas px) where an officer top_cm tall (linear model: height = g * top_cm / cam_h)
    is at least min_px tall, has the feet above band[1] and the top below band[0]; None when no row works."""
    top, bottom = band
    k = top_cm / cam_h                                               # officer height px per row below the horizon
    lo, hi = min_px / k, bottom - v0
    if k > 1:                                                        # head above the horizon: it reaches the top first
        hi = min(hi, (v0 - top) / (k - 1))
    elif k < 1 and v0 < top:                                         # head below the horizon, horizon above the band
        lo = max(lo, (top - v0) / (1 - k))
    lo = max(lo, 1.0)
    return (lo, hi) if hi >= lo else None


def free_offsets(cands: np.ndarray, scale: float, oy: float, objs: list[dict], officer_feet_v: float, officer_rows: tuple[float, float],
                 officer_u_range: tuple[float, float], officer_h_px: float, frame_h: float, feet_uv, mask: np.ndarray | None,
                 scaled: tuple[int, int], width: int, height: int) -> np.ndarray:
    """The horizontal crop offsets (of `cands`) at which the officer (drawn over the photo) does not cover an object that is
    nearer to the camera than them (feet lower in the frame, or about as low) and overlaps their rows, and the footprint
    (feet_uv, canvas pixels) is on the ground mask when there is one. Objects farther away may be covered."""
    clear = np.ones(len(cands), bool)
    pad = SIDE_MARGIN * officer_h_px
    for o in objs:
        x0, y0, x1, y1 = (c * scale for c in o["bbox"])
        y0, y1, feet = y0 + oy, y1 + oy, o["v"] * scale + oy
        if feet < officer_feet_v - NEAR_TOLERANCE * frame_h or y1 < officer_rows[0] or y0 > officer_rows[1]:
            continue
        clear &= (x1 + cands + pad <= officer_u_range[0]) | (x0 + cands - pad >= officer_u_range[1])
    if mask is not None:
        uv = np.asarray(feet_uv, float).reshape(-1, 2)
        uv = uv[np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)]
        if len(uv):
            mh, mw = mask.shape
            sw, sh = scaled
            rows = np.clip(((uv[:, 1] - oy) * mh / sh).astype(int), 0, mh - 1)
            cols = np.clip(((uv[None, :, 0] - cands[:, None]) * mw / sw).astype(int), 0, mw - 1)
            clear &= mask[rows[None, :], cols].all(axis=1)
    return cands[clear]
