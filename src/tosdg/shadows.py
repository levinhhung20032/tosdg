"""Shadows: the officer's cast shadow on the ground, and the sun read from the shadows of the people / vehicles in a photo.

A 2D photo does not say where the sun is. Two sources give the officer's shadow a direction and a length:
  * the photo itself: at import, the shadows that the detected reference people / vehicles (people.py) cast on the road are
    found (`find_photo_shadows`) and stored in the library index as image vectors, feet -> shadow tip:
        "shadows": {"method": ..., "status": "found" | "none", "found": [{"kind", "feet": [u, v], "tip": [u, v],
                    "height_cm", "score"}, ...], "manual": [...]}
    At generation, `sun_from_photo` turns them into a sun direction through the sample's own virtual camera (the photo is
    composed for that camera), so the officer's shadow points to the same vanishing point as the shadows in the picture;
  * a custom vector (Studio, config): the sun's azimuth / elevation, or the shadow's direction and length.
`cast` draws the shadow: every triangle of the posed mesh (the same skinned geometry as the picture and the ground truth) is
projected along the sun onto the ground plane z = 0 and rasterised in the camera, so the shape is exactly the officer's; the
penumbra widens with the height of the casting point, and a contact shadow sits under the shoes. `apply` darkens the
backplate with it before the officer is drawn on top. The shadow is never part of the instance mask.

Contents:
  constants: DEFAULT_SHADOW, PEOPLE_HEIGHT_CM, ...
  find_photo_shadows — shadows of the reference people / vehicles of a photo (image vectors)
  sun_from_photo — sun azimuth / elevation (world, degrees) from a photo's shadow vectors seen through a camera
  sun_from_vector — sun azimuth / elevation from a shadow direction (relative to the camera) and a length / height ratio
  sun_direction — unit vector towards the sun
  cast — shadow coverage of a mesh on the ground plane, (H, W) float in 0..1
  apply — the backplate darkened by a shadow coverage
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_SHADOW = {"strength": 0.55, "softness": 0.25, "contact": 0.5}
PEOPLE_HEIGHT_CM = {"standing": 165.0, "rider": 150.0, "car": 150.0, "motorbike": 110.0, "bicycle": 100.0, "bus": 310.0, "truck": 300.0}
MIN_SUN_ELEVATION = 12.0        # degrees; a lower sun gives shadows longer than the frame: never drawn
MAX_SUN_ELEVATION = 85.0
PENUMBRA_TAN = (0.01, 0.12)     # penumbra width per cm of casting height at softness 0 / 1 (the sun's disc is ~0.009)
CONTACT_HEIGHT_CM = 6.0         # vertices this low (soles) cast the contact shadow straight down
FIND_WIDTH = 1280               # photos are searched at this width
RASTER_MAX_W = 800              # cast(): the shadow is rasterised at most this wide, then scaled up
SUNNY_SKY = 0.3                 # sky_blueness of a photo taken in sunshine
AGREE_DEG = 25.0                # sun_from_photo: vectors within this angle (on the ground) agree on one sun
MIN_AGREE = 2                   # a photo has sun shadows when at least this many objects show theirs under one sun
AZ_STEP = 5.0                   # find_photo_shadows: suns tried every AZ_STEP degrees ...
ELEVATIONS = (15, 20, 25, 30, 35, 40, 45, 50, 55, 60)       # ... at these elevations (a higher sun's shadow hides in the dark patch at the feet)
CONTACT_SKIP_CM = 40.0          # the first this-many cm from the feet are not scored: dark in any weather (ambient occlusion)
MIN_CONTRAST = 0.22             # a strip is a shadow when it is this much darker than the road beside it (0.22 = 22 %)
MIN_SCORE = 0.12                # mean contrast of all scored strips at the best sun
MIN_PROMINENCE = 0.1            # ... above the median over all suns (a uniformly dark road does not pass)
MIN_SCORED = 3                  # objects whose strip can be scored at the best sun (two can always be lined up with some dark patch)
MIN_DARK_FRAC = 0.6             # ... and this share of them must be dark there
MIN_DISTINCT = 0.08             # the best sun beats every sun more than DISTINCT_DEG away by this much
DISTINCT_DEG = 60.0
WIDTH_CM = {"standing": 45.0, "rider": 70.0, "motorbike": 60.0, "bicycle": 50.0, "car": 175.0, "bus": 250.0, "truck": 250.0,
            "barrier": 60.0}


# ----------------------------------------------------------------------------- the sun
def sun_direction(azimuth_deg: float, elevation_deg: float) -> list[float]:
    """Unit vector towards the sun, UE world frame (as lighting.sample_lighting)."""
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    return [float(np.cos(el) * np.cos(az)), float(np.cos(el) * np.sin(az)), float(np.sin(el))]


def sun_from_vector(camera_yaw_deg: float, direction_deg: float, length_ratio: float) -> tuple[float, float]:
    """(azimuth, elevation) of the sun for a shadow that points `direction_deg` on the ground relative to the camera's view
    (0 = away from the camera, 90 = to the right, 180 = towards the camera, -90 = left) and is `length_ratio` times as long
    as the object is tall."""
    shadow_az = float(camera_yaw_deg) + float(direction_deg)
    el = float(np.degrees(np.arctan2(1.0, max(float(length_ratio), 1e-3))))
    return (shadow_az + 180.0 + 180.0) % 360.0 - 180.0, float(np.clip(el, MIN_SUN_ELEVATION, MAX_SUN_ELEVATION))


def vector_of_sun(camera_yaw_deg: float, azimuth_deg: float, elevation_deg: float) -> tuple[float, float]:
    """Inverse of sun_from_vector: (direction relative to the camera, length / height ratio)."""
    d = (float(azimuth_deg) + 180.0 - float(camera_yaw_deg) + 180.0) % 360.0 - 180.0
    return d, float(1.0 / np.tan(np.radians(max(float(elevation_deg), 1e-3))))


def _ground_point(cam, uv) -> np.ndarray | None:
    u, v = float(uv[0]), float(uv[1])
    d = cam.R @ np.array([1.0, (u - cam.cx) / cam.fx, -(v - cam.cy) / cam.fy])
    z0 = float(cam.position[2])
    if d[2] >= -1e-9 or z0 <= 0:
        return None
    return np.asarray(cam.position, float) + d * (z0 / -d[2])


def sun_from_photo(shadows: dict | None, camera, composition: dict) -> dict | None:
    """The sun of a photo from its stored shadow vectors, seen through `camera` (the pinhole camera of the composed canvas:
    photo pixel p -> canvas pixel p * scale + offset). Manual vectors win over found ones. Each vector gives a ground
    direction (feet -> tip) and, with the object's height, an elevation; the largest group within AGREE_DEG decides
    (median). None when the photo has no usable vector."""
    if not shadows:
        return None
    vecs = shadows.get("manual") or (shadows.get("found") if shadows.get("status") == "found" else None)
    if not vecs:
        return None
    s, (ox, oy) = float(composition["scale"]), composition["offset_px"]
    az, el, w = [], [], []
    for r in vecs:
        f = _ground_point(camera, (r["feet"][0] * s + ox, r["feet"][1] * s + oy))
        t = _ground_point(camera, (r["tip"][0] * s + ox, r["tip"][1] * s + oy))
        if f is None or t is None:
            continue
        d = t[:2] - f[:2]
        L = float(np.hypot(*d))
        if L < 1.0:
            continue
        az.append(float(np.degrees(np.arctan2(d[1], d[0]))) + 180.0)          # the sun is opposite the shadow
        el.append(float(np.degrees(np.arctan2(float(r.get("height_cm", 165.0)), L))))
        w.append(float(r.get("score", 1.0)))
    if not az:
        return None
    az, el, w = np.array(az), np.array(el), np.array(w)
    best = None
    for a in az:                                                               # largest agreeing group
        diff = np.abs((az - a + 180.0) % 360.0 - 180.0)
        grp = diff <= AGREE_DEG
        key = (float(w[grp].sum()), -float(diff[grp].sum()))
        if best is None or key > best[0]:
            best = (key, grp)
    grp = best[1]
    ref = az[grp][0]
    rel = (az[grp] - ref + 180.0) % 360.0 - 180.0
    a = float((ref + np.median(rel) + 180.0) % 360.0 - 180.0)
    e = float(np.clip(np.median(el[grp]), MIN_SUN_ELEVATION, MAX_SUN_ELEVATION))
    return {"azimuth_deg": round(a, 3), "elevation_deg": round(e, 3), "vectors_used": int(grp.sum()), "vectors": int(len(az)),
            "source": "manual" if shadows.get("manual") else "found"}


# ----------------------------------------------------------------------------- reading the photo
def _luma(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def sky_blueness(img: Image.Image) -> float:
    """Median saturation of the bright, bluish pixels in the top fifth of a photo: a clear sky is blue (0.3 and more), an
    overcast, hazy or night sky is grey or dark (0.0 - 0.2). A clear sky is the photo-wide sign that the sun casts shadows."""
    im = img.convert("RGB")
    im.thumbnail((640, 640))
    a = np.asarray(im, np.float32) / 255.0
    top = a[: max(1, a.shape[0] // 5)]
    mx, mn = top.max(axis=2), top.min(axis=2)
    sky = (mx > 0.55) & (top[..., 2] >= top[..., 0])
    if sky.sum() < 50:
        return 0.0
    return float(np.median(((mx - mn) / np.maximum(mx, 1e-3))[sky]))


def find_photo_shadows(img: Image.Image, references: list[dict], ground: np.ndarray | None = None,
                       horizon_row: float | None = None, camera_height_cm: float | None = None, tags=()) -> dict:
    """The sun of a photo, read from the shadows its reference people / vehicles cast on the road.

    `references`: calibrate.reference_objects of the entry ({"kind", "feet", "bbox", "H"} in photo pixels); `ground`: the
    photo's standable mask (any size) or None; `horizon_row`, `camera_height_cm`: the photo's camera (calibrate.py);
    `tags`: the photo's tags (night / dusk: no sun).
    Weather first: a photo tagged night / dusk, or whose sky is not blue (sky_blueness < SUNNY_SKY), has no sun shadows
    ("weather": "night" | "dusk" | "overcast"). A sunny photo then gets a direction when its objects show one.
    Model-based search: a nominal camera (focal = photo width) at that height and horizon puts every object on the ground
    plane; for each sun (azimuth every AZ_STEP degrees around the view, elevation in ELEVATIONS) each object's shadow is a
    strip on the ground from its feet away from the sun, as wide as the object, as long as height / tan(elevation). A strip
    is scored by how much darker it is than the road on both sides of it (photo pixels on the road and outside every
    detection box only). The sun is the one that darkens the strips of most objects at once: a curb, a crack or the dark
    patch under a car lines up with one object, not with all of them. The photo has sun shadows when the best score is
    high, stands out from the other suns, and MIN_AGREE objects show their shadow there.
    Returns the index block {"method", "status": "found" | "none", "weather": "sunny" | "overcast" | "night" | "dusk",
    "found": [{"kind", "feet", "tip", "height_cm", "score"}], "sun_nominal": {...}, "reason"}, vectors in photo pixels
    (feet -> tip of the predicted shadow)."""
    method = f"sky>={SUNNY_SKY:g}+strip-contrast:az{AZ_STEP:g}deg,el{ELEVATIONS[0]:g}-{ELEVATIONS[-1]:g},min{MIN_CONTRAST:g}"
    for tag in ("night", "dusk"):
        if tag in tags:
            return {"method": method, "status": "none", "weather": tag, "found": [], "reason": f"photo tagged {tag}: no sun"}
    blue = sky_blueness(img)
    if blue < SUNNY_SKY:
        return {"method": method, "status": "none", "weather": "overcast", "sky_blueness": round(blue, 3), "found": [],
                "reason": f"grey sky (blueness {blue:.2f} < {SUNNY_SKY}): overcast or hazy, no sun shadows"}
    sunny = {"weather": "sunny", "sky_blueness": round(blue, 3)}
    if horizon_row is None:
        return {"method": method, "status": "none", **sunny, "found": [], "reason": "sunny, but the photo's horizon is unknown"}
    W0, H0 = img.size
    k = min(1.0, FIND_WIDTH / W0)
    w, h = max(1, round(W0 * k)), max(1, round(H0 * k))
    rgb = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255.0
    Y = _luma(rgb)
    usable = np.ones((h, w), bool)
    if ground is not None:
        usable = np.asarray(Image.fromarray(ground.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    else:
        usable[: int(np.clip(horizon_row * k, 0, h))] = False
    for r in references:
        x0, y0, x1, y1 = (v * k for v in r["bbox"])
        usable[int(max(0, y0)):int(min(h, y1)), int(max(0, x0)):int(min(w, x1))] = False
    if usable.sum() < 500:
        return {"method": method, "status": "none", **sunny, "found": [], "reason": "sunny, but too little visible road"}
    Y = np.minimum(Y, float(np.percentile(Y[usable], 95)))                  # lane paint does not light the flanks
    f, cx, v0, hc = float(W0), W0 / 2.0, float(horizon_row), float(camera_height_cm or 150.0)
    objs = []
    for r in references:
        dv = float(r["feet"][1]) - v0
        if dv < 0.02 * H0:
            continue
        Z = f * hc / dv
        objs.append({**r, "G": np.array([(float(r["feet"][0]) - cx) * Z / f, Z]),
                     "Hc": float(r.get("H", PEOPLE_HEIGHT_CM.get(r["kind"], 165.0))), "Wc": WIDTH_CM.get(r["kind"], 60.0)})
    if len(objs) < MIN_AGREE:
        return {"method": method, "status": "none", **sunny, "found": [],
                "reason": f"sunny, but {len(objs)} object(s) below the horizon, {MIN_AGREE} needed for the direction"}
    az = np.radians(np.arange(0.0, 360.0, AZ_STEP))                          # shadow direction on the ground, 0 = away from the camera
    el = np.radians(np.asarray(ELEVATIONS, float))
    D = np.stack([np.sin(az), np.cos(az)], axis=1)                           # (A, 2) ground (lateral, depth)
    Nrm = np.stack([np.cos(az), -np.sin(az)], axis=1)                        # sideways on the ground
    ts, ss = np.linspace(0.35, 1.0, 6), np.linspace(-0.4, 0.4, 3)
    tb_frac = np.linspace(1.15, 1.5, 3)                                      # just past the tip

    def sample(P):                                                          # ground points (..., 2) -> luminance, valid
        X, Z = P[..., 0], P[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (cx + f * X / Z) * k
            v = (v0 + f * hc / Z) * k
        ok = (Z > 1.0) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        ui = np.clip(np.nan_to_num(u), 0, w - 1).astype(int)
        vi = np.clip(np.nan_to_num(v), 0, h - 1).astype(int)
        ok &= usable[vi, ui]
        return Y[vi, ui], ok

    C = np.full((len(objs), len(az), len(el)), np.nan)                       # contrast of each object's strip, each sun
    for i, o in enumerate(objs):
        L = o["Hc"] / np.tan(el)                                             # (E,)
        start = 0.5 * o["Wc"] if o["kind"] not in ("standing", "rider") else 0.0
        t = start + np.maximum(ts[None, :] * L[:, None], CONTACT_SKIP_CM)     # (E, T)
        base = o["G"][None, None, None, :] + D[:, None, None, :] * t[None, :, :, None]        # (A, E, T, 2)
        inside = base[..., None, :] + Nrm[:, None, None, None, :] * (ss * o["Wc"])[None, None, None, :, None]
        side = 1.1 * o["Wc"]
        left = base + Nrm[:, None, None, :] * side
        right = base - Nrm[:, None, None, :] * side
        yi, oi = sample(inside)
        yl, ol = sample(left)
        yr, orr = sample(right)
        n_in = oi.sum(axis=(2, 3))
        m_in = np.where(oi, yi, 0).sum(axis=(2, 3)) / np.maximum(n_in, 1)
        n_fl = ol.sum(axis=2) + orr.sum(axis=2)
        m_fl = (np.where(ol, yl, 0).sum(axis=2) + np.where(orr, yr, 0).sum(axis=2)) / np.maximum(n_fl, 1)
        both = (ol.sum(axis=2) >= 2) & (orr.sum(axis=2) >= 2)               # road on both sides: an edge is not a strip
        ok = (n_in >= 0.5 * oi.shape[2] * oi.shape[3]) & both
        # past the tip the road is lit again: a strip shorter than the real shadow is still dark inside, but its "beyond" is not
        tb = start + np.maximum(tb_frac[None, :] * L[:, None], CONTACT_SKIP_CM)
        beyond = (o["G"][None, None, None, :] + D[:, None, None, :] * tb[None, :, :, None])[..., None, :]             + Nrm[:, None, None, None, :] * (ss * o["Wc"])[None, None, None, :, None]
        yb, ob = sample(beyond)
        n_b = ob.sum(axis=(2, 3))
        m_b = np.where(ob, yb, 0).sum(axis=(2, 3)) / np.maximum(n_b, 1)
        dark_beyond = np.where(n_b >= 3, np.clip((m_fl - m_b) / np.maximum(m_fl, 1e-3), 0.0, None), 0.0)
        C[i] = np.where(ok, (m_fl - m_in) / np.maximum(m_fl, 1e-3) - dark_beyond, np.nan)
    n_valid = np.isfinite(C).sum(axis=0)
    hits = (np.nan_to_num(C, nan=0.0) >= MIN_CONTRAST).sum(axis=0)
    S = np.where(n_valid >= MIN_AGREE, np.nansum(np.clip(np.nan_to_num(C, nan=0.0), -0.5, 0.8), axis=0) / np.maximum(n_valid, 1), -1.0)
    eligible = (n_valid >= MIN_SCORED) & (hits >= np.maximum(MIN_AGREE, np.ceil(MIN_DARK_FRAC * n_valid)))
    ia, ie = np.unravel_index(int(np.argmax(np.where(eligible, S, -1.0))), S.shape)
    best, n_hit, n_sc = float(S[ia, ie]), int(hits[ia, ie]), int(n_valid[ia, ie])
    rest = S[S > -1.0]
    prominence = best - float(np.median(rest)) if rest.size else 0.0
    far = np.abs((np.degrees(az) - np.degrees(az[ia]) + 180.0) % 360.0 - 180.0) > DISTINCT_DEG
    rival = float(S[far].max()) if far.any() else -1.0
    nominal = {"direction_deg": float(np.degrees(az[ia])), "elevation_deg": float(np.degrees(el[ie])), "score": round(best, 3),
               "prominence": round(prominence, 3), "distinct": round(best - rival, 3), "objects_scored": n_sc, "objects_dark": n_hit}
    if not eligible.any():
        return {"method": method, "status": "none", **sunny, "found": [], "sun_nominal": nominal,
                "reason": (f"sunny, direction unclear: no sun darkens the shadows of {MIN_SCORED} or more people / vehicles with "
                           "open road around them; mark a shadow by hand in Studio")}
    if not eligible[ia, ie] or best < MIN_SCORE or prominence < MIN_PROMINENCE or best - rival < MIN_DISTINCT:
        return {"method": method, "status": "none", **sunny, "found": [], "sun_nominal": nominal,
                "reason": (f"sunny, direction unclear: at the best sun {n_hit} of {n_sc} object strips are dark (score {best:.2f}, "
                           f"{prominence:.2f} above the median, {best - rival:.2f} above other directions); mark a shadow by hand in Studio")}
    found = []
    for i, o in enumerate(objs):
        c = C[i, ia, ie]
        if not np.isfinite(c) or c < MIN_CONTRAST:
            continue
        tip = o["G"] + D[ia] * (o["Hc"] / np.tan(el[ie]))
        if tip[1] <= 1.0:
            continue
        tu, tv = cx + f * tip[0] / tip[1], v0 + f * hc / tip[1]
        found.append({"kind": o["kind"], "feet": [round(float(o["feet"][0]), 1), round(float(o["feet"][1]), 1)],
                      "tip": [round(float(tu), 1), round(float(tv), 1)], "height_cm": round(o["Hc"], 1), "score": round(float(c), 3)})
    return {"method": method, "status": "found", **sunny, "found": found, "sun_nominal": nominal, "reason": "ok"}


# ----------------------------------------------------------------------------- drawing
def _raster(uv: np.ndarray, value: np.ndarray, tris: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Coverage of projected triangles (any winding) and, per pixel, the smallest per-vertex `value` interpolated
    (the lowest casting point). Returns (covered bool (H, W), value (H, W), inf where empty)."""
    from .render_mesh import rasterize
    sx, sy = np.nan_to_num(uv[:, 0], nan=-1e6), np.nan_to_num(uv[:, 1], nan=-1e6)
    ok = np.isfinite(uv[tris]).all(axis=(1, 2))
    t = tris[ok]
    tri_id, bary = rasterize(sx, sy, -value, t, W, H)                     # the z-buffer keeps the largest: -value = the lowest point
    cov = tri_id >= 0
    out = np.full((H, W), np.inf, np.float32)
    ids = tri_id[cov]
    out[cov] = (bary[cov] * value[t[ids]]).sum(axis=1)
    return cov, out


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a (H, W) image in 0..1 (8-bit inside: a shadow darkens by at most ~70 %, so a step of 1/255 is invisible)."""
    im = Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(sigma))), np.float32) / 255.0


def cast(pos: np.ndarray, tris: np.ndarray, sun: list[float] | np.ndarray, camera, softness: float = DEFAULT_SHADOW["softness"],
         contact: float = DEFAULT_SHADOW["contact"], sun_on: bool = True) -> np.ndarray:
    """Shadow coverage (H, W) float in 0..1 of a posed mesh (`pos` world vertices, `tris`) on the ground plane z = 0, in the
    pixels of `camera` (pinhole). Sun shadow: each vertex slides along the sun to the ground, the triangles are rasterised
    (exactly the officer's silhouette as the sun sees it) and blurred by a penumbra that grows with the height of the casting
    point (`softness` 0..1). Contact shadow (`contact` 0..1 of the sun shadow's darkness): the soles straight down, blurred a
    little. `sun_on` False: contact shadow only (no sun: overcast, night). Drawn at most RASTER_MAX_W wide (the same field of
    view) and scaled up: a shadow is soft, the CPU rasteriser is the cost."""
    if camera.width > RASTER_MAX_W:
        from dataclasses import replace
        k = RASTER_MAX_W / camera.width
        small = replace(camera, width=RASTER_MAX_W, height=max(1, round(camera.height * k)), lens=None)
        cov = cast(pos, tris, sun, small, softness, contact, sun_on)
        return np.asarray(Image.fromarray(cov, mode="F").resize((camera.width, camera.height), Image.BILINEAR), np.float32)
    W, H = camera.width, camera.height
    pos = np.asarray(pos, float)
    out = np.zeros((H, W), np.float32)
    zs = np.maximum(pos[:, 2], 0.0)
    _, dep = camera.project_pinhole(pos * [1, 1, 0])
    near = np.isfinite(dep) & (dep > 1.0)
    if not near.any():
        return out
    # pixels per cm at the officer's feet: blur radii in pixels
    g = pos[np.argmin(pos[:, 2])] * [1, 1, 0]
    uv_g, d_g = camera.project_pinhole(np.array([g, g + [0, 0, 100.0]]))
    px_per_cm = float(abs(uv_g[0, 1] - uv_g[1, 1]) / 100.0) if np.isfinite(uv_g).all() else 1.0
    s = np.asarray(sun, float)
    s = s / max(np.linalg.norm(s), 1e-9)
    if sun_on and s[2] > np.sin(np.radians(MIN_SUN_ELEVATION * 0.5)):
        flat = pos - (zs / s[2])[:, None] * s                                 # along the sun to z = 0
        flat[:, 2] = 0.0
        uv, _ = camera.project_pinhole(flat)
        cov, hgt = _raster(uv, zs.astype(np.float32), tris, W, H)
        if cov.any():
            pen = np.interp(softness, [0.0, 1.0], PENUMBRA_TAN)                # penumbra width / casting height
            h_hi = float(np.percentile(hgt[cov], 90))
            base = cov.astype(np.float32)
            sig_lo = max(0.6, float(pen) * 5.0 * px_per_cm)
            sig_hi = max(sig_lo, float(pen) * h_hi * px_per_cm * 0.5)
            lo, hi = _blur(base, sig_lo), _blur(base, sig_hi)
            hn = np.where(cov, np.clip(hgt / max(h_hi, 1.0), 0, 1), 0.0).astype(np.float32)
            t = np.clip(_blur(hn, sig_hi) / np.maximum(hi, 1e-3), 0, 1)        # average casting height around each pixel
            out = np.maximum(out, (1 - t) * lo + t * hi)
    if contact > 0:
        low = zs <= CONTACT_HEIGHT_CM
        keep = low[tris].all(axis=1)
        if keep.any():
            uv, _ = camera.project_pinhole(pos * [1, 1, 0])
            cov, _ = _raster(uv, zs.astype(np.float32), tris[keep], W, H)
            sig = max(1.0, 2.5 * float(px_per_cm))
            grown = np.asarray(Image.fromarray(cov.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3)), np.float32) / 255.0
            out = np.maximum(out, float(contact) * _blur(grown, sig))
    return out


def apply(plate: Image.Image, coverage: np.ndarray, strength: float = DEFAULT_SHADOW["strength"]) -> Image.Image:
    """The backplate darkened by `coverage` (H, W in 0..1, the size of the plate): x (1 - strength * coverage), a little less
    on blue (the sky still lights a shadow)."""
    a = np.asarray(plate.convert("RGB"), np.float32)
    k = float(strength) * coverage[..., None] * np.array([1.0, 0.97, 0.88], np.float32)
    return Image.fromarray(np.clip(a * (1.0 - k) + 0.5, 0, 255).astype(np.uint8))
