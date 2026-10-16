"""Lens distortion (radial, Brown model) and the black area outside a lens' image circle.

Model, in normalised image coordinates x = (u - cx) / fx, y = (v - cy) / fx and r^2 = x^2 + y^2:
    x_d = x * (1 + k1 r^2 + k2 r^4)            (same for y)
k1 < 0 is barrel distortion (wide-angle, dashcam / action camera / CCTV), k1 > 0 pincushion. The polynomial is only used up to the
radius where it stops increasing (beyond that the picture would fold back on itself): there projected points are NaN and the pixels black.

`image_circle` is the radius of the lit part of the frame as a fraction of the half-diagonal (0 = the whole frame is lit; below 1 the
corners go black, as in cheap wide-angle CCTV or fisheye-ish lenses).

Ground truth: `Camera.project` applies this distortion, so keypoints / baton / full bbox come out in the final distorted image.
Pixels: the renderers only know the pinhole model, so `render_with_lens` renders the scene pinhole on a canvas large enough to hold every
source pixel of the distorted frame, then remaps image and instance mask with the exact inverse of the distortion. The background is drawn on that
same pinhole canvas, so the photo is distorted together with the officer (it behaves as a pinhole picture of the scene).

Contents:
  constants: MAX_CANVAS, NO_LENS
  class Lens
  sample_lens — Draw a lens from a camera profile's "lens" ranges
  render_with_lens — Render an (H, W) image through a lens…
  canvas_size / canvas_camera / pinhole_horizon_row / distort_plate — Size and camera of the pinhole canvas, its horizon row, a backplate as the lens sees it
  layer_with_lens — The same for an RGBA layer (the Studio's 3D model layer)
  (+ 6 private helpers)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import rng as rngmod

MAX_CANVAS = 3.0       # the pinhole render canvas is at most this many times the output frame, per axis


@dataclass(frozen=True)
class Lens:
    k1: float = 0.0
    k2: float = 0.0
    image_circle: float = 0.0

    @property
    def distorts(self) -> bool:
        return self.k1 != 0.0 or self.k2 != 0.0

    @property
    def active(self) -> bool:
        return self.distorts or self.image_circle > 0.0

    def to_dict(self) -> dict:
        return {"model": "brown_radial_k1k2", "k1": float(self.k1), "k2": float(self.k2), "image_circle_frac": float(self.image_circle)}

    # -- polynomial ----------------------------------------------------------------------------------------------
    def _factor(self, r2: np.ndarray) -> np.ndarray:
        return 1.0 + self.k1 * r2 + self.k2 * r2 * r2

    def r_limit(self) -> float:
        """Largest undistorted normalised radius the polynomial can be used up to (where d r_d / d r first reaches 0); inf if never."""
        if not self.distorts:
            return float("inf")
        r = np.linspace(0.0, 8.0, 8001)
        r2 = r * r
        bad = np.nonzero(1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2 <= 1e-6)[0]
        return float(r[bad[0] - 1]) if bad.size else float("inf")

    @staticmethod
    def auto_circle(k1: float, k2: float, width: int, height: int, fx: float) -> float:
        """The image circle that follows the distortion: the corner of the undistorted (pinhole) frame lands at this fraction of the
        half-diagonal, and what lies beyond it is outside the lens' picture (0 = no border, e.g. pincushion or no distortion). It changes with
        k1, k2, the focal length, the sensor and the resolution, which is why the Studio can keep the black border in step with them."""
        lens = Lens(k1, k2)
        if not lens.distorts:
            return 0.0
        rc = float(np.hypot(width, height) / 2.0 / fx)
        d = rc * float(lens._factor(np.float64(rc * rc))) if rc <= lens.r_limit() else lens.r_reach()
        frac = d / rc
        return round(float(np.clip(frac, 0.0, 1.0)), 4) if frac < 1.0 else 0.0

    def r_reach(self) -> float:
        """Largest distorted normalised radius the lens can show (the polynomial's value at r_limit); inf if there is no fold."""
        rl = self.r_limit()
        return float("inf") if rl == float("inf") else float(rl * self._factor(np.float64(rl * rl)))

    def fitted(self, width: int, height: int, fx: float) -> "Lens":
        """This lens scaled down (k1, k2 together) just enough that the whole frame, or the image circle when it is smaller, is
        reachable: a distortion too strong for the field of view would leave black, unusable corners that nobody asked for."""
        need = np.hypot(width, height) / 2.0
        if self.image_circle > 0:
            need = min(need, self.image_circle * need)
        need = need / fx * 1.01
        if not self.distorts or self.r_reach() >= need:
            return self
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if dataclasses.replace(self, k1=self.k1 * mid, k2=self.k2 * mid).r_reach() >= need:
                lo = mid
            else:
                hi = mid
        return dataclasses.replace(self, k1=round(self.k1 * lo, 5), k2=round(self.k2 * lo, 5))

    # -- forward / inverse ---------------------------------------------------------------------------------------
    def distort(self, uv: np.ndarray, cx: float, cy: float, f: float) -> np.ndarray:
        """Pinhole pixel coordinates -> distorted ones. NaN stays NaN; points beyond the usable radius become NaN."""
        uv = np.asarray(uv, float)
        if not self.distorts:
            return uv.copy()
        x, y = (uv[..., 0] - cx) / f, (uv[..., 1] - cy) / f
        r2 = x * x + y * y
        fac = self._factor(r2)
        out = np.stack([cx + f * x * fac, cy + f * y * fac], axis=-1)
        rl = self.r_limit()
        with np.errstate(invalid="ignore"):
            out[r2 > rl * rl] = np.nan
        return out

    def undistort(self, uv: np.ndarray, cx: float, cy: float, f: float) -> tuple[np.ndarray, np.ndarray]:
        """Distorted pixel coordinates -> pinhole ones (Newton on the radius). Returns (uv, ok); ok is False where no pinhole point maps there."""
        uv = np.asarray(uv, float)
        if not self.distorts:
            return uv.copy(), np.ones(uv.shape[:-1], bool)
        x, y = (uv[..., 0] - cx) / f, (uv[..., 1] - cy) / f
        rd = np.hypot(x, y)
        r = rd.copy()
        rl = self.r_limit()
        for _ in range(12):
            r2 = r * r
            g = r * self._factor(r2) - rd
            dg = 1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2
            r = np.clip(r - g / np.maximum(dg, 1e-3), 0.0, min(rl, 1e3))
        r2 = r * r
        ok = (np.abs(r * self._factor(r2) - rd) < 1e-5) & (r <= rl)
        scale = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
        return np.stack([cx + f * x * scale, cy + f * y * scale], axis=-1), ok

    def local_scale(self, uv_pinhole: np.ndarray, cx: float, cy: float, f: float) -> np.ndarray:
        """How much the distortion magnifies a small circle at these pinhole pixels (the larger of the radial and tangential factors)."""
        uv = np.asarray(uv_pinhole, float)
        if not self.distorts:
            return np.ones(uv.shape[:-1])
        r2 = ((uv[..., 0] - cx) ** 2 + (uv[..., 1] - cy) ** 2) / (f * f)
        radial = 1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2
        return np.maximum(np.abs(radial), np.abs(self._factor(r2)))

    def inside_circle(self, uv: np.ndarray, width: int, height: int) -> np.ndarray:
        """True where a pixel position is in the lit image circle (everything when image_circle is 0)."""
        uv = np.asarray(uv, float)
        if self.image_circle <= 0.0:
            return np.ones(uv.shape[:-1], bool)
        with np.errstate(invalid="ignore"):
            return np.hypot(uv[..., 0] - width / 2.0, uv[..., 1] - height / 2.0) <= self.image_circle * np.hypot(width, height) / 2.0


def sample_lens(spec: dict | None, rng: np.random.Generator) -> Lens | None:
    """Lens from a camera profile's "lens" block ({k1, k2, image_circle}: each a [lo, hi] range; "black_border_probability": chance that
    the image circle is used at all, default 1); None when the profile has none. Always draws the same four values in this order, so the
    stream does not depend on which ranges are degenerate."""
    if not spec:
        return None
    k1 = rngmod.uniform(rng, spec.get("k1", [0, 0]))
    k2 = rngmod.uniform(rng, spec.get("k2", [0, 0]))
    circle = rngmod.uniform(rng, spec.get("image_circle", [0, 0]))
    use_border = rng.random() < float(spec.get("black_border_probability", 1.0))
    lens = Lens(round(k1, 5), round(k2, 5), round(circle, 5) if use_border else 0.0)
    return lens if lens.active else None


# ----------------------------------------------------------------------------------------------------------------------
def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a float (h, w, c) image at pixel coordinates (pixel i covers [i, i+1), so its centre is i + 0.5); borders are clamped."""
    h, w = img.shape[:2]
    fx, fy = np.clip(x - 0.5, 0.0, w - 1.0), np.clip(y - 0.5, 0.0, h - 1.0)
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
    top = img[y0, x0] * (1 - ax) + img[y0, x1] * ax
    bot = img[y1, x0] * (1 - ax) + img[y1, x1] * ax
    return top * (1 - ay) + bot * ay


class _Plan:
    """Where every output pixel of the distorted frame comes from in the pinhole render canvas."""

    def __init__(self, cam):
        lens, W, H = cam.lens, cam.width, cam.height
        ys, xs = np.mgrid[0:H, 0:W]
        out_uv = np.stack([xs + 0.5, ys + 0.5], axis=-1)
        src, ok = lens.undistort(out_uv, cam.cx, cam.cy, cam.fx)
        use = ok & lens.inside_circle(out_uv, W, H)
        dx, dy = src[..., 0] - cam.cx, src[..., 1] - cam.cy
        use &= (np.abs(dx) <= MAX_CANVAS * W / 2) & (np.abs(dy) <= MAX_CANVAS * H / 2)
        ex = float(np.abs(dx[use]).max()) if use.any() else 0.0
        ey = float(np.abs(dy[use]).max()) if use.any() else 0.0
        self.Wr = max(W, 2 * int(np.ceil(ex)) + 4)
        self.Hr = max(H, 2 * int(np.ceil(ey)) + 4)
        self.use = use
        self.x = np.where(use, dx + self.Wr / 2.0, 0.0)                       # source position in the render canvas
        self.y = np.where(use, dy + self.Hr / 2.0, 0.0)
        self.cam = dataclasses.replace(cam, lens=None, width=self.Wr, height=self.Hr, sensor_width_mm=cam.sensor_width_mm * self.Wr / W)

    def fit_plate(self, plate: Image.Image) -> Image.Image:
        return plate.convert("RGB") if plate.size == (self.Wr, self.Hr) else plate.convert("RGB").resize((self.Wr, self.Hr), Image.BILINEAR)


def canvas_size(cam) -> tuple[int, int]:
    """Size (Wr, Hr) of the pinhole picture that has to be drawn for this lens camera: everything the distorted frame shows."""
    plan = _Plan(cam)
    return plan.Wr, plan.Hr


def canvas_camera(cam):
    """The pinhole camera that draws the render canvas of a lens camera (size `canvas_size(cam)`, no lens)."""
    return _Plan(cam).cam


def pinhole_horizon_row(cam, canvas_height: int) -> float:
    """Row of the pinhole horizon in a render canvas of that height (the camera's principal point is the canvas centre; roll ignored)."""
    return canvas_height / 2.0 + cam.fy * float(np.tan(np.radians(cam.pitch)))


def distort_plate(plate: Image.Image, cam) -> Image.Image:
    """A backplate drawn for the pinhole canvas (`canvas_size`), as the lens camera sees it: distorted, black outside the image circle."""
    plan = _Plan(cam)
    out = _bilinear(np.asarray(plan.fit_plate(plate), dtype=np.float32), plan.x, plan.y)
    out[~plan.use] = 0.0
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def render_with_lens(render_fn, cam, backplate: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """`render_fn(pinhole_camera, backplate) -> (rgb uint8, mask bool)` is a renderer backend. `backplate` is the pinhole picture of the
    background for the canvas of `canvas_size(cam)`. Returns the whole image (background and officer distorted together) and the instance
    mask as the lens camera `cam` sees them (cam.width x cam.height); pixels outside the image circle are black and not in the mask."""
    plan = _Plan(cam)
    rgb, mask = render_fn(plan.cam, plan.fit_plate(backplate))
    out = _bilinear(rgb.astype(np.float32), plan.x, plan.y)
    out[~plan.use] = 0.0
    m = _bilinear(mask.astype(np.float32)[..., None], plan.x, plan.y)[..., 0] >= 0.5
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), m & plan.use


def layer_with_lens(layer_fn, cam) -> np.ndarray:
    """`layer_fn(pinhole_camera) -> RGBA uint8` (colour + coverage). Returns the same layer as the lens camera sees it."""
    plan = _Plan(cam)
    rgba = layer_fn(plan.cam).astype(np.float32)
    a = rgba[..., 3:4] / 255.0
    pm = np.concatenate([rgba[..., :3] * a, a], axis=-1)                       # premultiplied, so that edges do not bleed
    s = _bilinear(pm, plan.x, plan.y)
    s[~plan.use] = 0.0
    alpha = s[..., 3:4]
    out = np.zeros(s.shape, np.float32)
    out[..., :3] = np.where(alpha > 1e-6, s[..., :3] / np.maximum(alpha, 1e-6), 0.0)
    out[..., 3:4] = alpha * 255.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
