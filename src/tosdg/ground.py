"""Standable ground of a backplate: where the officer's feet may be in the photo (road, sidewalk), and the check that keeps
a sample's officer off walls, buildings, cars and flowerbeds.

Aligning the horizon only fixes the vertical placement; the random horizontal crop can still put the feet on the front of a
house, so every backplate may carry a ground mask (a PNG next to the image, listed in the library index with its sha256):
  - real photos: `detect_ground` - semantic segmentation (SegFormer trained on Cityscapes; classes road + sidewalk), run once
    when the photo is imported (`tosdg backgrounds import`, `tosdg backgrounds ground`, Studio "add to library"). It needs
    torch + transformers, which `generate` does not: generation only reads the stored mask;
  - fixtures: the exact mask (everything below the drawn horizon);
  - any library: a hand-made mask (`tosdg backgrounds ground --mask`).

At generation time (`fit_feet`) the ground points under the feet (or the podium) are projected into the composed backplate.
If one lands off the ground, the backplate crop is moved sideways to a position where all of them are on the ground, drawn
from the RNG stream "ground"; when there is none the sample is rejected and resampled (PlacementError). Points outside the
picture are not judged. A background without a mask is not checked (as before).

Contents:
  constants: DEFAULT_MODEL, GROUND_CLASSES, WORK_WIDTH, ERODE_PX, MAX_MASK_WIDTH
  class GroundDetectorUnavailable
  class PlacementError
  detect_ground — Standable-ground mask of a photo (uint8 0/255)…
  mask_below_horizon — Exact mask of a synthetic plate: every row from the horizon down
  save_mask / load_mask
  footprint_points — World points on the ground under the feet (or the podium rim)
  fit_feet — Check / move the crop so the footprint is on the ground…
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
GROUND_CLASSES = ("road", "sidewalk")        # Cityscapes labels the officer may stand on (not terrain: grass verges, flowerbeds)
WORK_WIDTH = 1024                            # the photo is segmented at this width (the model was trained on 1024 px crops)
ERODE_PX = 3                                 # the mask is shrunk by this much (at WORK_WIDTH) so feet do not touch a wall's foot
MAX_MASK_WIDTH = 1024                        # stored masks are at most this wide (same aspect ratio as the photo)
FOOT_JOINTS = ("foot_l", "ball_l", "toe_l", "foot_r", "ball_r", "toe_r")
PODIUM_RIM_POINTS = 12


class GroundDetectorUnavailable(RuntimeError):
    pass


class PlacementError(ValueError):
    """The officer's feet cannot be put on the standable ground of the chosen backplate (the sample is resampled)."""


_MODELS: dict[str, tuple] = {}


def _load_model(model_id: str):
    if model_id not in _MODELS:
        try:
            import torch  # noqa: F401
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as e:
            raise GroundDetectorUnavailable("ground detection needs torch and transformers (pip install torch transformers); "
                                            "or give a mask by hand with 'tosdg backgrounds ground --mask'") from e
        try:                                     # the cached copy first: no network access when the model is already there
            proc = SegformerImageProcessor.from_pretrained(model_id, local_files_only=True)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id, local_files_only=True)
        except OSError:
            proc = SegformerImageProcessor.from_pretrained(model_id)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        _MODELS[model_id] = (proc, model.eval())
    return _MODELS[model_id]


def detect_ground(img: Image.Image, horizon_y_frac: float | None = None, model_id: str = DEFAULT_MODEL) -> tuple[np.ndarray, str]:
    """Standable-ground mask of a photo (uint8 0/255, at most MAX_MASK_WIDTH wide) and a method string for the index.
    Rows above the horizon (when known) are never ground. Raises GroundDetectorUnavailable without torch / transformers."""
    import torch
    proc, model = _load_model(model_id)
    w = WORK_WIDTH
    h = max(1, round(img.height * w / img.width))
    im = img.convert("RGB").resize((w, h), Image.BILINEAR)
    labels = {v: int(k) for k, v in model.config.id2label.items()}
    keep = [labels[c] for c in GROUND_CLASSES]
    with torch.no_grad():
        logits = model(**proc(images=im, return_tensors="pt", do_resize=False)).logits
        lab = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0]
        g = torch.isin(lab, torch.tensor(keep)).float()[None, None]
        if ERODE_PX:                              # erosion = min filter = -maxpool(-x)
            g = -torch.nn.functional.max_pool2d(-g, 2 * ERODE_PX + 1, stride=1, padding=ERODE_PX)
    mask = (g[0, 0].numpy() > 0.5)
    if horizon_y_frac is not None:
        mask[: int(np.ceil(horizon_y_frac * h))] = False
    return (mask.astype(np.uint8) * 255), f"segformer:{model_id}:{'+'.join(GROUND_CLASSES)}"


def mask_below_horizon(width: int, height: int, horizon_y_frac: float) -> np.ndarray:
    """Exact mask of a synthetic plate whose ground fills everything from the horizon row down."""
    w = min(width, MAX_MASK_WIDTH)
    h = max(1, round(height * w / width))
    m = np.zeros((h, w), np.uint8)
    m[int(np.ceil(horizon_y_frac * h)):] = 255
    return m


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    m = Image.fromarray(np.where(np.asarray(mask) >= 128, 255, 0).astype(np.uint8), "L")
    if m.width > MAX_MASK_WIDTH:
        m = m.resize((MAX_MASK_WIDTH, max(1, round(m.height * MAX_MASK_WIDTH / m.width))), Image.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m.save(path, optimize=True)


def load_mask(path: str | Path) -> np.ndarray:
    """Bool array (h, w): True where the officer may stand."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) >= 128


def footprint_points(fk_world, officer_position, podium: dict | None) -> np.ndarray:
    """World points on the ground (z = 0) that must be on standable ground: under the feet, or around the podium's base."""
    if podium:
        r = podium["diameter_cm"] / 2.0
        a = np.linspace(0.0, 2 * np.pi, PODIUM_RIM_POINTS, endpoint=False)
        pts = np.stack([officer_position[0] + r * np.cos(a), officer_position[1] + r * np.sin(a), np.zeros_like(a)], axis=1)
        return np.vstack([[officer_position[0], officer_position[1], 0.0], pts])
    pts = np.array([fk_world.positions[j] for j in FOOT_JOINTS if j in fk_world.positions], dtype=float)
    pts[:, 2] = 0.0
    return pts


def fit_feet(mask: np.ndarray, plate_size: tuple[int, int], composition: dict, feet_uv: np.ndarray, width: int, height: int,
             rng_factory) -> tuple[dict, dict]:
    """Check that the footprint (`feet_uv`: pixels in the composed canvas of `width` x `height`) lies on the ground mask of a
    backplate of `plate_size` composed with `composition` (compose_backplate). Returns (composition, report): the
    composition gets a new horizontal offset when the drawn one puts a foot off the ground; `rng_factory()` gives the
    RNG (stream "ground") and is only called then. Raises PlacementError when no horizontal offset works."""
    sw, sh = scaled_size(plate_size, composition["scale"], width, height)
    ox, oy = composition["offset_px"]
    uv = np.asarray(feet_uv, dtype=float).reshape(-1, 2)
    vis = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[vis]
    if not len(uv):
        return composition, {"status": "not_visible", "points": 0}
    mh, mw = mask.shape
    rows = np.clip(((uv[:, 1] - oy) * mh / sh).astype(int), 0, mh - 1)

    def on_ground(oxs: np.ndarray) -> np.ndarray:          # (len(oxs),) all visible points on the ground for each offset
        cols = np.clip(((uv[None, :, 0] - oxs[:, None]) * mw / sw).astype(int), 0, mw - 1)
        return mask[rows[None, :], cols].all(axis=1)

    if on_ground(np.array([ox]))[0]:
        return composition, {"status": "ok", "points": int(len(uv))}
    cands = np.arange(-(sw - width), 1)
    good = cands[on_ground(cands)]
    if not len(good):
        raise PlacementError(f"the officer's feet are off the road/sidewalk at every horizontal crop ({len(uv)} visible ground points)")
    new_ox = int(good[int(rng_factory().integers(len(good)))])
    return {**composition, "offset_px": [new_ox, oy]}, {"status": "shifted", "points": int(len(uv))}


def scaled_size(plate_size: tuple[int, int], scale: float, width: int, height: int) -> tuple[int, int]:
    """Size of the scaled backplate before cropping (as compose_backplate makes it)."""
    bw, bh = plate_size
    return max(width, round(bw * scale)), max(height, round(bh * scale))
