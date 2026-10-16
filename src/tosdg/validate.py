"""Sample and dataset validation. A sample that fails any check is never emitted.

Contents:
  constants: SIGNAL_ARM_KPS, ID_RE
  check_render
  check_ground_truth
  check_metadata
  validate_dataset — Re-check a written dataset from disk
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .backgrounds import sha256_file
from .schema import schema_errors

SIGNAL_ARM_KPS = {"l": ("left_shoulder", "left_elbow", "left_wrist"), "r": ("right_shoulder", "right_elbow", "right_wrist")}
ID_RE = re.compile(r"^[A-Za-z0-9_\-]+_\d{6}$")


def check_render(rgb: np.ndarray, mask: np.ndarray, width: int, height: int, min_std: float,
                 officer_levels: tuple[float, float] = (0.0, 255.0)) -> list[str]:
    """officer_levels: (min, max) mean sRGB level of the officer's pixels; outside = too dark / blown out to learn from."""
    errs = []
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        errs.append(f"render: rgb shape/dtype {rgb.shape}/{rgb.dtype} != ({height},{width},3)/uint8")
    elif float(rgb.std()) < min_std:
        errs.append(f"render: image nearly uniform (std={rgb.std():.2f})")
    if mask.shape != (height, width):
        errs.append(f"render: mask shape {mask.shape}")
    elif not mask.any():
        errs.append("render: empty instance mask (subject not visible)")
    elif rgb.shape[:2] == mask.shape:
        level = float((rgb[mask].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])).mean())
        if not officer_levels[0] <= level <= officer_levels[1]:
            errs.append(f"render: officer mean level {level:.1f} outside [{officer_levels[0]:g}, {officer_levels[1]:g}] "
                        "(too dark or blown out)")
    return errs


def check_ground_truth(meta: dict, rules: dict, sides: list[str]) -> list[str]:
    errs = []
    W, H = meta["image"]["width"], meta["image"]["height"]
    off = meta["officer"]
    bb = off.get("bbox")
    if bb is None:
        return ["bbox: none (empty mask)"]
    x0, y0, x1, y1 = bb["xyxy"]
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        errs.append(f"bbox: invalid {bb['xyxy']}")
    if (y1 - y0) < rules["min_bbox_height_px"]:
        errs.append(f"bbox: height {y1 - y0}px < {rules['min_bbox_height_px']}")
    if bb["truncated"] and not rules["max_bbox_edge_touch"]:
        errs.append("bbox: subject truncated by image border")

    kp = off["keypoints_2d"]["coco17"]
    if any(d <= 0 for d in kp["depth_cm"]):
        errs.append("keypoints: point at or behind camera")
    frac = sum(kp["in_frame"]) / len(kp["in_frame"])
    if frac < rules["min_keypoints_in_frame_frac"]:
        errs.append(f"keypoints: only {frac:.0%} in frame")
    tol = 2.0  # px; keypoints lie inside the rendered capsules, so must lie in the mask bbox
    for name, uv, inside in zip(kp["names"], kp["uv"], kp["in_frame"]):
        if inside and not (x0 - tol <= uv[0] <= x1 + tol and y0 - tol <= uv[1] <= y1 + tol):
            errs.append(f"keypoints: {name} {uv} outside bbox {bb['xyxy']} (projection/raster mismatch)")
    if rules["require_signal_arm_in_frame"]:
        idx = {n: i for i, n in enumerate(kp["names"])}
        for s in sides:
            if not all(kp["in_frame"][idx[n]] for n in SIGNAL_ARM_KPS[s]):
                errs.append(f"keypoints: signalling arm '{s}' not fully in frame")
    return errs


def check_metadata(meta: dict) -> list[str]:
    errs = [f"metadata: {e}" for e in schema_errors(meta, "sample_metadata")]
    sid = meta.get("sample_id", "")
    if not ID_RE.match(sid):
        errs.append(f"metadata: bad sample_id '{sid}'")
    for k, f in meta.get("files", {}).items():
        if not f.startswith(sid + "."):
            errs.append(f"metadata: file '{f}' does not match sample id")
    return errs


def validate_dataset(out_dir: str | Path) -> list[str]:
    """Re-check a written dataset from disk: manifest, metadata, files, hashes, images."""
    out = Path(out_dir)
    errs = []
    man_path = out / "manifest.json"
    if not man_path.exists():
        return [f"{man_path}: missing"]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    errs += [f"manifest: {e}" for e in schema_errors(man, "manifest")]
    if errs:
        return errs
    listed = set()
    for s in man["samples"]:
        sid = s["sample_id"]
        meta_path = out / s["metadata"]
        listed.add(s["metadata"])
        if not meta_path.exists():
            errs.append(f"{sid}: metadata file missing")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        errs += [f"{sid}: {e}" for e in check_metadata(meta)]
        if meta.get("seed", {}).get("sample_seed") != s["sample_seed"]:
            errs.append(f"{sid}: seed mismatch manifest vs metadata")
        if sha256_file(meta_path) != s["sha256"][s["metadata"]]:
            errs.append(f"{sid}: metadata sha256 mismatch")
        W, H = meta["image"]["width"], meta["image"]["height"]
        for kind, fname in meta["files"].items():
            p = out / fname
            listed.add(fname)
            if not p.exists():
                errs.append(f"{sid}: {kind} file missing ({fname})")
                continue
            if s["sha256"].get(fname) != sha256_file(p):
                errs.append(f"{sid}: {kind} sha256 mismatch")
            try:
                with Image.open(p) as im:
                    im.load()
                    if im.size != (W, H):
                        errs.append(f"{sid}: {kind} size {im.size} != {(W, H)}")
                    if kind == "mask":
                        area = int((np.asarray(im) > 0).sum())
                        if area != meta["officer"]["bbox"]["mask_area_px"]:
                            errs.append(f"{sid}: mask area {area} != metadata {meta['officer']['bbox']['mask_area_px']}")
            except Exception as e:  # corrupted / truncated image
                errs.append(f"{sid}: {kind} unreadable ({e})")
    extra = {p.name for p in out.iterdir() if p.is_file()} - listed - {"manifest.json"}
    if extra:
        errs.append(f"unlisted files in dataset dir: {sorted(extra)[:10]}")
    ids = [s["sample_id"] for s in man["samples"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate sample ids")
    return errs
