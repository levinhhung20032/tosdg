"""Export a generated dataset to the formats training and labelling tools read: COCO keypoints, YOLO-pose, CVAT.

Every format gets the same 59 keypoints per officer (COCO 17 body, then 21 left hand, then 21 right hand, as in
COCO-WholeBody), the mask-based bbox and the gesture. A keypoint outside the picture is "not labelled" (visibility 0);
one inside is 2 (self-occlusion is not modelled, see ARCHITECTURE "Ground truth"). Images are copied as <sample_id>.<ext>.
The train / val / test split is a seeded permutation of the samples (stream "export_split" of the dataset's master seed
unless a seed is given), so the same dataset and ratios always give the same split.

Every format also gets labels.csv (one row per image: the officer's signal, if any, and what it means per side).

Layouts (in the output dir):
  coco:  images/<split>/<id>.png, annotations/person_keypoints_<split>.json (+ RLE mask, "gesture" per annotation)
  yolo:  images/<split>/<id>.png, labels/<split>/<id>.txt, data.yaml (kpt_shape [59, 3], flip_idx)
  cvat:  images/<id>.png, annotations.xml ("CVAT for images 1.1": box + mask + three skeletons, subset = split),
         cvat_labels.json (paste into the task's label constructor, "Raw", before uploading the annotations)

Contents:
  constants: FORMATS, SPLITS, BODY, HAND, KEYPOINTS, EDGES, FLIP_IDX, _SKELETONS
  split_samples — sample ids -> split name
  keypoints_of — 59 x (x, y, v) of one sample's metadata
  mask_rle_coco / mask_rle_cvat — mask run lengths in each tool's convention
  write_csv / csv_row — labels.csv: the officer's signal per image
  export — write one format
  (+ 9 private helpers)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from PIL import Image

from . import rng as rngmod
from .schema import load_json, repo_root
from .validate import validate_dataset

FORMATS = ("coco", "yolo", "cvat")
SPLITS = ("train", "val", "test")
BODY = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
        "right_ankle"]
HAND = ["wrist"] + [f"{f}_{i}" for f in ("thumb", "index", "middle", "ring", "pinky") for i in range(1, 5)]
KEYPOINTS = BODY + [f"left_hand_{n}" for n in HAND] + [f"right_hand_{n}" for n in HAND]
_BODY_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
               (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]          # COCO's person skeleton (0-based)
_HAND_EDGES = [(0, 1 + 4 * f) for f in range(5)] + [(1 + 4 * f + i, 2 + 4 * f + i) for f in range(5) for i in range(3)]
EDGES = _BODY_EDGES + [(17 + a, 17 + b) for a, b in _HAND_EDGES] + [(38 + a, 38 + b) for a, b in _HAND_EDGES] + \
        [(9, 17), (10, 38)]                                                              # body wrist -> hand wrist
_BODY_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
FLIP_IDX = _BODY_FLIP + list(range(38, 59)) + list(range(17, 38))                        # mirror image: left <-> right


def split_samples(ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    """Seeded split: ids sorted, permuted by stream(seed, "export_split"), cut by the ratios (normalised)."""
    r = np.asarray(ratios, float)
    if len(r) != 3 or (r < 0).any() or r.sum() <= 0:
        raise ValueError(f"split must be three non-negative ratios train,val,test with a positive sum, got {list(ratios)}")
    order = sorted(ids)
    perm = rngmod.stream(seed, "export_split").permutation(len(order))
    n = len(order)
    n_train = int(round(r[0] / r.sum() * n))
    n_val = min(n - n_train, int(round(r[1] / r.sum() * n)))
    out = {}
    for k, i in enumerate(perm):
        out[order[i]] = "train" if k < n_train else "val" if k < n_train + n_val else "test"
    return out


def keypoints_of(meta: dict) -> np.ndarray:
    """(59, 3): pixel x, y and COCO visibility: 2 = visible, 1 = in the picture but hidden by the officer's own body
    (metadata "occluded"), 0 = outside the picture (then x = y = 0)."""
    kp = meta["officer"]["keypoints_2d"]
    rows = []
    for name in ("coco17", "hand21_l", "hand21_r"):
        s = kp[name]
        occ = s.get("occluded") or [False] * len(s["uv"])               # metadata before 1.8 has no occlusion
        for uv, inside, hidden in zip(s["uv"], s["in_frame"], occ):
            rows.append((uv[0], uv[1], 1 if hidden else 2) if (uv and inside) else (0.0, 0.0, 0))
    return np.asarray(rows, float)


def mask_rle_coco(mask: np.ndarray) -> list[int]:
    """COCO uncompressed RLE: run lengths over the column-major (Fortran) flattened mask, starting with background."""
    return _runs(np.asarray(mask, bool).flatten(order="F"))


def mask_rle_cvat(mask: np.ndarray, box: list[int]) -> list[int]:
    """CVAT mask RLE: run lengths over the row-major pixels of the box [x0, y0, x1, y1), starting with background."""
    x0, y0, x1, y1 = box
    return _runs(np.asarray(mask, bool)[y0:y1, x0:x1].flatten())


def _runs(flat: np.ndarray) -> list[int]:
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    return ([0] + runs) if flat.size and flat[0] else runs


def _load(dataset: Path) -> tuple[dict, list[dict]]:
    errs = validate_dataset(dataset)
    if errs:
        raise ValueError(f"{dataset} does not pass 'tosdg validate' ({len(errs)} error(s), first: {errs[0]}); export refused")
    man = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    metas = [json.loads((dataset / s["metadata"]).read_text(encoding="utf-8")) for s in man["samples"]]
    return man, metas


def _gestures(profile: str) -> list[dict]:
    root = repo_root() / "profiles" / profile
    files = load_json(root / "profile.json", "profile")["files"]
    return load_json(root / files["gestures"], "gestures")["gestures"]


def _gesture_ids(profile: str) -> list[str]:
    """The profile's gesture ids in file order: a stable class list, the same for every split and every dataset."""
    return [g["id"] for g in _gestures(profile)]


CSV_COLUMNS = ["image", "split", "signalling", "gesture_id", "gesture_label", "hieu_lenh", "instruction_front", "instruction_back",
               "instruction_left", "instruction_right", "observer_side", "instruction_for_observer"]


def write_csv(out: Path, rows: list[dict]) -> Path:
    """labels.csv: one row per image - which signal the officer gives (if any) and what it means for each side and for the
    camera's road user. UTF-8 with BOM so Excel shows the Vietnamese text."""
    import csv
    path = out / "labels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image"]))
    return path


def csv_row(meta: dict, image: str, split: str, gestures: dict) -> dict:
    g = meta["officer"]["gesture"]
    spec = gestures.get(g["id"], {})
    tc = meta["traffic_control"]
    mine = tc if "instruction_for_observer" in tc else next(               # datasets before 1.8 kept it in a signal list
        (x for x in tc.get("signals", []) if x.get("type") == "officer_gesture"), {})
    instr = spec.get("instructions", {})
    return {"image": image, "split": split, "signalling": "yes" if g.get("signalling", True) else "no", "gesture_id": g["id"],
            "gesture_label": g["label"], "hieu_lenh": spec.get("description", "") if g.get("signalling", True) else "không ra hiệu lệnh",
            **{f"instruction_{k}": instr.get(k, "") for k in ("front", "back", "left", "right")},
            "observer_side": tc.get("approach_side") or "", "instruction_for_observer": mine.get("instruction_for_observer") or ""}


def _copy_image(dataset: Path, meta: dict, dest_dir: Path) -> str:
    src = dataset / meta["files"]["rgb"]
    name = f"{meta['sample_id']}{src.suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest_dir / name)
    return name


def export(dataset: str | Path, out: str | Path, fmt: str, ratios=(0.8, 0.1, 0.1), seed: int | None = None,
           yolo_classes: str = "officer") -> dict:
    """Write `dataset` (a generated, valid dataset dir) to `out` (new or empty) in format `fmt`. Returns a summary."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {list(FORMATS)}, not '{fmt}'")
    if yolo_classes not in ("officer", "gesture"):
        raise ValueError(f"yolo classes must be 'officer' or 'gesture', not '{yolo_classes}'")
    dataset, out = Path(dataset), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; export into a new or empty directory")
    man, metas = _load(dataset)
    if not metas:
        raise ValueError(f"{dataset} has no samples")
    seed = int(man["config"]["master_seed"] if seed is None else seed)
    split = split_samples([m["sample_id"] for m in metas], tuple(ratios), seed)
    gestures = _gesture_ids(man["config"]["profile"])
    out.mkdir(parents=True, exist_ok=True)
    {"coco": _coco, "yolo": _yolo, "cvat": _cvat}[fmt](dataset, out, metas, split, gestures, yolo_classes)
    specs = {g["id"]: g for g in _gestures(man["config"]["profile"])}
    rows = []
    for m in metas:
        name = m["sample_id"] + Path(m["files"]["rgb"]).suffix
        image = f"images/{name}" if fmt == "cvat" else f"images/{split[m['sample_id']]}/{name}"
        rows.append(csv_row(m, image, split[m["sample_id"]], specs))
    write_csv(out, rows)
    counts = {s: sum(1 for v in split.values() if v == s) for s in SPLITS}
    summary = {"format": fmt, "dataset": str(dataset), "split_seed": seed, "ratios": list(ratios), "counts": counts,
               "keypoints": KEYPOINTS, "gestures": gestures}
    (out / "export.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _mask(dataset: Path, meta: dict) -> np.ndarray:
    return np.asarray(Image.open(dataset / meta["files"]["mask"])) > 0


def _coco(dataset, out, metas, split, gestures, _):
    cat = {"id": 1, "name": "traffic_officer", "supercategory": "person", "keypoints": KEYPOINTS,
           "skeleton": [[a + 1, b + 1] for a, b in EDGES]}                              # COCO skeleton is 1-based
    for s in SPLITS:
        items = [m for m in metas if split[m["sample_id"]] == s]
        if not items:
            continue
        images, anns = [], []
        for i, m in enumerate(items, 1):
            name = _copy_image(dataset, m, out / "images" / s)
            W, H = m["image"]["width"], m["image"]["height"]
            images.append({"id": i, "file_name": name, "width": W, "height": H})
            kp = keypoints_of(m)
            b = m["officer"]["bbox"]
            g = m["officer"]["gesture"]
            anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [float(v) for v in b["xywh"]], "area": int(b["mask_area_px"]),
                         "segmentation": {"size": [H, W], "counts": mask_rle_coco(_mask(dataset, m))},
                         "keypoints": [round(float(v), 3) for v in kp.flatten()], "num_keypoints": int((kp[:, 2] > 0).sum()),
                         "gesture": g["id"], "gesture_index": gestures.index(g["id"]), "signalling": g["signalling"],
                         "truncated": b["truncated"]})
        doc = {"info": {"description": "tosdg synthetic traffic officer dataset", "source": str(dataset.name)},
               "licenses": [], "images": images, "annotations": anns, "categories": [cat],
               "gestures": [{"index": i, "id": g} for i, g in enumerate(gestures)]}
        (out / "annotations").mkdir(parents=True, exist_ok=True)
        (out / "annotations" / f"person_keypoints_{s}.json").write_text(json.dumps(doc), encoding="utf-8")


def _yolo(dataset, out, metas, split, gestures, classes):
    for m in metas:
        s = split[m["sample_id"]]
        name = _copy_image(dataset, m, out / "images" / s)
        W, H = m["image"]["width"], m["image"]["height"]
        x0, y0, x1, y1 = m["officer"]["bbox"]["xyxy"]
        kp = keypoints_of(m)
        cls = 0 if classes == "officer" else gestures.index(m["officer"]["gesture"]["id"])
        vals = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
        for x, y, v in kp:
            vals += [x / W, y / H, v]
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s / (Path(name).stem + ".txt")).write_text(
            f"{cls} " + " ".join(f"{v:.6f}" if i < 4 or (i - 4) % 3 != 2 else str(int(v)) for i, v in enumerate(vals)) + "\n",
            encoding="utf-8")
    names = ["traffic_officer"] if classes == "officer" else gestures
    present = [s for s in SPLITS if any(v == s for v in split.values())]
    lines = ["# tosdg export (YOLO-pose). Keypoints: COCO 17 body, 21 left hand, 21 right hand.",
             "# path is absolute (Ultralytics resolves a relative one against its own datasets dir); edit it after moving the folder.",
             f"path: {out.resolve().as_posix()}",
             *[f"{s}: images/{s}" for s in present], f"kpt_shape: [{len(KEYPOINTS)}, 3]",
             f"flip_idx: {FLIP_IDX}", "names:", *[f"  {i}: {n}" for i, n in enumerate(names)]]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


_SKELETONS = (("officer_body", BODY, 0, _BODY_EDGES), ("left_hand", HAND, 17, _HAND_EDGES), ("right_hand", HAND, 38, _HAND_EDGES))


def _svg(names: list[str], edges) -> str:
    """CVAT skeleton drawing: nodes on a circle (any layout works; CVAT only needs the nodes and the edges)."""
    parts = []
    for a, b in edges:
        parts.append(f'<line x1="0" y1="0" x2="0" y2="0" data-type="edge" data-node-from="{a + 1}" data-node-to="{b + 1}"></line>')
    for i, n in enumerate(names):
        t = 2 * np.pi * i / len(names)
        parts.append(f'<circle r="0.75" cx="{50 + 40 * np.cos(t):.2f}" cy="{50 + 40 * np.sin(t):.2f}" data-type="element node" '
                     f'data-element-id="{i + 1}" data-node-id="{i + 1}" data-label-name="{n}"></circle>')
    return "".join(parts)


def _cvat(dataset, out, metas, split, gestures, _):
    labels = [{"name": "traffic_officer", "type": "any", "attributes": [
        {"name": "gesture", "mutable": False, "input_type": "select", "default_value": gestures[0], "values": gestures}]}]
    labels += [{"name": n, "type": "skeleton", "attributes": [], "svg": _svg(names, e),
                "sublabels": [{"name": k, "type": "points", "attributes": []} for k in names]} for n, names, _o, e in _SKELETONS]
    (out / "cvat_labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    xml = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for i, m in enumerate(sorted(metas, key=lambda m: m["sample_id"])):
        name = _copy_image(dataset, m, out / "images")
        W, H = m["image"]["width"], m["image"]["height"]
        x0, y0, x1, y1 = m["officer"]["bbox"]["xyxy"]
        g = m["officer"]["gesture"]["id"]
        kp = keypoints_of(m)
        xml.append(f'  <image id="{i}" name={quoteattr(name)} subset="{split[m["sample_id"]]}" width="{W}" height="{H}">')
        attr = f'<attribute name="gesture">{g}</attribute>'
        xml.append(f'    <box label="traffic_officer" source="auto" occluded="0" xtl="{x0}" ytl="{y0}" xbr="{x1}" ybr="{y1}" '
                   f'z_order="0">{attr}</box>')
        rle = ", ".join(map(str, mask_rle_cvat(_mask(dataset, m), [x0, y0, x1, y1])))
        xml.append(f'    <mask label="traffic_officer" source="auto" occluded="0" rle="{rle}" left="{x0}" top="{y0}" '
                   f'width="{x1 - x0}" height="{y1 - y0}" z_order="0">{attr}</mask>')
        for label, names, off, _e in _SKELETONS:
            xml.append(f'    <skeleton label="{label}" source="auto" z_order="0">')
            for k, n in enumerate(names):
                x, y, v = kp[off + k]
                xml.append(f'      <points label="{n}" source="auto" outside="{0 if v else 1}" occluded="{1 if v == 1 else 0}" points="{x:.2f},{y:.2f}"></points>')
            xml.append("    </skeleton>")
        xml.append("  </image>")
    xml.append("</annotations>")
    (out / "annotations.xml").write_text("\n".join(xml) + "\n", encoding="utf-8")
