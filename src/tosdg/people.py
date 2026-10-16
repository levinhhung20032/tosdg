"""Reference people in a backplate: how tall a person is in the photo where they stand, to size the officer from it.

People (standing, or sitting on a motorbike / bicycle) and vehicles of known typical height tell the camera that took the
photo (calibrate.py, size_reference "auto"). Detected once per photo when it is imported
(`tosdg backgrounds import`, `tosdg backgrounds people`, Studio upload) and stored in the library index:
    "people": [{"kind": "standing" | "rider", "feet": [u, v], "height_px": h, "bbox": [x0, y0, x1, y1], "score": s}, ...]
in photo pixels; "feet" is the ground point (ankles / wheels), "height_px" from there to the top of the head.
The same detector run also gives the vehicles without a rider (car, bus, truck, motorbike, bicycle), stored as "objects"
(same fields, "feet" = bottom of the box): calibrate.py uses them when the photo has too few people.

Detection: Ultralytics YOLO11 (COCO): the detector finds people, bicycles and motorbikes; the pose model's keypoints tell
whether a person stands (hips over knees over visible ankles, legs long against the torso). A person over a bike is a
rider: height from the head to the bottom of the wheels. People and vehicles cut by the picture edge, too small, uncertain,
or (vehicles) partly hidden behind a nearer detection are left out. Needs `ultralytics` (+ torch); weights are downloaded once into <repo>/data/models.

Contents:
  constants: MODELS, MIN_HEIGHT_PX, MIN_SCORE, VEHICLES, MAX_HIDDEN
  class PeopleDetectorUnavailable
  detect_people — Reference people (standing people and riders) and vehicles of a photo
  (+ 3 private helpers)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .schema import repo_root

MODELS = ("yolo11m.pt", "yolo11m-pose.pt")
MIN_HEIGHT_PX = 40           # smaller people are too coarse to measure
MIN_SCORE = 0.45
VEHICLES = {2: "car", 5: "bus", 7: "truck", 3: "motorbike", 1: "bicycle"}     # COCO class -> calibrate.PRIORS kind
MAX_HIDDEN = 0.25            # a vehicle whose box is covered more than this by a nearer detection is not measured
_KP = {"l_sh": 5, "r_sh": 6, "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16}


class PeopleDetectorUnavailable(RuntimeError):
    pass


_MODELS: dict = {}


def _models():
    if not _MODELS:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise PeopleDetectorUnavailable(f"person detection needs ultralytics (pip install ultralytics): {e}") from None
        wdir = repo_root() / "data" / "models"
        wdir.mkdir(parents=True, exist_ok=True)
        _MODELS["det"], _MODELS["pose"] = (YOLO(str(wdir / name)) for name in MODELS)   # downloaded on first use
    return _MODELS["det"], _MODELS["pose"]


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _standing(kp: np.ndarray) -> tuple[bool, float | None]:
    """(stands, feet row) from COCO keypoints (17 x 3: x, y, confidence)."""
    ok = kp[:, 2] > 0.5
    k = {n: kp[i, :2] for n, i in _KP.items() if ok[i]}
    if not all(n in k for n in ("l_hip", "r_hip", "l_ank", "r_ank")) or not ("l_sh" in k or "r_sh" in k):
        return False, None                           # ankles must be seen: the feet are the ground point
    hip = (k["l_hip"] + k["r_hip"]) / 2
    ank = (k["l_ank"] + k["r_ank"]) / 2
    sh = np.mean([k[n] for n in ("l_sh", "r_sh") if n in k], axis=0)
    leg, torso = ank[1] - hip[1], hip[1] - sh[1]
    if torso <= 0 or leg < 1.3 * torso:              # sitting / crouching: short legs in the picture
        return False, None
    if abs(ank[0] - hip[0]) > 0.35 * leg:            # leaning / walking far apart: not a clean height
        return False, None
    for side in ("l", "r"):                          # knees between hips and ankles
        if f"{side}_knee" in k and not (hip[1] < k[f"{side}_knee"][1] < ank[1]):
            return False, None
    return True, float(max(k["l_ank"][1], k["r_ank"][1]))


def _covered(b, others) -> float:
    """Fraction of box b covered by the nearer boxes (bottom lower in the picture) of `others`."""
    area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    worst = 0.0
    for o in others:
        if o is b or o[3] <= b[3]:
            continue
        x0, y0, x1, y1 = max(b[0], o[0]), max(b[1], o[1]), min(b[2], o[2]), min(b[3], o[3])
        worst = max(worst, max(0.0, x1 - x0) * max(0.0, y1 - y0) / area)
    return worst


def _bike_of(b, boxes, bikes) -> int | None:
    """Index (in boxes) of the bike (bicycle / motorbike, indices `bikes`) a person box b sits on, or None."""
    x0, y0, x1, y1 = (float(v) for v in b)
    return next((k for k in bikes if (m := boxes[k])[0] <= (x0 + x1) / 2 <= m[2] and m[1] < y1 <= m[3] + 0.1 * (m[3] - m[1])
                 and m[3] > y1 - 0.05 * (y1 - y0)), None)


def detect_people(img: Image.Image) -> tuple[list[dict], list[dict], str]:
    """(reference people, vehicles without a rider, method string for the index) of a photo (photo pixels), best first."""
    det, pose = _models()
    rgb = img.convert("RGB")
    W, H = rgb.size
    rd = det(rgb, verbose=False, imgsz=1280, conf=MIN_SCORE, classes=[0, *VEHICLES])[0]
    rp = pose(rgb, verbose=False, imgsz=1280, conf=MIN_SCORE)[0]
    boxes, cls, conf = rd.boxes.xyxy.cpu().numpy(), rd.boxes.cls.cpu().numpy().astype(int), rd.boxes.conf.cpu().numpy()
    persons = [(b, s) for b, c, s in zip(boxes, cls, conf) if c == 0]
    bikes = [k for k, c in enumerate(cls) if c in (1, 3)]
    ridden = {_bike_of(b, boxes, bikes) for b, _ in persons} - {None}                 # bikes with someone on them
    pboxes, pkps = rp.boxes.xyxy.cpu().numpy(), rp.keypoints.data.cpu().numpy()
    edge = 2
    out = []
    for b, s in persons:
        x0, y0, x1, y1 = (float(v) for v in b)
        if x0 <= edge or y0 <= edge or x1 >= W - edge:
            continue                                 # cut by the picture edge: the height is unknown
        k = _bike_of(b, boxes, bikes)
        rider = None if k is None else boxes[k]
        if rider is not None:
            if rider[3] >= H - edge:
                continue
            feet = [float((rider[0] + rider[2]) / 2), float(rider[3])]
            entry = {"kind": "rider", "feet": feet, "height_px": float(rider[3] - y0),
                     "bbox": [min(x0, float(rider[0])), y0, max(x1, float(rider[2])), float(rider[3])]}
        else:
            if y1 >= H - edge:
                continue
            j = max(range(len(pboxes)), key=lambda i: _iou(b, pboxes[i]), default=None)
            if j is None or _iou(b, pboxes[j]) < 0.6:
                continue
            stands, feet_v = _standing(pkps[j])
            if not stands:
                continue
            feet_v = max(feet_v, y1 - 0.04 * (y1 - y0))          # the box bottom (shoe soles), not the ankle joint
            entry = {"kind": "standing", "feet": [(x0 + x1) / 2, feet_v], "height_px": feet_v - y0, "bbox": [x0, y0, x1, y1]}
        if entry["height_px"] < MIN_HEIGHT_PX:
            continue
        entry = {**entry, "score": float(s)}
        entry["feet"] = [round(v, 1) for v in entry["feet"]]
        entry["bbox"] = [round(v, 1) for v in entry["bbox"]]
        entry["height_px"] = round(entry["height_px"], 1)
        entry["score"] = round(entry["score"], 3)
        out.append(entry)
    out.sort(key=lambda e: -e["score"])
    objects = []
    for k, (b, c, s) in enumerate(zip(boxes, cls, conf)):
        if c not in VEHICLES or k in ridden:
            continue
        x0, y0, x1, y1 = (float(v) for v in b)
        if x0 <= edge or y0 <= edge or x1 >= W - edge or y1 >= H - edge or y1 - y0 < MIN_HEIGHT_PX:
            continue
        if _covered(b, boxes) > MAX_HIDDEN:
            continue                                 # the nearer one may hide its wheels or roof
        objects.append({"kind": VEHICLES[int(c)], "feet": [round((x0 + x1) / 2, 1), round(y1, 1)], "height_px": round(y1 - y0, 1),
                        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "score": round(float(s), 3)})
    objects.sort(key=lambda e: -e["score"])
    return out, objects, f"yolo11:{'+'.join(MODELS)}:standing+rider+vehicles"
