"""2D backplate library: import, tagging, seeded selection, camera-aware composition.

Library layout:
    <library>/index.json          (schema: background_library)
    <library>/images/<id>.<ext>
    <library>/images/<id>.ground.png   optional standable-ground mask (see ground.py)
    index "people" / "objects" of a background: reference people / vehicles and barriers found in the photo (people.py),
    from which calibrate.py tells the height and horizon of the camera that took it

Contents:
  constants: INDEX_VERSION, MAX_COVER_SCALE, PHOTO_CAMERA_HEIGHT_CM, DASHCAM_CAMERA_HEIGHT_CM
  sha256_file
  class BackgroundLibrary
  compose_backplate — Scale/crop a backplate to (width, height) so that its…
  plan_backplate — The scale and crop offset compose_backplate uses (no pixels)
  crop_backplate — The (width, height) picture of a planned composition
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import imageio
from . import ground as groundmod
from .horizon import detect_horizon
from .schema import load_json, validate_instance

INDEX_VERSION = "1.2"                       # 1.1: optional "ground" mask per background; 1.2: "people", "road_yaw_deg"
MAX_COVER_SCALE = 4.0
PHOTO_CAMERA_HEIGHT_CM = 155.0              # a street photo taken at eye level (default camera_height_cm)
DASHCAM_CAMERA_HEIGHT_CM = 125.0            # ... with the tag "dashcam": behind a car's windscreen


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BackgroundLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        if self.index_path.exists():
            self.index = load_json(self.index_path, "background_library")
        else:
            self.index = {"schema_version": INDEX_VERSION, "backgrounds": []}
        for b in self.index["backgrounds"]:          # libraries written on Windows before 2026-10 used "\"; the next save writes "/"
            b["file"] = b["file"].replace("\\", "/")
            b.pop("scene", None)                     # traffic light / signs of libraries before 2026-10-16: no longer used (D44)
        self.by_id = {b["id"]: b for b in self.index["backgrounds"]}
        self._masks: dict[str, np.ndarray] = {}          # ground masks by sha256 (generation reads them for every sample)

    def save(self) -> None:
        self.index["backgrounds"].sort(key=lambda b: b["id"])
        self.index["schema_version"] = INDEX_VERSION
        validate_instance(self.index, "background_library")
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.index, indent=1) + "\n", encoding="utf-8")

    def add(self, image_path: str | Path, tags: list[str], category: str = "uncategorized",
            bg_id: str | None = None, horizon_y_frac: float | None = None,
            ground: str | np.ndarray | None = None) -> dict:
        """ground: None = no ground mask; "detect" = run the detector (ground.py; needs torch + transformers);
        an array = this mask (uint8/bool, any size with the photo's aspect ratio)."""
        if horizon_y_frac is not None and not 0.0 <= float(horizon_y_frac) <= 1.0:
            raise ValueError(f"horizon must be a fraction of the image height in [0, 1], got {horizon_y_frac}")
        src = Path(image_path)
        data = src.read_bytes()
        try:
            with Image.open(io.BytesIO(data)) as im:
                keep = im.format in ("PNG", "JPEG")
        except Image.UnidentifiedImageError:
            raise ValueError(f"{src}: not an image Pillow can read") from None                               # these are copied as they are; WebP / BMP / GIF / TIFF become JPG
        if keep:
            with Image.open(src) as im:
                im.verify()
            with Image.open(src) as im:
                w, h = im.size
            ext = ".jpg" if src.suffix.lower() == ".jpeg" else src.suffix.lower()
        else:
            data, ext, w, h = imageio.normalise(data)
        bg_id = bg_id or re.sub(r"[^A-Za-z0-9_\-]", "_", src.stem)
        if bg_id in self.by_id:
            raise ValueError(f"background id '{bg_id}' already in library")
        dst = self.root / "images" / f"{bg_id}{ext}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if keep:
            shutil.copyfile(src, dst)
        else:
            dst.write_bytes(data)
        manual = horizon_y_frac is not None
        if horizon_y_frac is None:
            with Image.open(dst) as im:
                horizon_y_frac = detect_horizon(im)
        entry = {"id": bg_id, "file": dst.relative_to(self.root).as_posix(), "sha256": sha256_file(dst),
                 "width": w, "height": h, "category": category, "tags": sorted(set(tags))}
        if horizon_y_frac is not None:
            entry["horizon_y_frac"] = float(horizon_y_frac)
            entry["horizon_source"] = "manual" if manual else "detected"
        self.index["backgrounds"].append(entry)
        self.by_id[bg_id] = entry
        if ground is not None:
            try:
                self.set_ground(bg_id, ground)
            except Exception:
                self.index["backgrounds"].remove(entry)
                del self.by_id[bg_id]
                dst.unlink(missing_ok=True)
                raise
        return entry

    def set_ground(self, bg_id: str, ground: str | np.ndarray, method: str = "manual") -> dict:
        """Give a background its standable-ground mask: "detect" runs the detector, an array is used as it is.
        Raises ground.GroundDetectorUnavailable when detection is asked for without torch / transformers."""
        entry = self.by_id[bg_id]
        if isinstance(ground, str):
            if ground != "detect":
                raise ValueError(f"ground must be 'detect' or a mask array, got {ground!r}")
            with Image.open(self.path(entry)) as im:
                mask, method = groundmod.detect_ground(im, entry.get("horizon_y_frac"))
        else:
            mask = np.asarray(ground)
            mask = np.where(mask.astype(np.float64) >= (128 if mask.dtype != bool else 1), 255, 0).astype(np.uint8)
            ar_mask, ar_img = mask.shape[1] / mask.shape[0], entry["width"] / entry["height"]
            if abs(ar_mask / ar_img - 1) > 0.02:
                raise ValueError(f"{bg_id}: ground mask is {mask.shape[1]}x{mask.shape[0]}, not the shape of the photo "
                                 f"({entry['width']}x{entry['height']})")
        dst = self.root / "images" / f"{bg_id}.ground.png"
        groundmod.save_mask(mask, dst)
        entry["ground"] = {"file": dst.relative_to(self.root).as_posix(), "sha256": sha256_file(dst), "method": method,
                           "standable_frac": round(float(groundmod.load_mask(dst).mean()), 4)}
        return entry

    def remove(self, bg_id: str) -> dict:
        """Take a background out of the library: its entry, its image and its ground mask (files inside the library only).
        Call save() afterwards. Raises KeyError for an unknown id."""
        entry = self.by_id.pop(bg_id)
        self.index["backgrounds"].remove(entry)
        for rel in (entry["file"], (entry.get("ground") or {}).get("file")):
            if rel:
                p = (self.root / rel.replace("\\", "/")).resolve()
                if self.root.resolve() in p.parents:              # never delete outside the library, whatever the index says
                    p.unlink(missing_ok=True)
        return entry

    def set_people(self, bg_id: str, people: str | list = "detect") -> dict:
        """Reference people of a photo (people.py): "detect" runs the detector and stores the people and the vehicles
        ("objects", kept with the barriers marked by hand); a list is stored as the people as it is.
        Raises people.PeopleDetectorUnavailable when detection is asked for without ultralytics."""
        from . import people as peoplemod
        entry = self.by_id[bg_id]
        if isinstance(people, str):
            if people != "detect":
                raise ValueError(f"people must be 'detect' or a list, got {people!r}")
            with Image.open(self.path(entry)) as im:
                found, vehicles, method = peoplemod.detect_people(im)
            hand = [o for o in (entry.get("objects") or {}).get("found", []) if o.get("manual")]
            entry["objects"] = {"method": method, "found": vehicles + hand}
        else:
            found, method = list(people), "manual"
        entry["people"] = {"method": method, "found": found}
        return entry

    def set_junction(self, bg_id: str, junction: dict | None) -> dict:
        """The road layout of a photo (junction.py; photo pixels), or None to remove it (a straight road)."""
        from .junction import validate
        entry = self.by_id[bg_id]
        if junction is None:
            entry.pop("junction", None)
            return entry
        validate(junction, entry["width"], entry["height"])
        entry["junction"] = {"type": junction["type"], "center": [round(float(x), 1) for x in junction["center"]],
                             "arms_deg": [round(float(a), 1) for a in junction["arms_deg"]]}
        return entry

    def set_objects(self, bg_id: str, objects: list[dict]) -> dict:
        """Vehicles / barriers of known kind (calibrate.PRIORS) in a photo, stored as they are (replaces the list)."""
        entry = self.by_id[bg_id]
        entry["objects"] = {"method": (entry.get("objects") or {}).get("method", "manual"), "found": list(objects)}
        return entry

    @staticmethod
    def calibration(entry: dict) -> dict:
        """The camera that took the photo (calibrate.calibrate): from its people, else its objects, else its horizon."""
        from .calibrate import calibrate
        return calibrate(entry, BackgroundLibrary.camera_height_cm(entry))

    @staticmethod
    def horizon_reliable(entry: dict) -> bool:
        """A horizon set by hand (not the automatic estimate)."""
        return entry.get("horizon_y_frac") is not None and entry.get("horizon_source") == "manual"

    @staticmethod
    def camera_height_cm(entry: dict) -> float:
        """Height of the camera that took the photo (camera_height_cm, else a default from the tags)."""
        if entry.get("camera_height_cm"):
            return float(entry["camera_height_cm"])
        return DASHCAM_CAMERA_HEIGHT_CM if "dashcam" in entry.get("tags", []) else PHOTO_CAMERA_HEIGHT_CM

    def clear_people(self, bg_id: str) -> None:
        """Forget the people and the detected vehicles (barriers marked by hand stay)."""
        e = self.by_id[bg_id]
        e.pop("people", None)
        hand = [o for o in (e.get("objects") or {}).get("found", []) if o.get("manual")]
        if hand:
            e["objects"] = {"method": "manual", "found": hand}
        else:
            e.pop("objects", None)

    def clear_ground(self, bg_id: str) -> None:
        entry = self.by_id[bg_id]
        g = entry.pop("ground", None)
        if g:
            (self.root / g["file"]).unlink(missing_ok=True)

    def ground_mask(self, entry: dict) -> np.ndarray | None:
        """The background's standable-ground mask (bool, True = may stand), or None when it has none."""
        g = entry.get("ground")
        if not g:
            return None
        if g["sha256"] not in self._masks:
            self._masks[g["sha256"]] = groundmod.load_mask(self.root / g["file"])
        return self._masks[g["sha256"]]

    def path(self, entry: dict) -> Path:
        return self.root / entry["file"].replace("\\", "/")

    def absent(self) -> list[str]:
        """Ids of the photos the index lists that are not on this machine (the photos are not in git: fetch_backgrounds.bat)."""
        return [b["id"] for b in self.index["backgrounds"] if not self.path(b).exists()]

    def integrity_errors(self, missing_photos: bool = True) -> list[str]:
        """missing_photos False: a listed photo that is not on this machine is not an error (it is optional; see absent)."""
        errs = []
        for b in self.index["backgrounds"]:
            p = self.path(b)
            if not p.exists():
                if missing_photos:
                    errs.append(f"{b['id']}: missing file {p} (not downloaded: fetch_backgrounds.bat)")
                continue
            elif sha256_file(p) != b["sha256"]:
                errs.append(f"{b['id']}: sha256 mismatch")
            g = b.get("ground")
            if g:
                gp = self.root / g["file"]
                if not gp.exists():
                    errs.append(f"{b['id']}: missing ground mask {gp}")
                elif sha256_file(gp) != g["sha256"]:
                    errs.append(f"{b['id']}: ground mask sha256 mismatch (re-run 'tosdg backgrounds ground')")
        return errs

    def select(self, rng: np.random.Generator, tags_any: list[str] | None = None) -> dict:
        cands = sorted(self.index["backgrounds"], key=lambda b: b["id"])
        if tags_any:
            tagged = [b for b in cands if set(b["tags"]) & set(tags_any)]
            cands = tagged or cands  # fall back rather than fail; recorded in metadata
        if not cands:
            raise ValueError("background library is empty")
        return cands[int(rng.integers(len(cands)))]


def compose_backplate(img: Image.Image, entry: dict, camera_horizon_v: float, width: int, height: int,
                      rng: np.random.Generator, best_effort: bool = False) -> tuple[Image.Image, dict]:
    """Scale/crop a backplate to (width, height) so that its horizon row lines up
    with the camera's horizon row where possible (2D, roll ignored).
    best_effort (Studio preview only): when the camera horizon is outside the frame or needs more than MAX_COVER_SCALE,
    still shift/zoom as far as allowed towards it instead of giving up and cropping at random; `generate` leaves it off."""
    composition = plan_backplate(img.size, entry, camera_horizon_v, width, height, rng, best_effort)
    return crop_backplate(img, composition, width, height), composition


def plan_backplate(plate_size: tuple[int, int], entry: dict, camera_horizon_v: float, width: int, height: int,
                   rng: np.random.Generator, best_effort: bool = False) -> dict:
    """The scale and crop offset that compose_backplate uses (same RNG draws), without touching pixels."""
    bw, bh = plate_size
    hf = entry.get("horizon_y_frac")
    scale = max(width / bw, height / bh)
    aligned = False
    if hf is not None and 0 < hf < 1:
        vh = float(camera_horizon_v)
        need_h = max(height, vh / hf if vh > 0 else 0, (height - vh) / (1 - hf) if vh < height else 0)
        s = max(scale, need_h / bh)
        if best_effort:
            scale, aligned = min(s, scale * MAX_COVER_SCALE), True
        elif s <= scale * MAX_COVER_SCALE and 0 <= vh <= height:
            scale, aligned = s, True
    sw, sh = max(width, round(bw * scale)), max(height, round(bh * scale))
    if aligned:
        oy = int(round(np.clip(vh - hf * sh, height - sh, 0)))
    else:
        oy = -int(rng.integers(0, sh - height + 1))
    ox = -int(rng.integers(0, sw - width + 1))
    return {"scale": scale, "offset_px": [ox, oy], "horizon_aligned": aligned, "camera_horizon_v": float(camera_horizon_v)}


def crop_backplate(img: Image.Image, composition: dict, width: int, height: int) -> Image.Image:
    """The (width, height) picture of a composition from plan_backplate."""
    sw, sh = groundmod.scaled_size(img.size, composition["scale"], width, height)
    ox, oy = composition["offset_px"]
    return img.convert("RGB").resize((sw, sh), Image.BILINEAR).crop((-ox, -oy, -ox + width, -oy + height))
