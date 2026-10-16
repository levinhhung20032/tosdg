"""Studio workspace on disk (default <repo>/data/studio, git-ignored): UI state and exported scenes. Background photos are NOT kept here: the Studio reads and writes the generator's background library
(default <repo>/data/backgrounds), so the photos shown in the Studio are exactly the ones `generate` draws from. A workspace
from before that change (its own backgrounds/ folder) is moved into the library once, on start.

Uploads are verified with Pillow and stored under server-chosen names with an extension derived
from the *detected* format - never from the client's file name.

Contents:
  constants: MAX_UPLOAD_BYTES
  class WorkspaceError
  slug
  prepare_image — Validate, upright, strip metadata, store as PNG or JPG
  class Workspace
  (+ 2 private helpers)
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from .. import imageio
from ..backgrounds import BackgroundLibrary, sha256_file
from ..ground import GroundDetectorUnavailable
from ..people import PeopleDetectorUnavailable

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
_ID_RE = re.compile(r"[^A-Za-z0-9_\-]+")





class WorkspaceError(ValueError):
    pass


def slug(text: str, fallback: str = "item") -> str:
    s = _ID_RE.sub("_", Path(str(text)).stem).strip("_")[:48]
    return s or fallback


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def prepare_image(data: bytes) -> tuple[bytes, str, int, int]:
    """Validate, upright, strip metadata, store as PNG or JPG (see imageio)."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise WorkspaceError(f"image must be 1 byte .. {MAX_UPLOAD_BYTES // (1 << 20)} MB")
    try:
        return imageio.normalise(data)
    except ValueError as e:
        raise WorkspaceError(str(e)) from None


class Workspace:
    def __init__(self, root: str | Path, library: str | Path | None = None):
        """library: the generator's background library the Studio's photos live in (default <root>/library)."""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lib_dir = Path(library).resolve() if library else self.root / "library"
        self.scene_dir = self.root / "scenes"
        self.migrated = self._migrate_old_backgrounds()

    # -- safe file serving ----------------------------------------------------------
    def resolve_bg(self, rel: str) -> Path | None:
        """A file of the background library (for serving thumbnails), never outside it."""
        p = (self.lib_dir / rel).resolve()
        return p if p.is_file() and self.lib_dir in p.parents else None

    # -- backgrounds = the generator's library ------------------------------------------
    def _lib(self) -> BackgroundLibrary:
        return BackgroundLibrary(self.lib_dir)

    def list_backgrounds(self) -> list[dict]:
        """The library's photos that are on this machine (listed but not downloaded ones are optional: not shown)."""
        lib = self._lib()
        return [b for b in lib.index["backgrounds"] if lib.path(b).exists()]

    def get_background(self, bg_id: str) -> dict | None:
        return self._lib().by_id.get(bg_id)

    def add_background(self, name: str, data: bytes, detect_ground: bool = True) -> dict:
        """Upload a photo into the library: checked and re-encoded (prepare_image), horizon detected, tags day + ground_level,
        and (when torch + transformers are installed) its standable-ground mask detected."""
        data, ext, _w, _h = prepare_image(data)
        lib = self._lib()
        base, n = slug(name, "background"), 1
        bg_id = base
        while bg_id in lib.by_id:
            n += 1
            bg_id = f"{base}_{n}"
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / f"{bg_id}{ext}"
            src.write_bytes(data)
            entry = lib.add(src, ["day", "ground_level"], "studio", bg_id=bg_id)
        if detect_ground:
            try:                                           # where the officer may stand in this photo (ground.py); a few seconds
                lib.set_ground(bg_id, "detect")
            except GroundDetectorUnavailable:
                pass                                       # no mask: the generator does not check the feet on this photo
            try:                                           # people / vehicles: the photo's camera (people.py, calibrate.py)
                lib.set_people(bg_id, "detect")
            except PeopleDetectorUnavailable:
                pass
        lib.save()
        return entry

    def delete_background(self, bg_id: str) -> None:
        """Remove a photo from the library (entry, image, ground mask): `generate` no longer uses it."""
        lib = self._lib()
        if bg_id not in lib.by_id:
            raise WorkspaceError(f"unknown background '{bg_id}'")
        lib.remove(bg_id)
        lib.save()

    def update_background(self, bg_id: str, **fields) -> dict:
        """horizon_y_frac (number in (0, 1) or None), tags (list), road_yaw_deg (direction of the traffic relative to the view, -180..180, None = along the view), camera_height_cm
        (height of the camera that took the photo, None = default)."""
        lib = self._lib()
        entry = lib.by_id.get(bg_id)
        if entry is None:
            raise WorkspaceError(f"unknown background '{bg_id}'")
        if "horizon_y_frac" in fields:
            h = fields["horizon_y_frac"]
            if h is not None and not (isinstance(h, (int, float)) and 0 < h < 1):
                raise WorkspaceError("horizon_y_frac must be a number in (0, 1) or null")
            if h is None:
                entry.pop("horizon_y_frac", None)
                entry.pop("horizon_source", None)
            else:
                entry["horizon_y_frac"] = float(h)
                entry["horizon_source"] = "manual"            # placed by hand in the Studio: reliable (size_reference auto)
        if "tags" in fields:
            entry["tags"] = sorted({slug(t, "") for t in fields["tags"] if slug(t, "")})
        if "camera_height_cm" in fields:
            c = fields["camera_height_cm"]
            if c is not None and not (isinstance(c, (int, float)) and 30 <= c <= 1000):
                raise WorkspaceError("camera_height_cm must be a number in 30..1000 or null")
            if c is None:
                entry.pop("camera_height_cm", None)
            else:
                entry["camera_height_cm"] = float(c)
        if "road_yaw_deg" in fields:
            r = fields["road_yaw_deg"]
            if r is not None and not (isinstance(r, (int, float)) and -180 <= r <= 180):
                raise WorkspaceError("road_yaw_deg must be a number in -180..180 or null")
            if r is None or r == 0:
                entry.pop("road_yaw_deg", None)
            else:
                entry["road_yaw_deg"] = float(r)
        lib.save()
        return entry

    def _migrate_old_backgrounds(self) -> list[str]:
        """Move the photos of an old workspace (<root>/backgrounds, its own index) into the library, then remove that folder.
        A photo already in the library (same id, and same bytes or same size) is not copied again; same id, another photo
        gets a new id."""
        old_dir = self.root / "backgrounds"
        old = _read_json(old_dir / "index.json", None)
        if old is None or old_dir.resolve() == self.lib_dir:
            return []
        lib, moved = self._lib(), []
        for b in old.get("backgrounds", []):
            src = (self.root / b.get("file", "")).resolve()
            if not src.is_file() or self.root not in src.parents:
                continue
            same = lib.by_id.get(b["id"])
            if same and (same.get("sha256") == sha256_file(src) or
                         (same.get("width"), same.get("height")) == (b.get("width"), b.get("height"))):
                continue                                   # the same photo (an upload re-encodes it, so the bytes may differ)
            bg_id, n = b["id"], 1
            while bg_id in lib.by_id:
                n += 1
                bg_id = f"{b['id']}_{n}"
            lib.add(src, b.get("tags") or ["day", "ground_level"], "studio", bg_id=bg_id, horizon_y_frac=b.get("horizon_y_frac"))
            moved.append(bg_id)
        lib.save()
        shutil.rmtree(old_dir)
        return moved

    # -- ui state / scenes -------------------------------------------------------------
    def load_state(self) -> dict:
        return _read_json(self.root / "state.json", {})

    def save_state(self, state: dict) -> None:
        _write_json(self.root / "state.json", state)

    def save_scene(self, name: str, scene: dict) -> Path:
        p = self.scene_dir / f"{slug(name, 'scene')}.json"
        _write_json(p, scene)
        return p
