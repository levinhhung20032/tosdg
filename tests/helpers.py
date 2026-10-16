"""Shared test helpers (paths, skeleton / pose loaders, config builders).

Contents:
  skeleton()
  poses()
  vn_gestures()
  fixture_library()
  make_config()
"""
import functools
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tosdg.fixtures import build_fixture_library  # noqa: E402
from tosdg.poses import PoseLibrary  # noqa: E402
from tosdg.schema import load_json  # noqa: E402
from tosdg.skeleton import Skeleton  # noqa: E402
from tosdg.gestures import GestureLibrary  # noqa: E402


@functools.lru_cache(maxsize=None)
def skeleton(character: str = "proxy_v1") -> Skeleton:
    """proxy_v1 (the generic test character and the legacy models' skeleton) or tripo_v1 (the active Tripo model's)."""
    return Skeleton.load(ROOT / "characters" / character / "skeleton.json")


@functools.lru_cache(maxsize=None)
def poses() -> PoseLibrary:
    return PoseLibrary(ROOT / "poses")


@functools.lru_cache(maxsize=None)
def vn_gestures() -> GestureLibrary:
    return GestureLibrary(load_json(ROOT / "profiles" / "VN" / "gestures.json", "gestures"), poses())


_LIB = None


def fixture_library() -> Path:
    """Session-wide fixture backplate library (small images for speed)."""
    global _LIB
    if _LIB is None:
        _LIB = Path(tempfile.mkdtemp(prefix="tosdg_bg_"))
        build_fixture_library(_LIB, 480, 270)
    return _LIB


def make_config(tmp: Path, **over) -> Path:
    cfg = json.loads((ROOT / "configs" / "smoke.json").read_text())
    cfg["background_library"] = str(fixture_library())
    cfg["resolution"] = [320, 180]
    cfg.update(over)
    p = Path(tmp) / "config.json"
    p.write_text(json.dumps(cfg))
    return p
