"""Register a user-supplied 3D model of a person in uniform as the profile's active uniform.

`register_model` is the one entry point behind `tosdg import-model` and the Studio upload:
  1. keeps the source GLB in profiles/<CC>/models/<id>.glb;
  2. writes profiles/<CC>/assets/<id>.fit.json (only if absent, so hand-edited fits survive a re-upload);
  3. imports it onto the skeleton: the model's own skeleton/skin when it has a usable one, else bone-proximity rigging (import_mesh.import_glb);
  4. writes profiles/<CC>/assets/<id>.glb and makes it the profile's only sampled uniform (weight 1, listed first, others weight 0).

Contents:
  constants: MAX_MODEL_BYTES, DEFAULT_PROXY_COLORS, DEFAULT_FIT
  model_id — A uniform id from a file name or label
  register_model — Import `src` (a GLB of a person wearing the uniform) and…
  list_models — Every saved model of the profile
  activate_model — Make a saved model the active uniform
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .import_mesh import import_glb, orient_outward, read_static_glb, skeleton_usable
from .schema import validate_instance
from .skeleton import Skeleton
from .uniform_mesh import check_against_skeleton, read_glb, spec_sha256, write_glb

MAX_MODEL_BYTES = 200 * 1024 * 1024
_ID = re.compile(r"[^A-Za-z0-9_\-]+")
# colours of the capsule (proxy) backend and of the regions a textured model does not paint; a model's own texture is never recoloured
DEFAULT_PROXY_COLORS = {
    "torso": [211, 175, 143], "upper_arm": [211, 175, 143], "forearm": [211, 175, 143], "hand": [189, 146, 123], "finger": [189, 146, 123],
    "neck": [189, 146, 123], "head": [189, 146, 123], "headwear": [211, 175, 143], "pelvis": [211, 175, 143], "thigh": [211, 175, 143],
    "shin": [211, 175, 143], "foot": [24, 22, 24], "red": [181, 35, 26], "leather": [36, 28, 25], "metal": [230, 170, 60],
}
UNIFORMS_VERSION = "1.3"       # 1.3: no baton-free variants (variant_of / baton_model removed): the baton is a prop, held or on the belt
DEFAULT_FIT = {"height_cm": 178.0, "skin_hue_deg": [0.0, 32.0], "skin_saturation": [0.12, 0.6]}


def model_id(name: str) -> str:
    """A uniform id from a file name or label: letters, digits, _ and - only."""
    stem = Path(name).stem if name.lower().endswith(".glb") else name
    mid = _ID.sub("_", stem).strip("_")[:48]
    return mid or "model"


def register_model(root: Path, profile: str, src: str | Path, name: str | None = None, character: str = "tripo_v1",
                   profile_dir: Path | None = None, activate: bool = True) -> dict:
    """Import `src` (a GLB of a person wearing the uniform) and make it the active uniform of `profile` (`activate`; otherwise it is
    only registered and keeps its place and weight). The model must not carry a baton of its own: the baton is a prop the renderers add
    (in the hand, or on the belt at the place the model's fit `belt_baton` gives). Returns a summary."""
    src = Path(src)
    data = src.read_bytes()
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError(f"{src.name}: model larger than {MAX_MODEL_BYTES // 2**20} MB")
    mid = model_id(name or src.name)
    pdir = Path(profile_dir) if profile_dir else Path(root) / "profiles" / profile
    sk = Skeleton.load(Path(root) / "characters" / character / "skeleton.json")
    g = read_static_glb(src)                                    # raises ValueError for a file that is not a usable GLB 2.0
    if g["texture"] is None:
        raise ValueError(f"{src.name}: the model needs a base-colour texture (a person in uniform, textured)")
    digest = hashlib.sha256(data).hexdigest()

    (pdir / "models").mkdir(exist_ok=True)
    (pdir / "assets").mkdir(exist_ok=True)
    model_path = pdir / "models" / f"{mid}.glb"
    if not model_path.exists() or model_path.read_bytes() != data:
        model_path.write_bytes(data)
    fit_rel, asset_rel = f"assets/{mid}.fit.json", f"assets/{mid}.glb"
    fit_path = pdir / fit_rel
    fit = json.loads(fit_path.read_text(encoding="utf-8")) if fit_path.exists() else dict(DEFAULT_FIT)
    fit["source"] = {"name": f"models/{mid}.glb", "sha256": digest}
    fit_path.write_text(json.dumps(fit, indent=1) + "\n", encoding="utf-8")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))

    ok, unmapped = skeleton_usable(g, sk, fit)
    mesh = import_glb(src, sk, fit, {"uniform": mid, "fit_sha256": spec_sha256(fit), "source_sha256": digest})
    flipped = orient_outward(mesh)
    write_glb(mesh, sk, pdir / asset_rel)
    errs = check_against_skeleton(read_glb(pdir / asset_rel), sk)
    if errs:
        raise ValueError(f"{src.name}: imported mesh does not fit the skeleton: {errs}")

    upath = pdir / "uniforms.json"
    doc = json.loads(upath.read_text(encoding="utf-8")) if upath.exists() else {"schema_version": UNIFORMS_VERSION, "uniforms": []}
    old = next((u for u in doc["uniforms"] if u["id"] == mid), None)
    entry = {
        "id": mid, "weight": 1.0, "verified": False,
        "description": f"User-supplied 3D model of a person in uniform ({src.name}); the model's own texture is the appearance. "
                       f"Skin: {'the model\'s own skeleton and weights' if ok else 'rigged to the skeleton by bone proximity'}. Origin/licence of the model are the uploader's responsibility.",
        "proxy_colors": (old or {}).get("proxy_colors") or dict(DEFAULT_PROXY_COLORS),
        "mesh": {"asset": asset_rel, "fit": fit_rel},
    }
    doc["schema_version"] = UNIFORMS_VERSION
    if activate:
        others = [{**u, "weight": 0.0} for u in doc["uniforms"] if u["id"] != mid]
        doc["uniforms"] = [entry] + others
    else:
        entry["weight"] = (old or {}).get("weight", 0.0)
        doc["uniforms"] = [entry if u["id"] == mid else u for u in doc["uniforms"]] if old else [*doc["uniforms"], entry]
    validate_instance(doc, "uniforms")
    upath.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"id": mid, "skin": mesh.info.get("skin", "rigged"), "unmapped_joints": unmapped, "vertices": mesh.n_vertices,
            "triangles": mesh.n_triangles, "winding_flipped": flipped, "bind_pose": mesh.info.get("bind_pose", {}),
            "bind_direction_error_deg": mesh.info.get("bind_direction_error_deg"), "model": f"models/{mid}.glb", "asset": asset_rel}


def list_models(profile_dir: Path) -> list[dict]:
    """Every saved model of the profile: the source GLBs in models/ plus the uniforms already registered. `active` = the one
    `tosdg generate` samples (first uniform with weight > 0)."""
    pdir = Path(profile_dir)
    upath = pdir / "uniforms.json"
    uniforms = json.loads(upath.read_text(encoding="utf-8"))["uniforms"] if upath.exists() else []
    by_id = {u["id"]: u for u in uniforms}
    first = next((u["id"] for u in uniforms if u.get("weight", 0) > 0), None)
    ids = list(by_id) + sorted(p.stem for p in (pdir / "models").glob("*.glb") if p.stem not in by_id)
    out = []
    for mid in ids:
        u = by_id.get(mid)
        out.append({"id": mid, "registered": u is not None, "has_mesh": bool(u and "mesh" in u), "active": mid == first,
                    "source": (pdir / "models" / f"{mid}.glb").exists()})
    return out


def activate_model(root: Path, profile: str, mid: str, character: str = "tripo_v1", profile_dir: Path | None = None) -> dict:
    """Make a saved model the active uniform. Already registered with an up-to-date asset -> just reorder the weights; otherwise
    (re)import it from profiles/<CC>/models/<id>.glb with its stored fit."""
    pdir = Path(profile_dir) if profile_dir else Path(root) / "profiles" / profile
    mid = model_id(mid)
    src = pdir / "models" / f"{mid}.glb"
    upath = pdir / "uniforms.json"
    doc = json.loads(upath.read_text(encoding="utf-8"))
    entry = next((u for u in doc["uniforms"] if u["id"] == mid), None)
    if entry is not None and (("mesh" not in entry) or (pdir / entry["mesh"]["asset"]).exists()):
        doc["uniforms"] = [{**entry, "weight": 1.0}] + [{**u, "weight": 0.0} for u in doc["uniforms"] if u["id"] != mid]
        validate_instance(doc, "uniforms")
        upath.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"id": mid, "reused": True}
    if not src.exists():
        raise ValueError(f"no saved model '{mid}' (expected {src.name} in models/)")
    return {**register_model(root, profile, src, mid, character, profile_dir=pdir), "reused": False}
