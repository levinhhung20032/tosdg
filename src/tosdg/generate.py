"""Batch generation: seed -> scene -> render -> ground truth -> validate -> write.

Per-sample files (in the output dir):
    <id>.rgb.png    rendered image (or <id>.rgb.jpg when the config says image_format "jpg")
    <id>.mask.png   officer instance mask (source of the bbox)
    <id>.json       metadata (schema: sample_metadata)
plus manifest.json (schema: manifest) with hashes of every file.

A sample that fails validation is resampled with a derived attempt seed; if all
attempts fail it is recorded under `failures` in the manifest and nothing is
written for it. The run then reports failure (non-zero exit from the CLI).

Contents:
  constants: METADATA_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION
  class SampleResult
  render_scene
  generate_sample
  write_sample
  prepare_output_dir — Create the output dir, or refuse to mix a new dataset into an old one
  portable_path / resolve_portable_path — Manifest paths without user names, "/" on every OS
  run
  (+ 2 private helpers)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import groundtruth as gt
from . import imageio
from . import rng as rngmod
from .backgrounds import sha256_file
from .gestures import PoseSamplingError
from .ground import PlacementError
from .lens import render_with_lens
from . import baton as batonmod
from . import podium as podiummod
from . import shadows as shadowmod
from .schema import repo_root
from .scene import GENERATOR_VERSION, GenerationContext, SizeReferenceError, build_scene
from .validate import check_ground_truth, check_metadata, check_render

METADATA_SCHEMA_VERSION = "1.9"        # 1.8: background.size_reference, camera.placement.road_quarter; 1.9: lighting.shadow
MANIFEST_SCHEMA_VERSION = "1.1"        # 1.1: background_library_resolved is repo-relative (or absolute outside the repo), "/" separators


@dataclass
class SampleResult:
    ok: bool
    meta: dict | None = None
    rgb: np.ndarray | None = None
    mask: np.ndarray | None = None
    errors: list[str] = field(default_factory=list)


def _r(x, nd=4):
    return round(float(x), nd)


def render_scene(ctx: GenerationContext, scene):
    """(rgb, mask) as the scene camera sees them: the backend draws a pinhole picture, a camera lens then distorts it (lens.py)."""
    cam = scene.camera
    if cam.lens is None or not cam.lens.active:
        return _render_pinhole(ctx, scene, cam, scene.backplate)
    return render_with_lens(lambda pin_cam, plate: _render_pinhole(ctx, scene, pin_cam, plate), cam, scene.backplate)


def _render_pinhole(ctx: GenerationContext, scene, camera, backplate):
    backend = ctx.config["backend"]
    if backend == "proxy":
        from .render_proxy import render
        colors = ctx.uniforms[scene.uniform_id]["proxy_colors"]
        return render(ctx.skeleton, scene.fk_world, camera, backplate, colors, scene.lighting,
                      extra_capsules=gt.baton_capsules(ctx.skeleton, scene))
    if backend == "mesh":
        from .mesh_renderer import get as get_mesh_renderer
        render_mesh = get_mesh_renderer(ctx.config.get("mesh_renderer", "cpu"))[0]
        mesh = ctx.meshes[scene.uniform_id]
        mesh = batonmod.mesh_with_officer_baton(mesh, scene.baton, scene.baton["mode"] if scene.baton else "across", prop=ctx.baton_prop)
        if scene.podium:
            mesh = podiummod.mesh_with_podium(mesh, ctx.podium_prop, scene.podium["sole_z_cm"])
        if scene.shadow and scene.shadow.get("enabled"):          # the officer's shadow on the photo, then the officer over it
            backplate = cast_shadow(mesh, scene.fk_world, camera, backplate, scene.shadow)
        return render_mesh(mesh, scene.fk_world, camera, backplate, ctx.uniforms[scene.uniform_id]["proxy_colors"],
                           scene.lighting, scene.uniform_brightness, ground_shadow=ctx.shadow is None)
    raise NotImplementedError(f"backend '{backend}' is not implemented yet (see docs/ARCHITECTURE.md)")


def cast_shadow(mesh, fk_world, camera, backplate, shadow: dict):
    """The backplate (pinhole canvas of `camera`) darkened by the posed mesh's shadow (shadows.py)."""
    from .render_mesh import skin_mesh
    pos, _ = skin_mesh(mesh, fk_world)
    sun = shadowmod.sun_direction(shadow.get("sun_azimuth_deg", 0.0), shadow.get("sun_elevation_deg", 90.0))
    cov = shadowmod.cast(pos, mesh.triangles, sun, camera, shadow["softness"], shadow["contact"], bool(shadow.get("sun")))
    return shadowmod.apply(backplate, cov, shadow["strength"])


def generate_sample(ctx: GenerationContext, index: int, sample_seed: int, attempt: int) -> SampleResult:
    cfg = ctx.config
    seed = rngmod.derive_attempt_seed(sample_seed, attempt)
    try:
        scene = build_scene(ctx, seed)
    except PoseSamplingError as e:
        return SampleResult(False, errors=[f"pose: {e}"])
    except SizeReferenceError as e:
        return SampleResult(False, errors=[f"size_reference: {e}"])
    except PlacementError as e:
        return SampleResult(False, errors=[f"ground: {e}"])
    rgb, mask = render_scene(ctx, scene)
    W, H = cfg["resolution"]
    errs = check_render(rgb, mask, W, H, ctx.validation["min_image_std"],
                        (ctx.validation["min_officer_level"], ctx.validation["max_officer_level"]))
    if errs:
        return SampleResult(False, errors=errs)

    sid = f"{cfg['name']}_{index:06d}"
    image_format = cfg.get("image_format", "png")                       # the rgb image only; masks are always lossless PNG
    sp, g, prof = scene.sampled_pose, scene.gesture, ctx.profile
    meta = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "sample_id": sid,
        "index": index,
        "seed": {"master_seed": cfg["master_seed"], "sample_seed": sample_seed, "attempt": attempt, "attempt_seed": seed},
        "generator": {"name": "tosdg", "version": GENERATOR_VERSION, "backend": cfg["backend"]},
        "config": {"name": cfg["name"], "sha256": ctx.config_sha256},
        "assets_sha256": ctx.assets_sha256,
        "files": {"rgb": f"{sid}.rgb.{image_format}", "mask": f"{sid}.mask.png"},
        "image": {"width": W, "height": H, "format": image_format},
        "profile": {"country": prof["country"], "authority": prof["authority"],
                    "regulation_id": prof["regulation"]["id"], "regulation_verified": prof["regulation"].get("verified", False)},
        "officer": {
            "instance_id": 1,
            "character": ctx.skeleton.id,
            "uniform": scene.uniform_id,
            "uniform_brightness": _r(scene.uniform_brightness, 6),
            "world_position": [_r(x) for x in scene.officer_position],
            "world_yaw_deg": _r(scene.officer_yaw_deg),
            "gesture": {"id": g["id"], "label": g["label"], "signalling": g.get("signalling", True),
                        "pose_variant": sp.pose_variant, "hand_variants": sp.hand_variants,
                        "signal_arms": sp.signal_arms, "pose_tries": sp.tries},
            "pose": sp.pose,
            "baton": gt.baton_info(scene, ctx.skeleton),
            "podium": {"enabled": False} if scene.podium is None else {"enabled": True, **scene.podium},
            "bbox": _with_full_bbox(gt.bbox_from_mask(mask), scene, ctx.skeleton),
            **gt.keypoints(scene, ctx.skeleton),
        },
        "camera": {**scene.camera.to_metadata(), "observer_role": scene.camera_profile["observer_role"],
                   "placement": {k: _r(v) for k, v in scene.camera_placement.items()}},
        "background": {"id": scene.background["id"], "sha256": scene.background["sha256"],
                       "category": scene.background.get("category"), "tags": scene.background["tags"],
                       "ground_check": scene.ground_check,
                       **({"size_reference": scene.size_reference} if scene.size_reference else {}),
                       **({"junction": scene.junction} if scene.junction else {}),
                       "composition": {**scene.composition, "scale": _r(scene.composition["scale"], 6),
                                       "camera_horizon_v": _r(scene.composition["camera_horizon_v"])}},
        "lighting": {**{k: ([_r(x, 6) for x in v] if isinstance(v, list) else _r(v, 6)) for k, v in scene.lighting.items()},
                     **({"shadow": scene.shadow} if scene.shadow is not None else {})},
        "traffic_control": gt.traffic_control(scene, ctx.validation["side_ambiguity_margin_deg"]),
    }
    meta = json.loads(json.dumps(meta))  # normalise numpy types / tuples
    errs = check_ground_truth(meta, ctx.validation, sp.signal_arms) + check_metadata(meta)
    if errs:
        return SampleResult(False, errors=errs)
    return SampleResult(True, meta, rgb, mask)


def _with_full_bbox(bbox: dict | None, scene, skeleton) -> dict | None:
    """Add the unclipped body extent (may extend beyond the image) next to the visible mask bbox."""
    full = gt.bbox_full(scene, skeleton)
    if bbox is not None and full is not None:
        bbox = {**bbox, "full_xyxy": full, "full_source": "capsule_extent"}
    return bbox


def write_sample(out: Path, res: SampleResult) -> dict[str, str]:
    m = res.meta
    imageio.save_rgb(res.rgb, out / m["files"]["rgb"], m["image"]["format"])
    Image.fromarray((res.mask * 255).astype(np.uint8)).save(out / m["files"]["mask"], optimize=False)
    meta_name = f"{m['sample_id']}.json"
    (out / meta_name).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {name: sha256_file(out / name) for name in [*m["files"].values(), meta_name]}


_DATASET_FILE = re.compile(r"^[^/\\]+\.(rgb\.(png|jpg)|mask\.png|json)$")


def prepare_output_dir(out: Path, overwrite: bool = False) -> None:
    """Create `out`, or check that it can take a new dataset: an existing dataset is never mixed with a new one. Not empty ->
    refused, unless `overwrite`; then only the files of a dataset (images, masks, metadata, manifest) are deleted, and a directory
    that holds anything else is refused, so a wrong path cannot wipe unrelated files."""
    if out.exists() and not out.is_dir():
        raise ValueError(f"output path {out} exists and is not a directory")
    out.mkdir(parents=True, exist_ok=True)
    entries = sorted(out.iterdir())
    if not entries:
        return
    if not overwrite:
        raise ValueError(f"output directory {out} is not empty ({len(entries)} entries); use a new directory, "
                         "or --overwrite to replace the dataset in it")
    foreign = [p.name for p in entries if not (p.is_file() and _DATASET_FILE.match(p.name))]
    if foreign:
        raise ValueError(f"--overwrite refused: {out} holds files that are not part of a tosdg dataset: {foreign[:5]}")
    for p in entries:
        p.unlink()


def portable_path(path: Path) -> str:
    """A path for a manifest: relative to the repository when inside it (no user name or drive letter in shared datasets),
    otherwise absolute; "/" separators on every OS."""
    path = Path(path).resolve()
    try:
        return path.relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_portable_path(text: str) -> Path:
    """Inverse of `portable_path` (also reads the absolute paths of manifests written before 1.1)."""
    p = Path(text.replace("\\", "/"))
    return p if p.is_absolute() else repo_root() / p


def run(ctx: GenerationContext, out_dir: str | Path, indices: list[int] | None = None,
        log=print, overwrite: bool = False) -> dict:
    cfg = ctx.config
    out = Path(out_dir)
    prepare_output_dir(out, overwrite)
    max_attempts = cfg.get("max_attempts_per_sample", 20)
    indices = list(range(cfg["num_samples"])) if indices is None else indices
    samples, failures, rejection_counts = [], [], {}
    for i in indices:
        sample_seed = rngmod.derive_sample_seed(cfg["master_seed"], i)
        reasons = []
        for attempt in range(max_attempts):
            res = generate_sample(ctx, i, sample_seed, attempt)
            if res.ok:
                hashes = write_sample(out, res)
                samples.append({"sample_id": res.meta["sample_id"], "index": i, "sample_seed": sample_seed,
                                "attempt": attempt, "metadata": f"{res.meta['sample_id']}.json", "sha256": hashes})
                log(f"[{i:04d}] ok  attempt={attempt} gesture={res.meta['officer']['gesture']['id']} "
                    f"camera={res.meta['camera']['profile']} bbox={res.meta['officer']['bbox']['xyxy']}")
                break
            reasons.append(res.errors[0])
            key = res.errors[0].split(":")[0]
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
        else:
            failures.append({"index": i, "sample_seed": sample_seed, "reasons": reasons[-5:]})
            log(f"[{i:04d}] FAILED after {max_attempts} attempts: {reasons[-1]}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": {"name": "tosdg", "version": GENERATOR_VERSION, "backend": cfg["backend"]},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {**cfg, "background_library": portable_path(ctx.backgrounds.root)},
        "config_sha256": ctx.config_sha256,
        "background_library_resolved": portable_path(ctx.backgrounds.root),
        "asset_hashes": ctx.asset_hashes,
        "assets_sha256": ctx.assets_sha256,
        "samples": sorted(samples, key=lambda s: s["index"]),
        "failures": failures,
        "rejection_counts": rejection_counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
