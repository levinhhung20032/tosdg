"""Command line entry point:  python -m tosdg <command> ...  (or the `tosdg` script).

Contents:
  constants: COCO_EDGES
  cmd_validate_profile
  cmd_generate
  cmd_validate
  cmd_reproduce — Check the files, then regenerate samples and compare hashes
  cmd_bg_import
  cmd_bg_ground — Detect (or set by hand, or remove) the standable-ground masks
  cmd_bg_people — Find (or remove) the reference people of a library's backgrounds
  cmd_bg_objects — List, mark by hand (barriers...) or remove the reference objects of a background
  cmd_bg_shadows — Read (or mark by hand, or remove) the weather and shadows of backgrounds (the officer's shadow direction)
  cmd_bg_calibrate — The camera of each background as size_reference auto sees it
  cmd_bg_junction — Set, show or remove the road layout (junction) of a background
  cmd_bg_set — Set by hand a background's horizon, camera height, road direction, tags
  cmd_bg_remove — Take backgrounds out of a library (entry, image, ground mask)
  cmd_bg_check
  cmd_fixtures
  cmd_pose_preview — Render a body pose (+ hand poses) from the front for…
  cmd_import_model — Register a 3D model of a person in uniform as the…
  cmd_studio
  cmd_overlay — Contact sheet with ground truth drawn over each image…
  cmd_export — Write a dataset as COCO keypoints, YOLO-pose or CVAT, split into train / val / test
  main
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
              (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def _int_at_least(lo: int):
    def parse(text: str) -> int:
        try:
            v = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{text}' is not an integer") from None
        if v < lo:
            raise argparse.ArgumentTypeError(f"must be >= {lo}, got {v}")
        return v
    return parse


def _fraction(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number") from None
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"must be a fraction of the image height in [0, 1], got {v}")
    return v


def cmd_validate_profile(a) -> int:
    from .gestures import pose_validity
    from .poses import merge
    from .scene import GenerationContext
    try:
        ctx = GenerationContext(a.config, {"profile": a.profile} if a.profile else None)
    except ValueError as e:  # e.g. a selectable uniform without a (valid) mesh under backend 'mesh'
        print("ERROR", e)
        return 1
    errs = []
    for g in ctx.gestures.gestures.values():
        for v in g["pose_variants"]:
            hands = [ctx.poses.hand_pose(max(v.get("hands", {}).get(s, {"relaxed": 1}).items(), key=lambda kv: kv[1])[0], s)
                     for s in ("l", "r")]
            e, _ = pose_validity(ctx.skeleton, g, v, merge(ctx.poses.body_pose(v["pose"]), *hands))
            errs += [f"{g['id']}/{v['id']} base pose: {x}" for x in e]
    vocab = set(ctx.profile["regulation"]["instruction_vocabulary"])
    for g in ctx.gestures.gestures.values():
        bad = set(g["instructions"].values()) - vocab
        if bad:
            errs.append(f"{g['id']}: instructions not in vocabulary: {sorted(bad)}")
    errs += ctx.backgrounds.integrity_errors()
    for e in errs:
        print("ERROR", e)
    print(f"profile {ctx.profile['country']}: {len(ctx.gestures.gestures)} gestures, {len(ctx.uniforms)} uniforms, "
          f"{len(ctx.camera_profiles)} camera profiles, {len(ctx.backgrounds.index['backgrounds'])} backgrounds "
          f"-> {'OK' if not errs else f'{len(errs)} error(s)'}")
    return 1 if errs else 0


def cmd_generate(a) -> int:
    from .generate import run
    from .scene import GenerationContext
    from .validate import validate_dataset
    ov = {}
    if a.num is not None:
        ov["num_samples"] = a.num
    if a.seed is not None:
        ov["master_seed"] = a.seed
    if a.background_library:
        ov["background_library"] = str(Path(a.background_library).resolve())
    if a.image_format:
        ov["image_format"] = a.image_format
    ctx = GenerationContext(a.config, ov)
    man = run(ctx, a.out, overwrite=a.overwrite)
    errs = validate_dataset(a.out)
    for e in errs:
        print("INVALID", e)
    n_ok, n_fail = len(man["samples"]), len(man["failures"])
    print(f"generated {n_ok} sample(s), {n_fail} failure(s), rejections={man['rejection_counts']}, "
          f"dataset validation: {'OK' if not errs else f'{len(errs)} error(s)'}")
    return 1 if (errs or n_fail) else 0


def cmd_validate(a) -> int:
    from .validate import validate_dataset
    errs = validate_dataset(a.dataset)
    for e in errs:
        print("INVALID", e)
    print("dataset OK" if not errs else f"{len(errs)} error(s)")
    return 1 if errs else 0


def cmd_reproduce(a) -> int:
    """Check the dataset files against the manifest (validate), then regenerate samples from the manifest's config + seeds and
    compare their hashes with the manifest's. Non-zero when either check fails."""
    from .generate import resolve_portable_path, run
    from .scene import GenerationContext
    from .validate import validate_dataset
    src = Path(a.dataset)
    man_path = src / "manifest.json"
    if not man_path.exists():
        raise ValueError(f"{src} is not a dataset (no manifest.json)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    cfg = man["config"]
    # the files on disk first: a regenerated hash only proves anything about files that still match the manifest
    disk_errs = validate_dataset(src)
    for e in disk_errs:
        print("ON DISK", e)
    library = Path(a.background_library).resolve() if a.background_library else resolve_portable_path(man["background_library_resolved"])
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        ctx = GenerationContext(cfg_path, {"background_library": str(library)})
        if ctx.assets_sha256 != man["assets_sha256"]:
            print("WARNING assets changed since generation; byte-identical output not expected")
        idx = a.index if a.index else [s["index"] for s in man["samples"]]
        out = Path(td) / "out"
        rerun = run(ctx, out, idx, log=lambda *_: None)
    orig = {s["index"]: s for s in man["samples"]}
    bad = 0
    for s in rerun["samples"]:
        same = s["sha256"] == orig[s["index"]]["sha256"]
        bad += not same
        print(f"[{s['index']:04d}] {'identical' if same else 'DIFFERENT'}")
    print(f"regenerated {len(rerun['samples'])} sample(s): {len(rerun['samples']) - bad} identical to the manifest, {bad} different, "
          f"{len(rerun['failures'])} failed; files on disk: {'match the manifest' if not disk_errs else f'{len(disk_errs)} problem(s)'}")
    return 1 if bad or rerun["failures"] or disk_errs else 0


def cmd_bg_import(a) -> int:
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    if a.id and len(a.images) > 1:
        raise ValueError("--id names one background; import the images one at a time")
    from .ground import GroundDetectorUnavailable
    from .people import PeopleDetectorUnavailable
    detect, find = a.ground == "detect", a.people == "detect"
    for p in a.images:
        e = lib.add(p, a.tag or [], a.category, bg_id=a.id, horizon_y_frac=a.horizon)
        note = ""
        if detect:
            try:
                lib.set_ground(e["id"], "detect")
                note = f" ground={e['ground']['standable_frac']:.0%} standable"
            except GroundDetectorUnavailable as err:
                print(f"warning: {err}; imported without a ground mask (feet are not checked on these photos)")
                detect = False
        if find:                                       # people / vehicles tell the photo's camera (calibrate.py)
            try:
                lib.set_people(e["id"], "detect")
                note += f" people={len(e['people']['found'])} vehicles={len(e['objects']['found'])}"
            except PeopleDetectorUnavailable as err:
                print(f"warning: {err}; imported without reference people / vehicles (the officer is sized from the horizon)")
                find = False
        if find:                                       # the weather and shadows (shadows.py), from the people / vehicles just found
            sh = lib.set_shadows(e["id"])["shadows"]
            note += f" shadows={sh.get('weather', '?')}/{len(sh['found'])}"
        print(f"imported {e['id']} ({e['width']}x{e['height']}) tags={e['tags']}{note}")
    lib.save()
    return 0


def cmd_bg_ground(a) -> int:
    """Detect (or set by hand, or remove) the standable-ground masks of a library's backgrounds."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    ids = a.id or sorted(lib.by_id)
    unknown = [i for i in ids if i not in lib.by_id]
    if unknown:
        raise ValueError(f"unknown background id(s) {unknown}")
    if a.mask and len(ids) != 1:
        raise ValueError("--mask sets the mask of one background: give it with --id")
    for i in ids:
        if a.clear:
            lib.clear_ground(i)
            print(f"{i}: ground mask removed (feet not checked)")
            continue
        if a.mask:
            from PIL import Image
            with Image.open(a.mask) as im:
                m = np.asarray(im.convert("L"))
            lib.set_ground(i, m, method="manual")
        elif "ground" in lib.by_id[i] and not a.force:
            print(f"{i}: has a ground mask ({lib.by_id[i]['ground']['method']}); --force to detect again")
            continue
        else:
            lib.set_ground(i, "detect")
        g = lib.by_id[i]["ground"]
        print(f"{i}: {g['file']}  {g['standable_frac']:.0%} standable  ({g['method']})")
        lib.save()                                         # after each one: detection is slow, an interruption keeps the work done
    lib.save()
    return 0


def cmd_bg_people(a) -> int:
    """Find (or remove) the reference people of a library's backgrounds (people.py)."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    ids = a.id or sorted(lib.by_id)
    unknown = [i for i in ids if i not in lib.by_id]
    if unknown:
        raise ValueError(f"unknown background id(s) {unknown}")
    for i in ids:
        if a.clear:
            lib.clear_people(i)
            print(f"{i}: reference people removed")
            continue
        if "people" in lib.by_id[i] and "objects" in lib.by_id[i] and not a.force:
            print(f"{i}: already searched ({len(lib.by_id[i]['people']['found'])} people, "
                  f"{len(lib.by_id[i]['objects']['found'])} objects); --force to search again")
            continue
        e = lib.set_people(i, "detect")
        kinds = ", ".join(f"{p['kind']} {p['height_px']:.0f}px" for p in e["people"]["found"] + e["objects"]["found"]) or "none usable"
        print(f"{i}: {kinds}")
        lib.save()
    lib.save()
    return 0


def cmd_bg_shadows(a) -> int:
    """Read (or mark by hand, or remove) the weather and the shadows of a library's backgrounds (shadows.py): the direction and
    length of the officer's shadow in config shadow source "photo"."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    ids = a.id or sorted(lib.by_id)
    unknown = [i for i in ids if i not in lib.by_id]
    if unknown:
        raise ValueError(f"unknown background id(s) {unknown}")
    if a.mark:
        if len(ids) != 1:
            raise ValueError("--mark needs exactly one --id")
        fu, fv, tu, tv = a.mark
        e = lib.mark_shadow(ids[0], (fu, fv), (tu, tv), a.height_cm, a.kind)
        lib.save()
        print(f"{ids[0]}: {len(e['shadows']['manual'])} shadow(s) marked by hand")
        return 0
    for i in ids:
        e = lib.by_id[i]
        if a.clear or a.clear_marks:
            lib.clear_shadows(i, manual_only=a.clear_marks)
            print(f"{i}: {'marked shadows' if a.clear_marks else 'shadows'} removed")
            continue
        if "shadows" in e and not a.force:
            sh = e["shadows"]
        else:
            if lib.absent() and i in lib.absent():
                print(f"{i}: photo not on this machine, skipped")
                continue
            sh = lib.set_shadows(i)["shadows"]
        marks = len(sh.get("manual") or [])
        print(f"{i}: {sh.get('weather', '?')}, {sh['status']}, {len(sh['found'])} found" + (f", {marks} marked" if marks else "")
              + ("" if sh["status"] == "found" else f" ({sh['reason']})"))
    lib.save()
    return 0


def cmd_bg_objects(a) -> int:
    """List, mark by hand or remove the reference objects (vehicles, barriers) of one background."""
    from .backgrounds import BackgroundLibrary
    from .calibrate import PRIORS
    lib = BackgroundLibrary(a.library)
    if a.id not in lib.by_id:
        raise ValueError(f"unknown background id '{a.id}'; ids: {sorted(lib.by_id)}")
    e = lib.by_id[a.id]
    objs = list((e.get("objects") or {}).get("found", []))
    if a.clear:
        objs = []
    if a.add:
        kind, *box = a.add
        allowed = sorted(k for k in PRIORS if k not in ("standing", "rider"))
        if kind not in allowed:
            raise ValueError(f"--add: kind must be one of {allowed}, got '{kind}'")
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except ValueError:
            raise ValueError("--add KIND X0 Y0 X1 Y1: the box corners are numbers (photo pixels)") from None
        if not (0 <= x0 < x1 <= e["width"] and 0 <= y0 < y1 <= e["height"]):
            raise ValueError(f"--add: box {box} is not inside the {e['width']}x{e['height']} photo")
        o = {"kind": kind, "feet": [round((x0 + x1) / 2, 1), y1], "height_px": round(y1 - y0, 1), "bbox": [x0, y0, x1, y1],
             "score": 1.0, "manual": True}
        if a.height_cm is not None:
            if a.height_cm <= 0:
                raise ValueError("--height-cm must be > 0")
            o["height_cm"] = float(a.height_cm)
        objs.append(o)
    elif a.height_cm is not None:
        raise ValueError("--height-cm goes with --add")
    if a.clear or a.add:
        lib.set_objects(a.id, objs)
        lib.save()
    for o in objs:
        print(f"{o['kind']:9s} feet ({o['feet'][0]:.0f}, {o['feet'][1]:.0f})  {o['height_px']:.0f} px"
              f"{'  ' + str(o['height_cm']) + ' cm' if 'height_cm' in o else ''}{'  (by hand)' if o.get('manual') else ''}")
    print(f"{a.id}: {len(objs)} object(s)")
    return 0


def cmd_bg_calibrate(a) -> int:
    """The camera that took each background, as size_reference auto uses it (calibrate.py)."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    ids = a.id or sorted(lib.by_id)
    unknown = [i for i in ids if i not in lib.by_id]
    if unknown:
        raise ValueError(f"unknown background id(s) {unknown}")
    for i in ids:
        c = lib.calibration(lib.by_id[i])
        if c["method"] == "none":
            print(f"{i}: unknown camera ({c['reason']})")
            continue
        extra = (f" +-{c['height_se_cm']:.0f} cm, horizon +-{c['horizon_se_frac']:.3f}, kinds {','.join(c['kinds'])}"
                 if "height_se_cm" in c else f", camera height from {c['camera_height_from']}")
        print(f"{i}: {c['method']:7s} camera {c['camera_height_cm']:6.1f} cm  horizon {c['horizon_y_frac']:.3f}  "
              f"objects {c['objects_used']}/{c['candidates']}{extra}")
        if a.verbose and c["reason"] != "ok":
            print(f"    {c['reason']}")
    return 0


def cmd_bg_junction(a) -> int:
    """Set, show or remove the road layout of a background (officer_yaw_mode junction, junction.py)."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    if a.id not in lib.by_id:
        raise ValueError(f"unknown background id '{a.id}'; ids: {sorted(lib.by_id)}")
    e = lib.by_id[a.id]
    if a.clear:
        lib.set_junction(a.id, None)
        lib.save()
    elif a.type:
        if not a.center or not a.road:
            raise ValueError("--type needs --center U V and one --road DEG per road")
        k = (e["width"], e["height"]) if a.frac else (1.0, 1.0)
        lib.set_junction(a.id, {"type": a.type, "center": [a.center[0] * k[0], a.center[1] * k[1]], "arms_deg": a.road})
        lib.save()
    elif a.center or a.road:
        raise ValueError("--center / --road go with --type")
    j = e.get("junction")
    print(f"{a.id}: " + (f"{j['type']} junction, middle {j['center']}, roads at {j['arms_deg']} deg" if j else "no junction (a straight road)"))
    return 0


def cmd_bg_set(a) -> int:
    """Set by hand what the photo cannot tell reliably: horizon (then reliable), camera height, road direction, tags."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    if a.id not in lib.by_id:
        raise ValueError(f"unknown background id '{a.id}'; ids: {sorted(lib.by_id)}")
    e = lib.by_id[a.id]
    if a.horizon is not None:
        e["horizon_y_frac"], e["horizon_source"] = float(a.horizon), "manual"
    if a.camera_height is not None:
        if not 30 <= a.camera_height <= 1000:
            raise ValueError("--camera-height must be 30..1000 cm")
        e["camera_height_cm"] = float(a.camera_height)
    if a.road_yaw is not None:
        if not -180 <= a.road_yaw <= 180:
            raise ValueError("--road-yaw must be -180..180 degrees")
        e["road_yaw_deg"] = float(a.road_yaw)
    if a.tag:
        e["tags"] = sorted(set(a.tag))
    lib.save()
    print(f"{a.id}: horizon {e.get('horizon_y_frac')} ({e.get('horizon_source', 'none')}), camera height "
          f"{lib.camera_height_cm(e):g} cm, road {e.get('road_yaw_deg', 0):g} deg, tags {e['tags']}")
    return 0


def cmd_bg_remove(a) -> int:
    """Take backgrounds out of a library (entry, image, ground mask)."""
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    unknown = [i for i in a.ids if i not in lib.by_id]
    if unknown:
        raise ValueError(f"not in {a.library}: {unknown}; ids there: {sorted(lib.by_id)}")
    for i in a.ids:
        lib.remove(i)
    lib.save()
    print(f"removed {len(a.ids)} background(s); {len(lib.index['backgrounds'])} left in {a.library}")
    return 0


def cmd_bg_check(a) -> int:
    from .backgrounds import BackgroundLibrary
    lib = BackgroundLibrary(a.library)
    errs = lib.integrity_errors(missing_photos=False)
    for e in errs:
        print("ERROR", e)
    absent = lib.absent()
    if absent:
        print(f"note: {len(absent)} listed photo(s) not on this machine (optional; fetch_backgrounds.bat downloads them): "
              + ", ".join(absent[:8]) + (" ..." if len(absent) > 8 else ""))
    print(f"{len(lib.index['backgrounds']) - len(absent)} of {len(lib.index['backgrounds'])} backgrounds here, "
          f"{'OK' if not errs else f'{len(errs)} error(s)'}")
    return 1 if errs else 0


def cmd_fixtures(a) -> int:
    from .fixtures import build_fixture_library
    lib = build_fixture_library(a.library)
    print(f"fixture library at {a.library}: {sorted(lib.by_id)}")
    return 0


def cmd_pose_preview(a) -> int:
    """Render a body pose (+ hand poses) from the front for visual inspection."""
    from PIL import Image
    from .camera import Camera, look_at_rotator
    from .poses import PoseLibrary, merge
    from .render_proxy import render
    from .schema import load_json, repo_root
    from .skeleton import Skeleton
    root = repo_root()
    sk = Skeleton.load(root / "characters" / a.character / "skeleton.json")
    pl = PoseLibrary(root / "poses")
    pose = merge(pl.body_pose(a.pose), pl.hand_pose(a.hand_l, "l"), pl.hand_pose(a.hand_r, "r"))
    from . import baton as batonmod
    bspec = batonmod.normalise({"enabled": True, "hand": a.baton}) if a.baton else None
    if bspec:
        pose, grip = batonmod.apply_grip(sk, pose, bspec, a.baton_axis, pl.hand_pose("fist", a.baton))
        print(f"baton in the {a.baton} hand, {a.baton_axis}: fist closure {grip['closure']}, thumb {grip['thumb_closure']}")
    errs = sk.pose_errors(pose)
    fk = sk.fk(pose)
    errs += [f"collision {c}" for c in sk.collisions(fk)]
    if bspec:
        errs += [f"baton {h}" for h in batonmod.body_collisions(sk, fk, bspec, a.baton_axis)]
    unis = load_json(root / "profiles" / a.profile / "uniforms.json", "uniforms")["uniforms"]
    uni = next((u for u in unis if u["id"] == a.uniform), None) if a.uniform else unis[0]   # the first uniform is the profile's default
    if uni is None:
        print(f"unknown uniform '{a.uniform}'")
        return 1
    mesh = None
    lift = 0.0
    if a.renderer in ("mesh", "gpu"):
        if "mesh" not in uni:
            print(f"uniform {uni['id']} has no mesh; use --renderer proxy or import a model (tosdg import-model model.glb)")
            return 1
        from .mesh_renderer import get as get_mesh_renderer
        from .uniform_mesh import check_against_skeleton, read_glb
        try:
            render_mesh = get_mesh_renderer("gpu" if a.renderer == "gpu" else "cpu")[0]
        except RuntimeError as e:
            print(f"--renderer gpu: {e}")
            return 1
        mesh = read_glb(root / "profiles" / a.profile / uni["mesh"]["asset"])
        errs += [f"mesh: {m}" for m in check_against_skeleton(mesh, sk)]
        pp = batonmod.prop_path(root / "profiles" / a.profile)
        mesh = batonmod.mesh_with_officer_baton(mesh, bspec, a.baton_axis, prop=batonmod.load_prop(pp) if pp.exists() else None)
        if a.podium:
            from . import podium as podiummod
            prop = podiummod.load(podiummod.prop_path(root / "profiles" / a.profile))
            sole = podiummod.sole_level(mesh, sk)
            mesh = podiummod.mesh_with_podium(mesh, prop, sole)
            lift = podiummod.lift_cm(prop, sole)                       # the podium is on the ground, the officer on top of it
            fk = fk.transformed(np.eye(3), np.array([0.0, 0.0, lift]))
    light = {"sun_direction": [0.6, 0.3, 0.74], "sun_color_rgb": [1, 1, 1], "sun_intensity": 0.9,
             "ambient_intensity": 0.4, "exposure_ev": 0.0}
    tiles = []
    for az in (0, 90, 180, 270):
        eye = np.array([500 * np.cos(np.radians(az)), 500 * np.sin(np.radians(az)), 110.0 + lift])
        p, y = look_at_rotator(eye, [0, 0, 105 + lift])
        cam = Camera("preview", tuple(eye), p, y, 0, 35, 36, 360, 480)
        bg = Image.new("RGB", (360, 480), (200, 200, 200))
        if mesh is not None:
            rgb, _ = render_mesh(mesh, fk, cam, bg, uni["proxy_colors"], light, a.brightness)
        else:
            extra = [(*batonmod.segment(sk, fk, bspec, a.baton_axis), bspec["radius_cm"], "leather")] if bspec else None
            rgb, _ = render(sk, fk, cam, bg, uni["proxy_colors"], light, extra_capsules=extra)
        tiles.append(rgb)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(a.out)
    print(f"wrote {a.out} (front | right | back | left); renderer {a.renderer}, uniform {uni['id']}; pose errors: {errs or 'none'}")
    return 1 if errs else 0


def cmd_import_model(a) -> int:
    """Register a 3D model of a person in uniform as the profile's active uniform."""
    from .models import register_model
    from .schema import repo_root
    if a.name is not None and not re.search(r"[A-Za-z0-9]", a.name):
        print(f"import failed: --name '{a.name}' has no letter or digit (leave --name out to use the file name)")
        return 1
    try:
        r = register_model(repo_root(), a.profile, a.model, a.name, a.character, activate=not a.no_activate)
    except (ValueError, OSError) as e:
        print(f"import failed: {e}")
        return 1
    how = "own skeleton and skin weights kept" if r["skin"] == "model" else "no usable skeleton: rigged by bone proximity"
    state = "registered (not activated)" if a.no_activate else "now the active model"
    print(f"uniform '{r['id']}' is {state} ({r['vertices']} vertices): {how}; mesh {r['asset']}, source {r['model']}")
    return 0


def cmd_studio(a) -> int:
    from .studio.server import serve
    return serve(a.port, a.workspace, a.profile, a.library, not a.no_browser, a.exit_on_close)


def cmd_overlay(a) -> int:
    """Contact sheet with ground truth drawn over each image (visual GT check)."""
    from PIL import Image, ImageDraw
    d = Path(a.dataset)
    if not (d / "manifest.json").exists():
        raise ValueError(f"{d} is not a dataset (no manifest.json)")
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    tiles = []
    for s in man["samples"][: a.max]:
        m = json.loads((d / s["metadata"]).read_text())
        im = Image.open(d / m["files"]["rgb"]).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.rectangle(m["officer"]["bbox"]["xyxy"], outline=(0, 255, 255))
        for set_name, col in (("coco17", (255, 0, 255)), ("hand21_l", (255, 255, 0)), ("hand21_r", (0, 255, 0))):
            kp = m["officer"]["keypoints_2d"].get(set_name)
            if not kp:
                continue
            uv = kp["uv"]
            if set_name == "coco17":
                for i, j in COCO_EDGES:
                    if uv[i] and uv[j]:
                        dr.line([tuple(uv[i]), tuple(uv[j])], fill=col, width=1)
            for p in uv:
                if p:
                    dr.point(tuple(p), fill=col)
        tc = m["traffic_control"]
        label = (f"{m['officer']['gesture']['id']} | {m['camera']['profile']} {tc['approach_side']} -> "
                 f"{tc.get('instruction_for_observer', tc.get('effective_instruction'))}")
        dr.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
        dr.text((3, 1), label, fill=(255, 255, 255))
        if a.zoom:
            x0, y0, x1, y1 = m["officer"]["bbox"]["xyxy"]
            pad = max(x1 - x0, y1 - y0) * 0.3
            crop = im.crop((int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad)))
            crop = crop.resize((int(im.height * crop.width / max(crop.height, 1)), im.height), Image.NEAREST)
            im = Image.fromarray(np.concatenate([np.asarray(im), np.asarray(crop)], axis=1))
        tiles.append(np.asarray(im))
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 4), (0, w - t.shape[1]), (0, 0))) for t in tiles]
    Image.fromarray(np.concatenate(tiles, axis=0)).save(a.out)
    print(f"wrote {a.out} ({len(tiles)} samples)")
    return 0


def _ratios(text: str) -> tuple[float, float, float]:
    try:
        r = tuple(float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not three numbers train,val,test") from None
    if len(r) != 3 or min(r) < 0 or sum(r) <= 0:
        raise argparse.ArgumentTypeError(f"need three non-negative numbers train,val,test with a positive sum, got '{text}'")
    return r


def cmd_export(a) -> int:
    """Write a dataset as COCO keypoints, YOLO-pose or CVAT, split into train / val / test."""
    from .export import export
    r = export(a.dataset, a.out, a.format, a.split, a.seed, a.yolo_classes)
    c = r["counts"]
    print(f"wrote {a.out} ({a.format}): train {c['train']}, val {c['val']}, test {c['test']} "
          f"(split seed {r['split_seed']}, {len(r['keypoints'])} keypoints)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tosdg", description="Traffic Officer Synthetic Data Generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate-profile", help="schema + reference + base-pose checks")
    s.add_argument("config"); s.add_argument("--profile")
    s.set_defaults(func=cmd_validate_profile)

    s = sub.add_parser("generate", help="generate a batch")
    s.add_argument("config"); s.add_argument("--out", required=True)
    s.add_argument("--num", type=_int_at_least(1), help="number of samples (overrides the config)")
    s.add_argument("--seed", type=_int_at_least(0), help="master seed (overrides the config)")
    s.add_argument("--background-library")
    s.add_argument("--overwrite", action="store_true",
                   help="replace the dataset already in --out (only dataset files are deleted; other files make it refuse)")
    s.add_argument("--image-format", choices=["png", "jpg"], help="file format of the rgb images (default: the config's image_format, else png)")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("validate", help="validate a generated dataset directory")
    s.add_argument("dataset"); s.set_defaults(func=cmd_validate)

    s = sub.add_parser("reproduce", help="check the files against the manifest, then regenerate the samples and compare hashes")
    s.add_argument("dataset"); s.add_argument("--index", type=int, nargs="*")
    s.add_argument("--background-library", help="where the background library is now (default: the one recorded in the manifest)")
    s.set_defaults(func=cmd_reproduce)

    bg = sub.add_parser("backgrounds", help="backplate library").add_subparsers(dest="bgcmd", required=True)
    s = bg.add_parser("import"); s.add_argument("library"); s.add_argument("images", nargs="+")
    s.add_argument("--tag", action="append"); s.add_argument("--category", default="uncategorized")
    s.add_argument("--id", help="background id (default: from the file name); one image only")
    s.add_argument("--horizon", type=_fraction, help="horizon row as a fraction of image height, 0..1 (default: detected)")
    s.add_argument("--ground", choices=["detect", "none"], default="detect",
                   help="standable-ground mask (road/sidewalk) so the officer never stands on a wall: detect (default; needs torch + "
                        "transformers, else skipped with a warning) or none")
    s.add_argument("--people", choices=["detect", "none"], default="detect",
                   help="reference people (standing / on a motorbike) and vehicles to tell the photo's camera from: detect (default; needs ultralytics, "
                        "else skipped with a warning) or none")
    s.set_defaults(func=cmd_bg_import)
    s = bg.add_parser("people", help="find / remove the reference people and vehicles of the backgrounds (to tell their camera)")
    s.add_argument("library"); s.add_argument("--id", action="append", help="only these backgrounds (default: all)")
    s.add_argument("--force", action="store_true", help="search again even if already searched")
    s.add_argument("--clear", action="store_true", help="remove the reference people")
    s.set_defaults(func=cmd_bg_people)
    s = bg.add_parser("shadows", help="read / mark by hand / remove the weather and shadows of the backgrounds (config shadow source photo)")
    s.add_argument("library"); s.add_argument("--id", action="append", help="only these backgrounds (default: all)")
    s.add_argument("--force", action="store_true", help="read again even if already read (marks by hand are kept)")
    s.add_argument("--clear", action="store_true", help="remove the shadows (and the marks)")
    s.add_argument("--clear-marks", action="store_true", help="remove only the shadows marked by hand")
    s.add_argument("--mark", nargs=4, type=float, metavar=("FEET_U", "FEET_V", "TIP_U", "TIP_V"),
                   help="mark one shadow by hand (photo pixels: where an object stands, where its shadow ends); needs one --id")
    s.add_argument("--height-cm", type=float, default=165.0, help="--mark: height of the object that casts it (default 165, a person)")
    s.add_argument("--kind", default="standing", help="--mark: what casts it (default standing)")
    s.set_defaults(func=cmd_bg_shadows)
    s = bg.add_parser("objects", help="list / mark by hand / remove the reference objects (vehicles, barriers) of a background")
    s.add_argument("library"); s.add_argument("id")
    s.add_argument("--add", nargs=5, metavar=("KIND", "X0", "Y0", "X1", "Y1"),
                   help="mark an object standing on the ground: kind (car, bus, truck, motorbike, bicycle, barrier) and its box in "
                        "photo pixels (bottom = where it touches the ground)")
    s.add_argument("--height-cm", type=float, help="real height of the added object (default: typical for its kind)")
    s.add_argument("--clear", action="store_true", help="remove every object (then --add starts a new list)")
    s.set_defaults(func=cmd_bg_objects)
    s = bg.add_parser("junction", help="set / show / remove the road layout of a background (officer_yaw_mode junction)")
    s.add_argument("library"); s.add_argument("id")
    s.add_argument("--type", choices=["cross", "T", "Y", "roundabout"], help="ngã tư (or more roads) / ngã ba chữ T / chữ Y / bùng binh")
    s.add_argument("--center", nargs=2, type=float, metavar=("U", "V"), help="ground point where the officer stands (photo pixels)")
    s.add_argument("--road", type=float, action="append", metavar="DEG",
                   help="direction of one road leaving the junction, degrees from the view: 0 away from the camera, 90 right, "
                        "180 towards the camera, -90 left (repeat for each road)")
    s.add_argument("--frac", action="store_true", help="--center is a fraction of the photo width / height")
    s.add_argument("--clear", action="store_true", help="remove the junction (a straight road)")
    s.set_defaults(func=cmd_bg_junction)
    s = bg.add_parser("calibrate", help="show the camera of each background told by its people / objects / horizon (size_reference auto)")
    s.add_argument("library"); s.add_argument("--id", action="append", help="only these backgrounds (default: all)")
    s.add_argument("-v", "--verbose", action="store_true", help="also say why people / objects were not enough")
    s.set_defaults(func=cmd_bg_calibrate)
    s = bg.add_parser("ground", help="detect / set / remove the standable-ground masks (where the officer's feet may be)")
    s.add_argument("library"); s.add_argument("--id", action="append", help="only these backgrounds (default: all)")
    s.add_argument("--force", action="store_true", help="detect again even if a mask exists")
    s.add_argument("--mask", help="use this image as the mask (white = may stand), e.g. painted by hand; needs one --id")
    s.add_argument("--clear", action="store_true", help="remove the mask (the background is then not checked)")
    s.set_defaults(func=cmd_bg_ground)
    s = bg.add_parser("set", help="set a background's horizon (by hand = reliable), camera height, road direction or tags")
    s.add_argument("library"); s.add_argument("id")
    s.add_argument("--horizon", type=_fraction); s.add_argument("--camera-height", type=float, help="cm, height of the camera that took the photo")
    s.add_argument("--road-yaw", type=float, help="degrees: direction of the traffic relative to the view (+ = right)")
    s.add_argument("--tag", action="append", help="replace the tags (repeat)")
    s.set_defaults(func=cmd_bg_set)
    s = bg.add_parser("remove", help="take backgrounds out of the library (entry, image, ground mask)")
    s.add_argument("library"); s.add_argument("ids", nargs="+"); s.set_defaults(func=cmd_bg_remove)
    s = bg.add_parser("check"); s.add_argument("library"); s.set_defaults(func=cmd_bg_check)

    s = sub.add_parser("fixtures", help="build the synthetic fixture backplate library")
    s.add_argument("library"); s.set_defaults(func=cmd_fixtures)

    s = sub.add_parser("overlay", help="contact sheet with GT drawn over images")
    s.add_argument("dataset"); s.add_argument("--out", default="overlay.png"); s.add_argument("--max", type=int, default=12)
    s.add_argument("--zoom", action="store_true", help="append an enlarged crop around the bbox")
    s.set_defaults(func=cmd_overlay)

    s = sub.add_parser("export", help="write a dataset as COCO keypoints, YOLO-pose or CVAT (train / val / test split)")
    s.add_argument("dataset"); s.add_argument("--format", required=True, choices=["coco", "yolo", "cvat"])
    s.add_argument("--out", required=True, help="new or empty directory")
    s.add_argument("--split", type=_ratios, default=(0.8, 0.1, 0.1), help="train,val,test ratios (default 0.8,0.1,0.1)")
    s.add_argument("--seed", type=_int_at_least(0), help="split seed (default: the dataset's master seed)")
    s.add_argument("--yolo-classes", choices=["officer", "gesture"], default="officer",
                   help="yolo: one class 'traffic_officer' (default) or one class per gesture")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("pose-preview", help="render a pose from four sides")
    s.add_argument("pose"); s.add_argument("--hand-l", default="relaxed"); s.add_argument("--hand-r", default="relaxed")
    s.add_argument("--character", default="tripo_v1"); s.add_argument("--profile", default="VN")
    s.add_argument("--out", default="pose_preview.png")
    s.add_argument("--renderer", choices=["mesh", "gpu", "proxy"], default="mesh",
                   help="mesh = skinned uniform mesh on the CPU (default), gpu = the same mesh with the GPU renderer, proxy = capsules")
    s.add_argument("--uniform", help="uniform id (default: the profile's first uniform)")
    s.add_argument("--brightness", type=float, default=1.0, help="cloth brightness multiplier (mesh renderer)")
    s.add_argument("--baton", choices=["l", "r"], help="hold a baton in this hand (fingers close around it)")
    s.add_argument("--baton-axis", choices=["across", "along"], default="across", help="baton across the fist (default) or along the forearm")
    s.add_argument("--podium", action="store_true", help="stand the officer on the traffic podium (mesh renderer)")
    s.set_defaults(func=cmd_pose_preview)

    s = sub.add_parser("import-model", help="use a GLB of a person in uniform as the active model (own skeleton if it has one, else rig it)")
    s.add_argument("model"); s.add_argument("--name", help="uniform id (default: the file name)")
    s.add_argument("--profile", default="VN"); s.add_argument("--character", default="tripo_v1")
    s.add_argument("--no-activate", action="store_true", help="only register it; keep the current active model")
    s.set_defaults(func=cmd_import_model)

    s = sub.add_parser("studio", help="interactive setup tool: backgrounds, camera FOV, draggable skeleton, uniform views")
    s.add_argument("--port", type=int, default=8765); s.add_argument("--profile", default="VN")
    s.add_argument("--workspace", help="default: data/studio")
    s.add_argument("--library", help="background library used by 'add to library' (default: data/backgrounds)")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--exit-on-close", action="store_true", help="stop the server when the last browser tab is closed")
    s.set_defaults(func=cmd_studio)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except Exception as e:
        # a user mistake (missing file, bad JSON, schema violation, unknown id ...) is one line, not a traceback;
        # TOSDG_DEBUG=1 shows the traceback
        if os.environ.get("TOSDG_DEBUG") or not isinstance(e, (OSError, ValueError)):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
