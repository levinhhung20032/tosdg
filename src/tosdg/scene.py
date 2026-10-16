"""Generation context (loaded, hashed assets) and seeded scene sampling.

`build_scene(ctx, seed)` is a pure function of (config, asset contents, seed):
every random draw comes from a named stream derived from `seed`.

Contents:
  constants: GENERATOR_VERSION, ROAD_YAW_JITTER_DEG, CAMERA_OVERRIDE_KEYS, HEAD_TOP_CM, SIZE_MARGIN, DEFAULT_VALIDATION
  class SizeReferenceError
  class GenerationContext
  class Scene
  build_scene
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import rng as rngmod
from .backgrounds import BackgroundLibrary, crop_backplate, plan_backplate, sha256_file
from . import ground as groundmod
from .camera import Camera, sample_camera
from . import baton as batonmod
from . import podium as podiummod
from . import people as peoplemod
from . import calibrate as calmod
from . import junction as junctionmod
from .gestures import GestureLibrary, SampledPose
from .lens import canvas_camera, pinhole_horizon_row, sample_lens
from .lighting import kelvin_to_rgb, sample_lighting
from .poses import PoseLibrary
from .schema import SchemaError, load_json, repo_root, validate_instance
from .skeleton import FKResult, Skeleton, axis_rotation
from .uniform_mesh import UniformMesh, check_against_skeleton, read_glb, spec_sha256

GENERATOR_VERSION = "0.6.0"          # 0.4.0: feet kept on the backplate's standable ground (ground.py); 0.5.0: size_reference, officer_yaw_mode, locks; 0.6.0: size_reference auto = calibrated camera (calibrate.py)
ROAD_YAW_JITTER_DEG = 5.0           # officer_yaw_mode "road": an officer is never exactly square to the road
CAMERA_OVERRIDE_KEYS = ("distance_cm", "height_cm", "azimuth_deg", "focal_length_mm", "aim_height_cm", "roll_deg", "sensor_width_mm")
IMPLEMENTED_BACKENDS = ("proxy", "mesh")

HEAD_TOP_CM = 8.0      # the cap / hand reaches past the highest joint
SIZE_MARGIN = 1.15     # min_officer_height_px: aim this much above the minimum (lens, pitch and the pinhole estimate shrink it)
CALIBRATED_ZOOMS = (1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0)    # size_reference auto: photo zooms tried, smallest first (<= MAX_COVER_SCALE)
FRAME_MARGIN = 0.02    # size_reference auto: the officer is planned this far (x frame height) inside the top / bottom edge
JUNCTION_TRIES = 12    # officer_yaw_mode junction: spots tried around the junction's middle (feet on the road, clear of nearer objects)
LIGHT_REF = 0.2        # match_backplate_light: log-average linear luminance of a daylit street photo (median of the library: exposure 0)
LIGHT_MIN_EV = -2.0    # ... the darkest the officer is made (a night street lit by lamps; darker = an invisible officer)
EXPOSURE_RANGE = (-2.5, 2.0)   # the final exposure_ev of any sample (drawn + matched): never a black or a white officer
NIGHT_SUN = 0.25       # ... photos tagged night: the "sun" is a street lamp: this much of its intensity, warm light
NIGHT_AMBIENT = 0.6    # ... and this much of the sky / ambient light
NIGHT_KELVIN = 3400.0

DEFAULT_VALIDATION = {
    "min_bbox_height_px": 24, "min_keypoints_in_frame_frac": 0.25, "require_signal_arm_in_frame": False,
    "max_bbox_edge_touch": True, "min_image_std": 2.0, "side_ambiguity_margin_deg": 10.0,
    "min_officer_level": 12.0, "max_officer_level": 245.0,
}


class SizeReferenceError(groundmod.PlacementError):
    """size_reference "auto": the photo's camera is unknown, or the officer cannot be sized consistently with the photo."""


class GenerationContext:
    def __init__(self, config_path: str | Path, overrides: dict | None = None):
        self.config_path = Path(config_path).resolve()
        cfg = load_json(self.config_path, "generation_config")
        if overrides:                          # command-line overrides (--num, --seed ...) obey the same schema as the file
            cfg.update(overrides)
            try:
                validate_instance(cfg, "generation_config")
            except SchemaError as e:
                raise SchemaError(f"{self.config_path} with command-line overrides {sorted(overrides)}: {e}") from None
        self.config = cfg
        if cfg["backend"] not in IMPLEMENTED_BACKENDS:
            raise ValueError(f"backend '{cfg['backend']}' is not implemented yet; use one of {list(IMPLEMENTED_BACKENDS)}")
        if "mesh_renderer" in cfg:
            if cfg["backend"] != "mesh":
                raise ValueError("config mesh_renderer is for backend 'mesh' only")
            if cfg["mesh_renderer"] == "gpu":
                from .mesh_renderer import get as get_mesh_renderer
                try:
                    get_mesh_renderer("gpu")
                except (RuntimeError, ImportError) as e:
                    raise ValueError(f"config mesh_renderer 'gpu': {e}") from None
        root = repo_root()
        self.root = root
        self.profile_dir = root / "profiles" / cfg["profile"]
        self.profile = load_json(self.profile_dir / "profile.json", "profile")
        files = self.profile["files"]
        self.skeleton = Skeleton.load(root / "characters" / cfg["character"] / "skeleton.json")
        self.poses = PoseLibrary(root / "poses")
        self.gestures = GestureLibrary(load_json(self.profile_dir / files["gestures"], "gestures"), self.poses)
        self.uniforms = {u["id"]: u for u in load_json(self.profile_dir / files["uniforms"], "uniforms")["uniforms"]}
        self.camera_profiles = {p["id"]: p for p in
                                load_json(self.profile_dir / files["camera_profiles"], "camera_profiles")["profiles"]}
        bg = Path(cfg["background_library"])
        bg = bg if bg.is_absolute() else self.config_path.parent / bg
        if not (bg / "index.json").exists():
            raise ValueError(f"background library {bg.resolve()} does not exist (no index.json); build one with "
                             "'tosdg fixtures <dir>' or 'tosdg backgrounds import <dir> <images>'")
        self.backgrounds = BackgroundLibrary(bg)
        if cfg.get("background_ids"):                 # only these photos (Studio "random generate" with the background locked)
            unknown = sorted(set(cfg["background_ids"]) - set(self.backgrounds.by_id))
            if unknown:
                raise ValueError(f"config background_ids: not in the background library {bg}: {unknown}")
            self.backgrounds.index["backgrounds"] = [b for b in self.backgrounds.index["backgrounds"] if b["id"] in cfg["background_ids"]]
        # The library's photos are not in git: the ones listed but not on this machine are optional and skipped
        self.absent_backgrounds = self.backgrounds.absent()
        if self.absent_backgrounds:
            if cfg.get("background_ids"):
                raise ValueError(f"config background_ids: {self.absent_backgrounds} not on this machine (download them with "
                                 "fetch_backgrounds.bat, or upload / import your own photos)")
            self.backgrounds.index["backgrounds"] = [b for b in self.backgrounds.index["backgrounds"]
                                                     if b["id"] not in self.absent_backgrounds]
            if not self.backgrounds.index["backgrounds"]:
                raise ValueError(f"background library {bg.resolve()}: none of its {len(self.absent_backgrounds)} photos is on this "
                                 "machine. They are optional: download them with fetch_backgrounds.bat, or add your own "
                                 "(Studio upload, 'tosdg backgrounds import'), or build test plates with 'tosdg fixtures <dir>'")
            print(f"note: {len(self.absent_backgrounds)} photo(s) of {bg.resolve()} are listed but not on this machine (optional; "
                  f"fetch_backgrounds.bat downloads them); using the {len(self.backgrounds.index['backgrounds'])} that are")
        if cfg.get("min_officer_height_px") and cfg.get("size_reference") != "auto":
            raise ValueError("config min_officer_height_px needs \"size_reference\": \"auto\": a bigger officer must "
                             "come with a consistently scaled photo, not alone in front of a photo that stays the same")
        self.validation = {**DEFAULT_VALIDATION, **cfg.get("validation", {})}
        if cfg.get("min_officer_height_px"):  # the size is also enforced on the rendered bbox unless the config says otherwise
            if cfg["min_officer_height_px"] > cfg["resolution"][1]:
                raise ValueError(f"config min_officer_height_px {cfg['min_officer_height_px']} is taller than the image ({cfg['resolution'][1]} px)")
            if "min_bbox_height_px" not in cfg.get("validation", {}):
                self.validation["min_bbox_height_px"] = cfg["min_officer_height_px"]
        if cfg.get("size_reference") == "auto":                # after the config checks: a refused config runs no detector
            self._find_people(bg, log=print)
        # size_reference auto: the camera of every photo (calibrate.py), from the stored detections (no detector here)
        self.calibrations = ({b["id"]: BackgroundLibrary.calibration(b) for b in self.backgrounds.index["backgrounds"]}
                             if cfg.get("size_reference") == "auto" else {})
        self.baton = batonmod.normalise(cfg.get("baton"))             # None = no baton
        self.podium_probability = float((cfg.get("podium") or {}).get("probability", 0.0))   # chance that a sample stands on the traffic podium
        self.junction_mode = cfg.get("officer_yaw_mode") == "junction"
        if self.junction_mode and cfg.get("size_reference") != "auto":
            raise ValueError("config officer_yaw_mode \"junction\" needs \"size_reference\": \"auto\" (the officer is put in the middle "
                             "of the junction through the photo's calibrated camera)")
        # roundabouts (officer_yaw_mode junction): chance of the podium and of a held baton (the podium needs the mesh backend)
        self.roundabout = {**junctionmod.ROUNDABOUT_DEFAULTS, **(cfg.get("roundabout") or {})}
        self.podium_prop = None
        if self.junction_mode and self.podium_probability == 0 and self.roundabout["podium_probability"] > 0 \
                and cfg["backend"] == "mesh" and podiummod.prop_path(self.profile_dir).exists():
            self.podium_prop = podiummod.load(podiummod.prop_path(self.profile_dir))
        if self.podium_probability > 0:
            if cfg["backend"] != "mesh":
                raise ValueError("config podium is drawn by the 'mesh' backend only")
            pp = podiummod.prop_path(self.profile_dir)
            if not pp.exists():
                raise ValueError(f"config podium: profile {cfg['profile']} has no {pp.relative_to(self.profile_dir)}")
            self.podium_prop = podiummod.load(pp)
        pp = batonmod.prop_path(self.profile_dir)
        self.baton_prop = batonmod.load_prop(pp) if pp.exists() else None    # the 3D baton model (else a plain cylinder); held or on the belt
        self._check_weights()
        self.meshes = self._load_meshes()
        self.asset_hashes = self._hash_assets()
        self.assets_sha256 = hashlib.sha256(
            "\n".join(f"{k}:{v}" for k, v in sorted(self.asset_hashes.items())).encode()).hexdigest()
        # The library location is machine-local; its *content* is covered by assets_sha256.
        hashed_cfg = {k: v for k, v in cfg.items() if k != "background_library"}
        self.config_sha256 = hashlib.sha256(json.dumps(hashed_cfg, sort_keys=True).encode()).hexdigest()

    def _find_people(self, lib_dir: Path, log=print) -> None:
        """size_reference auto: run the detector (people.py) once, before any sampling, on the photos not searched yet (or
        searched before vehicles were detected: no "objects") and store the result in the library, so the assets (and their
        hash) are fixed before the first sample."""
        need = [b["id"] for b in self.backgrounds.index["backgrounds"] if "people" not in b or "objects" not in b]   # (background_ids applied)
        if not need:
            return
        lib = BackgroundLibrary(lib_dir)
        try:
            for i in need:
                e = lib.set_people(i, "detect")
                log(f"reference people of {i}: {len(e['people']['found'])}, vehicles: {len(e['objects']['found'])}")
                lib.save()
        except peoplemod.PeopleDetectorUnavailable as e:
            log(f"warning: {e}; photos not searched yet are sized from their horizon")
        keep = {b["id"] for b in self.backgrounds.index["backgrounds"]}
        self.backgrounds = BackgroundLibrary(lib_dir)
        self.backgrounds.index["backgrounds"] = [b for b in self.backgrounds.index["backgrounds"] if b["id"] in keep]

    def _check_weights(self) -> None:
        ranges = [("officer_yaw_deg", self.config.get("officer_yaw_deg"))]
        ranges += [(f"lighting.{k}", v) for k, v in (self.config.get("lighting") or {}).items()]
        for key, r in ranges:
            if r is not None and r[0] > r[1]:
                raise ValueError(f"config {key} must be [min, max] with min <= max, got {r}")
        for key, known in (("gesture_weights", self.gestures.gestures), ("camera_profile_weights", self.camera_profiles),
                           ("uniform_weights", self.uniforms)):
            unknown = set(self.config.get(key, {})) - set(known)
            if unknown:
                raise ValueError(f"config {key}: unknown ids {sorted(unknown)}")
        if not self.backgrounds.index["backgrounds"]:
            raise ValueError(f"background library {self.backgrounds.root} is empty")
        lo, hi = self.config.get("uniform_brightness_range", [1.0, 1.0])
        if lo > hi:
            raise ValueError("config uniform_brightness_range must be [lo, hi] with lo <= hi")
        if (lo, hi) != (1.0, 1.0) and self.config["backend"] != "mesh":
            raise ValueError("config uniform_brightness_range is applied by the 'mesh' backend only; set backend to 'mesh' or the range to [1, 1]")

    def selectable_uniforms(self) -> list[str]:
        """Uniforms a sample can actually get (weight > 0 after the config's uniform_weights)."""
        return sorted(k for k, w in self.weights("uniform_weights", self.uniforms).items() if w > 0)

    def _load_meshes(self) -> dict[str, UniformMesh]:
        """backend 'mesh': every selectable uniform must have a valid skinned mesh for THIS skeleton, built from the current spec."""
        if self.config["backend"] != "mesh":
            return {}
        out = {}
        for uid in self.selectable_uniforms():
            u = self.uniforms[uid]
            if "mesh" not in u:
                raise ValueError(f"backend 'mesh': uniform '{uid}' has no mesh entry (add one, or give it weight 0 / leave it out of uniform_weights)")
            m = u["mesh"]
            src = self.profile_dir / (m["spec"] if "spec" in m else m["fit"])
            asset = self.profile_dir / m["asset"]
            if not asset.exists() or not src.exists():
                raise ValueError(f"uniform '{uid}': mesh asset or its source ({asset.name}, {src.name}) missing; "
                                 "run tosdg import-model <model.glb>")
            mesh = read_glb(asset)
            errs = check_against_skeleton(mesh, self.skeleton)
            digest = spec_sha256(json.loads(src.read_text(encoding="utf-8")))
            if "spec" in m and mesh.info.get("spec_sha256") != digest:
                errs.append(f"{asset.name} is stale: it was built from a different spec (run tosdg import-model <model.glb>)")
            if "fit" in m and mesh.info.get("fit_sha256") != digest:
                errs.append(f"{asset.name} is stale: it was imported with a different fit (run tosdg import-model <model.glb>)")
            missing = sorted({x["region"] for x in mesh.materials if x.get("texture") is None} - set(u["proxy_colors"]))
            if missing:
                errs.append(f"uniform has no colour for mesh regions {missing}")
            if errs:
                raise ValueError(f"uniform '{uid}': " + "; ".join(errs))
            out[uid] = mesh
        return out

    def _hash_assets(self) -> dict[str, str]:
        files = [p for p in self.profile_dir.rglob("*.json")] + sorted(self.profile_dir.rglob("*.glb"))
        if self.podium_prop is None:                      # the podium model only counts as an asset of datasets that use it
            files = [p for p in files if p.name != podiummod.prop_path(".").name]
        files += [self.skeleton.source] + sorted((self.root / "poses").rglob("*.json"))
        out = {p.relative_to(self.root).as_posix(): sha256_file(p) for p in files}       # "/" on every OS: the manifest is portable
        out["backgrounds/index.json"] = sha256_file(self.backgrounds.index_path)
        if self.absent_backgrounds:        # only then: a dataset made with every photo keeps the hash it always had
            out["backgrounds/available"] = hashlib.sha256(
                "\n".join(sorted(b["id"] for b in self.backgrounds.index["backgrounds"])).encode()).hexdigest()
        return out

    def weights(self, key: str, items: dict) -> dict[str, float]:
        w = self.config.get(key)
        return dict(w) if w else {k: float(v.get("weight", 1.0)) for k, v in items.items()}


@dataclass
class Scene:
    seed: int
    gesture: dict
    sampled_pose: SampledPose
    uniform_id: str
    officer_position: np.ndarray
    officer_yaw_deg: float
    fk_world: FKResult
    camera: Camera
    camera_profile: dict
    camera_placement: dict
    background: dict
    backplate: Image.Image
    composition: dict
    lighting: dict
    uniform_brightness: float = 1.0      # multiplier on the cloth albedo (mesh backend); drawn from its own RNG stream
    baton: dict | None = None            # normalised baton spec + "mode" when a baton is held
    podium: dict | None = None           # {"diameter_cm", "depth_cm", "rim_cm", "sole_z_cm", "lift_cm"} when the officer stands on the traffic podium
    ground_check: dict | None = None     # {"status": ok | shifted | not_visible | no_mask, "points"}: feet on the backplate's road/sidewalk
    size_reference: dict | None = None   # config size_reference auto: {"mode": "calibrated", ...} (how the photo was scaled)
    junction: dict | None = None         # officer_yaw_mode junction: junction type, road directions in the officer's frame, sides (junction.py)


def build_scene(ctx: GenerationContext, seed: int) -> Scene:
    cfg = ctx.config
    W, H = cfg["resolution"]

    gid = rngmod.weighted_choice(rngmod.stream(seed, "gesture"), ctx.weights("gesture_weights", ctx.gestures.gestures))
    # The camera profile and the photo are chosen first (each stream draws the same values in any order): at a roundabout the
    # officer may hold the baton, which changes the pose.
    cam_rng = rngmod.stream(seed, "camera")
    cp_id = rngmod.weighted_choice(cam_rng, ctx.weights("camera_profile_weights", ctx.camera_profiles))
    cam_profile = ctx.camera_profiles[cp_id]
    if cfg.get("camera_overrides"):              # fixed ranges for some placement values (Studio locks); the draws stay the same
        cam_profile = {**cam_profile, **cfg["camera_overrides"]}
        if "sensor_width_mm" in cfg["camera_overrides"]:                              # a size, not a range
            cam_profile["sensor_width_mm"] = cfg["camera_overrides"]["sensor_width_mm"][0]
    bg_rng = rngmod.stream(seed, "background")   # its own stream: choosing the photo before the camera changes no draw
    bg = ctx.backgrounds.select(bg_rng, cam_profile.get("background_tags_any"))
    junction = bg.get("junction") if ctx.junction_mode else None
    baton, on_podium = ctx.baton, None
    if junction and junction["type"] == "roundabout":
        # officer_yaw_mode junction, at a roundabout: podium and held baton drawn on the new stream "roundabout"
        rr = rngmod.stream(seed, "roundabout")
        hold, stand = rr.random() < ctx.roundabout["baton_probability"], rr.random() < ctx.roundabout["podium_probability"]
        baton = (ctx.baton or batonmod.normalise({**batonmod.DEFAULT_BATON, "enabled": True})) if hold else None
        on_podium = stand and ctx.podium_prop is not None
    sp = ctx.gestures.sample(gid, ctx.skeleton, rngmod.stream(seed, "pose"), cfg.get("max_pose_tries", 200), baton)
    uniform_id = rngmod.weighted_choice(rngmod.stream(seed, "uniform"), ctx.weights("uniform_weights", ctx.uniforms))

    # New named stream, drawn only when the option is on: every other stream is unchanged.
    podium = None
    if on_podium if on_podium is not None else (ctx.podium_prop is not None and ctx.podium_probability > 0
                                                and rngmod.stream(seed, "podium").random() < ctx.podium_probability):
        sole = podiummod.sole_level(ctx.meshes[uniform_id], ctx.skeleton)
        podium = {**podiummod.size(ctx.podium_prop), "sole_z_cm": round(sole, 3), "lift_cm": round(podiummod.lift_cm(ctx.podium_prop, sole), 3)}
    officer_pos = np.zeros(3)
    if podium:                                   # the podium stands on the ground (z = 0); the officer stands on top of it
        officer_pos[2] = podium["lift_cm"]
    officer_yaw = rngmod.uniform(rngmod.stream(seed, "officer_yaw"), cfg.get("officer_yaw_deg", [0, 0]))
    fk_world = ctx.skeleton.fk(sp.pose).transformed(axis_rotation("z", officer_yaw), officer_pos)

    mode = cfg.get("size_reference", "horizon")
    overrides = cfg.get("camera_overrides") or {}
    cal = ctx.calibrations.get(bg["id"]) if mode == "auto" else None
    if cal is not None and cal["method"] == "none":
        raise SizeReferenceError(f"{bg['id']}: its camera is unknown ({cal['reason']})")
    # auto: the virtual camera stands as high as the photo's camera, pitched to the photo's horizon, and moves along its
    # line of sight to a distance where the officer fits; a Studio lock of the camera wins (then only the horizon is aligned)
    calibrated = cal is not None and not ({"height_cm", "distance_cm"} & set(overrides))
    if cal is not None and "height_cm" not in overrides:
        cam_profile = {**cam_profile, "height_cm": [cal["camera_height_cm"]] * 2}
    min_subject = None
    if cfg.get("min_officer_height_px") and not calibrated:   # officer at least this tall: cap the camera distance (same draws)
        z = np.array([p[2] for p in fk_world.positions.values()])
        min_subject = (float(z.max() - min(z.min(), officer_pos[2])) + HEAD_TOP_CM, cfg["min_officer_height_px"] * SIZE_MARGIN)
    camera, placement = sample_camera(cam_profile, cam_rng, officer_pos * [1, 1, 0], officer_yaw, W, H, min_subject)   # camera height / aim height are measured from the ground

    # New named stream: the lens (distortion, black corners) of the chosen camera profile; every other stream is unchanged.
    # lens_distortion false: pinhole images, the lens stream is not drawn.
    lens = sample_lens(cam_profile.get("lens"), rngmod.stream(seed, "lens")) if cfg.get("lens_distortion", True) else None
    camera.lens = lens.fitted(W, H, camera.fx) if lens else None             # too strong for this field of view: weakened until no frame corner goes black

    junction_report = None
    at_junction = bool(junction) and calibrated
    if ctx.junction_mode and not at_junction:
        # junction mode, a photo without a junction (or a locked camera): along / across the road, the facing whose sides
        # are the road for the gesture's directions (junction.choose_yaw, kind "straight"); stream "road_yaw"
        road = camera.yaw + float(bg.get("road_yaw_deg", 0.0))
        jit = cfg.get("road_yaw_jitter_deg", [-ROAD_YAW_JITTER_DEG, ROAD_YAW_JITTER_DEG])
        new_yaw, junction_report = junctionmod.choose_yaw("straight", [road, road + 180.0], ctx.gestures.gestures[gid],
                                                           camera.position[:2], officer_pos[:2], cam_profile["azimuth_deg"],
                                                           rngmod.stream(seed, "road_yaw"), jit)
        placement = {**placement, "azimuth_deg": (placement["azimuth_deg"] - (new_yaw - officer_yaw) + 180.0) % 360.0 - 180.0}
        officer_yaw = new_yaw
        fk_world = ctx.skeleton.fk(sp.pose).transformed(axis_rotation("z", officer_yaw), officer_pos)
    elif cfg.get("officer_yaw_mode") == "road":
        # The officer faces along or across the traffic: 0 / 90 / 180 / 270 degrees from the road direction (the photo's view
        # direction, turned by the background's road_yaw_deg), plus a small jitter. New stream "road_yaw"; the camera stays
        # where it is and the officer turns, so the camera azimuth (in the officer's frame) is recomputed.
        ry = rngmod.stream(seed, "road_yaw")
        quarter = int(ry.integers(4))
        jitter = rngmod.uniform(ry, cfg.get("road_yaw_jitter_deg", [-ROAD_YAW_JITTER_DEG, ROAD_YAW_JITTER_DEG]))
        new_yaw = camera.yaw + float(bg.get("road_yaw_deg", 0.0)) + 90.0 * quarter + jitter
        new_yaw = (new_yaw + 180.0) % 360.0 - 180.0
        placement = {**placement, "azimuth_deg": (placement["azimuth_deg"] - (new_yaw - officer_yaw) + 180.0) % 360.0 - 180.0,
                     "road_quarter": quarter}
        officer_yaw = new_yaw
        fk_world = ctx.skeleton.fk(sp.pose).transformed(axis_rotation("z", officer_yaw), officer_pos)

    size_reference = None
    with Image.open(ctx.backgrounds.path(bg)) as im:
        lens_on = camera.lens is not None and camera.lens.active
        # with a lens the photo is distorted together with the officer: it is composed for the larger pinhole canvas the lens needs
        plate_cam = canvas_camera(camera) if lens_on else camera
        Wc, Hc = (plate_cam.width, plate_cam.height) if lens_on else (W, H)
        ground_check = {"status": "no_mask", "points": 0}
        mask = ctx.backgrounds.ground_mask(bg)
        composition = None
        if at_junction:
            (camera, placement, composition, ground_check, size_reference, officer_pos, officer_yaw, fk_world,
             junction_report) = _at_junction(ctx, seed, cal, bg, im.size, camera, placement, cam_profile, sp,
                                             ctx.gestures.gestures[gid], podium, mask, lens_on)
        elif calibrated:
            camera, placement, composition, ground_check, size_reference = _calibrated(
                ctx, seed, cal, bg, im.size, camera, placement, cam_profile, fk_world, officer_pos, podium, mask, lens_on)
        else:
            plate = bg if cal is None else {**bg, "horizon_y_frac": cal["horizon_y_frac"]}       # auto with a locked camera
            composition = plan_backplate(im.size, plate, pinhole_horizon_row(camera, Hc) if lens_on else camera.horizon_v, Wc, Hc, bg_rng)
            if cal is not None:                    # auto with a locked camera: only the photo's horizon is aligned
                if not composition["horizon_aligned"]:
                    raise SizeReferenceError(f"{bg['id']}: the photo's horizon cannot be aligned with the locked camera")
                size_reference = {**_cal_report(cal), "camera_locked": True}
            # feet on the photo's road / sidewalk: the crop may move sideways (new stream "ground", drawn only then), or the
            # sample is rejected (groundmod.PlacementError). Backgrounds without a ground mask are not checked.
            if mask is not None:
                uv, _ = plate_cam.project_pinhole(groundmod.footprint_points(fk_world, officer_pos, podium))
                composition, ground_check = groundmod.fit_feet(mask, im.size, composition, uv, Wc, Hc,
                                                               lambda: rngmod.stream(seed, "ground"))
        backplate = crop_backplate(im, composition, Wc, Hc)
        if lens_on:
            composition = {**composition, "camera_horizon_v": float(camera.horizon_v)}      # reported: the horizon row in the final, distorted image

    lighting = sample_lighting(cfg.get("lighting", {}), rngmod.stream(seed, "lighting"))
    if cfg.get("match_backplate_light"):          # no draw: the officer as bright as the photo around them (night, dusk, shade)
        pc = canvas_camera(camera) if (camera.lens is not None and camera.lens.active) else camera
        lighting = match_light(lighting, backplate, pc.project_pinhole(np.array(list(fk_world.positions.values())))[0], bg)
    lighting = {**lighting, "exposure_ev": float(np.clip(lighting["exposure_ev"], *EXPOSURE_RANGE))}
    # New named stream: existing streams (gesture, pose, camera, ...) draw exactly the same values as before for any seed.
    brightness = rngmod.uniform(rngmod.stream(seed, "uniform_brightness"), cfg.get("uniform_brightness_range", [1.0, 1.0]))
    return Scene(seed, ctx.gestures.gestures[gid], sp, uniform_id, officer_pos, officer_yaw, fk_world,
                 camera, cam_profile, placement, bg, backplate, composition, lighting, brightness,
                 {**baton, "mode": sp.baton["mode"], "closure": sp.baton["closure"], "thumb_closure": sp.baton["thumb_closure"]} if baton else None,
                 podium, ground_check, size_reference, junction_report)


def _cal_report(cal: dict) -> dict:
    out = {"mode": "calibrated", "method": cal["method"], "camera_height_cm": round(float(cal["camera_height_cm"]), 2),
           "horizon_y_frac": round(float(cal["horizon_y_frac"]), 5), "objects_used": int(cal["objects_used"])}
    if "camera_height_from" in cal:
        out["camera_height_from"] = cal["camera_height_from"]
    return out


def _calibrated(ctx: GenerationContext, seed: int, cal: dict, bg: dict, plate_size, camera: Camera, placement: dict,
                cam_profile: dict, fk_world: FKResult, officer_pos: np.ndarray, podium: dict | None, mask, lens_on: bool):
    """size_reference auto: the virtual camera as the photo's camera (calibrate.py). It is cal["camera_height_cm"] high; the
    photo is shown whole (zoomed only when the officer cannot be big enough otherwise) with a vertical crop drawn among those
    that let the officer fit, and the camera pitched so that its horizon is the photo's horizon row there. The camera then
    moves along its line of sight to a distance in the band where the officer is in the frame and big enough (the drawn
    distance keeps its place in the profile's range, inside that band), and the horizontal crop keeps the officer's feet on
    the road and the officer off any object nearer to the camera. Same draws on the camera stream; new stream "calibration".
    Returns (camera, placement, composition, ground_check, size_reference)."""
    cfg = ctx.config
    W, H = cfg["resolution"]
    sr = rngmod.stream(seed, "calibration")
    hc, v0f = float(cal["camera_height_cm"]), float(cal["horizon_y_frac"])
    pc = canvas_camera(camera) if lens_on else camera
    Wc, Hc, fy = pc.width, pc.height, pc.fy
    top0 = (Hc - H) / 2.0                                        # the final frame inside the (lens) canvas
    band = (top0 + FRAME_MARGIN * H, top0 + (1 - FRAME_MARGIN) * H)
    z = np.array([p[2] for p in fk_world.positions.values()])
    top_cm = float(z.max()) + HEAD_TOP_CM                         # from the ground (a podium lift is in the positions)
    min_px = SIZE_MARGIN * max(float(ctx.validation["min_bbox_height_px"]), float(cfg.get("min_officer_height_px") or 0))
    bw, bh = plate_size
    cover = max(Wc / bw, Hc / bh)
    for zoom in CALIBRATED_ZOOMS:
        scale = cover * zoom
        sw, sh = groundmod.scaled_size(plate_size, scale, Wc, Hc)
        oys = np.arange(Hc - sh, 1)
        ok = np.array([calmod.feasible_gap(v0f * sh + oy, hc, top_cm, band, min_px) is not None for oy in oys])
        if ok.any():
            break
    else:
        raise SizeReferenceError(f"{bg['id']}: at the photo's camera ({hc:.0f} cm high, horizon {v0f:.2f}) the officer cannot "
                                 f"be {min_px:.0f} px tall inside the frame at any crop up to {CALIBRATED_ZOOMS[-1]}x")
    oy = int(oys[ok][int(sr.integers(int(ok.sum())))])
    v0 = v0f * sh + oy
    g_lo, g_hi = calmod.feasible_gap(v0, hc, top_cm, band, min_px)
    pitch = calmod.pitch_for_horizon(v0, Hc / 2.0, fy)
    d_near, d_far = (calmod.distance_for_gap(g, hc, pitch, fy) for g in (g_hi, g_lo))
    lo, hi = (float(x) for x in cam_profile["distance_cm"])
    a, b = max(lo, d_near), min(hi, d_far)
    if a > b:                                                    # the profile's distances do not fit this photo: the photo wins
        a, b = d_near, d_far
    t = (placement["distance_cm"] - lo) / (hi - lo) if hi > lo else 0.5
    d = a + t * (b - a)
    ground = officer_pos * [1, 1, 0]
    e = np.asarray(camera.position, float) - ground
    e2 = e[:2] / max(1e-9, float(np.hypot(*e[:2])))
    camera = Camera(camera.profile, (float(ground[0] + e2[0] * d), float(ground[1] + e2[1] * d), hc), pitch, camera.yaw,
                    camera.roll, camera.focal_length_mm, camera.sensor_width_mm, camera.width, camera.height, camera.lens)
    placement = {**placement, "distance_cm": d, "height_cm": hc}
    pc = canvas_camera(camera) if lens_on else camera

    pts = np.array(list(fk_world.positions.values()))
    u_all, _ = pc.project_pinhole(pts)
    feet_uv, _ = pc.project_pinhole(groundmod.footprint_points(fk_world, officer_pos, podium))
    g_uv, _ = pc.project_pinhole(ground[None, :])
    h_px = float(g_uv[0, 1] - np.nanmin(u_all[:, 1]))
    cands = np.arange(-(sw - Wc), 1)
    good = calmod.free_offsets(cands, scale, oy, calmod.reference_objects(bg), float(g_uv[0, 1]),
                               (float(np.nanmin(u_all[:, 1])), float(g_uv[0, 1])),
                               (float(np.nanmin(u_all[:, 0])), float(np.nanmax(u_all[:, 0]))), h_px, H, feet_uv, mask,
                               (sw, sh), Wc, Hc)
    if not len(good):
        raise groundmod.PlacementError(f"{bg['id']}: no horizontal crop puts the officer on the road clear of nearer people / vehicles")
    ox = int(good[int(sr.integers(len(good)))])
    composition = {"scale": float(scale), "offset_px": [ox, oy], "horizon_aligned": True,
                   "camera_horizon_v": float(pinhole_horizon_row(camera, Hc) if lens_on else camera.horizon_v)}
    vis = np.isfinite(feet_uv).all(axis=1) & (feet_uv[:, 0] >= 0) & (feet_uv[:, 0] < Wc) & (feet_uv[:, 1] >= 0) & (feet_uv[:, 1] < Hc)
    ground_check = {"status": ("ok" if vis.any() else "not_visible") if mask is not None else "no_mask", "points": int(vis.sum())}
    report = {**_cal_report(cal), "pitch_deg": round(pitch, 4), "zoom": zoom}
    return camera, placement, composition, ground_check, report


def _at_junction(ctx: GenerationContext, seed: int, cal: dict, bg: dict, plate_size, camera: Camera, placement: dict,
                 cam_profile: dict, sp: SampledPose, gesture: dict, podium: dict | None, mask, lens_on: bool):
    """officer_yaw_mode junction, a photo with a junction (junction.py): the officer stands within JITTER_CM of the junction's
    middle, facing as the junction type and the gesture want. The virtual camera is the photo's camera (calibrate.py): its
    height, and the pitch that puts the photo's horizon on the composed frame's horizon row; the photo is shown at the
    smallest zoom where the officer at that spot is big enough and fits the frame, with a crop drawn among those that keep them
    inside; the camera then stands where the junction's middle is seen at its pixel. Roads: arms_deg turned by the camera yaw.
    New stream "junction". Returns (camera, placement, composition, ground_check, size_reference, officer_pos, officer_yaw,
    fk_world, junction report)."""
    cfg = ctx.config
    W, H = cfg["resolution"]
    jr = rngmod.stream(seed, "junction")
    j = bg["junction"]
    hc, v0f = float(cal["camera_height_cm"]), float(cal["horizon_y_frac"])
    pc = canvas_camera(camera) if lens_on else camera
    Wc, Hc, fy = pc.width, pc.height, pc.fy
    top0, left0 = (Hc - H) / 2.0, (Wc - W) / 2.0                 # the final frame inside the (lens) canvas
    bv = (top0 + FRAME_MARGIN * H, top0 + (1 - FRAME_MARGIN) * H)
    bu = (left0 + FRAME_MARGIN * W, left0 + (1 - FRAME_MARGIN) * W)
    fk0 = ctx.skeleton.fk(sp.pose)
    lift = float(podium["lift_cm"]) if podium else 0.0
    top_cm = float(max(p[2] for p in fk0.positions.values())) + HEAD_TOP_CM + lift
    min_px = SIZE_MARGIN * max(float(ctx.validation["min_bbox_height_px"]), float(cfg.get("min_officer_height_px") or 0))
    bw, bh = plate_size
    uc, vc = (float(x) for x in j["center"])
    if vc <= v0f * bh + 1.0:
        raise SizeReferenceError(f"{bg['id']}: the junction's middle is not below the photo's horizon ({v0f:.3f})")
    cover = max(Wc / bw, Hc / bh)
    for zoom in CALIBRATED_ZOOMS:
        scale = cover * zoom
        sw, sh = groundmod.scaled_size(plate_size, scale, Wc, Hc)
        h = scale * (vc - v0f * bh) * top_cm / hc                    # officer height in the frame (linear model)
        oy_lo, oy_hi = max(Hc - sh, bv[0] + h - vc * scale), min(0.0, bv[1] - vc * scale)
        ox_lo, ox_hi = max(Wc - sw, bu[0] + 0.45 * h - uc * scale), min(0.0, bu[1] - 0.45 * h - uc * scale)
        if h >= min_px and np.ceil(oy_lo) <= np.floor(oy_hi) and np.ceil(ox_lo) <= np.floor(ox_hi):
            break
    else:
        raise SizeReferenceError(f"{bg['id']}: an officer in the middle of the junction cannot be {min_px:.0f} px tall inside "
                                 f"the frame at any crop up to {CALIBRATED_ZOOMS[-1]}x")
    oy = int(jr.integers(int(np.ceil(oy_lo)), int(np.floor(oy_hi)) + 1))
    ox = int(jr.integers(int(np.ceil(ox_lo)), int(np.floor(ox_hi)) + 1))
    pitch = calmod.pitch_for_horizon(v0f * sh + oy, Hc / 2.0, fy)

    def canvas_of(pt):
        return (float(pt[0]) * scale + ox, float(pt[1]) * scale + oy)

    probe = Camera(camera.profile, (0.0, 0.0, hc), pitch, camera.yaw, camera.roll, camera.focal_length_mm, camera.sensor_width_mm,
                   camera.width, camera.height, camera.lens)
    g = junctionmod.ground_point(canvas_camera(probe) if lens_on else probe, canvas_of(j["center"]))
    if g is None:
        raise SizeReferenceError(f"{bg['id']}: the junction's middle is not on the ground plane of the camera")
    camera = Camera(camera.profile, (float(-g[0]), float(-g[1]), hc), pitch, camera.yaw, camera.roll, camera.focal_length_mm,
                    camera.sensor_width_mm, camera.width, camera.height, camera.lens)        # the junction's middle at the world origin
    pc = canvas_camera(camera) if lens_on else camera
    arms = [float(camera.yaw) + float(a) for a in j["arms_deg"]]           # the photo's view direction = the camera's
    objs = calmod.reference_objects(bg)
    jit = cfg.get("road_yaw_jitter_deg", [-ROAD_YAW_JITTER_DEG, ROAD_YAW_JITTER_DEG])
    for _ in range(JUNCTION_TRIES):
        r, t = junctionmod.JITTER_CM * np.sqrt(jr.random()), jr.uniform(0.0, 2 * np.pi)
        officer_pos = np.array([r * np.cos(t), r * np.sin(t), lift])
        yaw, report = junctionmod.choose_yaw(j["type"], arms, gesture, camera.position[:2], officer_pos[:2],
                                             cam_profile["azimuth_deg"], jr, jit)
        fk_world = fk0.transformed(axis_rotation("z", yaw), officer_pos)
        u_all, _ = pc.project_pinhole(np.array(list(fk_world.positions.values())))
        feet_uv, _ = pc.project_pinhole(groundmod.footprint_points(fk_world, officer_pos, podium))
        g_uv, _ = pc.project_pinhole(np.array([[officer_pos[0], officer_pos[1], 0.0]]))
        top_v = float(np.nanmin(u_all[:, 1]))
        ok = calmod.free_offsets(np.array([ox]), scale, oy, objs, float(g_uv[0, 1]), (top_v, float(g_uv[0, 1])),
                                 (float(np.nanmin(u_all[:, 0])), float(np.nanmax(u_all[:, 0]))), float(g_uv[0, 1]) - top_v, H,
                                 feet_uv, mask, (sw, sh), Wc, Hc)
        if len(ok):
            break
    else:
        raise groundmod.PlacementError(f"{bg['id']}: no spot near the junction's middle is on the road and clear of nearer "
                                       "people / vehicles")
    eye = np.asarray(camera.position, float)
    az = float(np.degrees(np.arctan2(eye[1] - officer_pos[1], eye[0] - officer_pos[0])) - yaw)
    placement = {**placement, "distance_cm": float(np.hypot(*(eye[:2] - officer_pos[:2]))), "height_cm": hc,
                 "azimuth_deg": (az + 180.0) % 360.0 - 180.0}
    composition = {"scale": float(scale), "offset_px": [ox, oy], "horizon_aligned": True,
                   "camera_horizon_v": float(pinhole_horizon_row(camera, Hc) if lens_on else camera.horizon_v)}
    vis = np.isfinite(feet_uv).all(axis=1) & (feet_uv[:, 0] >= 0) & (feet_uv[:, 0] < Wc) & (feet_uv[:, 1] >= 0) & (feet_uv[:, 1] < Hc)
    ground_check = {"status": ("ok" if vis.any() else "not_visible") if mask is not None else "no_mask", "points": int(vis.sum())}
    size_reference = {**_cal_report(cal), "pitch_deg": round(pitch, 4), "zoom": zoom}
    report = {**report, "at_center": True, "offset_cm": [round(float(officer_pos[0]), 2), round(float(officer_pos[1]), 2)],
              "roads": len(arms)}
    return camera, placement, composition, ground_check, size_reference, officer_pos, yaw, fk_world, report


def match_light(lighting: dict, backplate: Image.Image, officer_uv: np.ndarray, bg: dict) -> dict:
    """config match_backplate_light: darken the officer to the photo. L = geometric mean of the log-average linear luminance
    (the "key" of tone mapping: lamps, signs and a bright sky do not dominate it) of the composed photo in a box twice the
    officer's size around them and of the whole photo, sets an exposure change log2(L / LIGHT_REF) in [LIGHT_MIN_EV, 0] (never
    brighter than the drawn lighting); photos tagged night also turn the sun into a weak warm street lamp and dim the sky
    light. No RNG draw."""
    a = np.asarray(backplate.convert("RGB"), np.float32) / 255.0
    Hh, Ww = a.shape[:2]
    uv = officer_uv[np.isfinite(officer_uv).all(axis=1)]
    if len(uv):
        (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
        du, dv = max(8.0, u1 - u0) / 2.0, max(8.0, v1 - v0) / 2.0
        box = a[int(np.clip(v0 - dv, 0, Hh - 1)):int(np.clip(v1 + dv, 1, Hh)), int(np.clip(u0 - du, 0, Ww - 1)):int(np.clip(u1 + du, 1, Ww))]
    else:
        box = a
    def luminance(x):
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        return float(np.exp(np.log(lin @ np.array([0.2126, 0.7152, 0.0722], np.float32) + 1e-3).mean())) if x.size else LIGHT_REF

    lum = float(np.sqrt(max(luminance(box), 1e-6) * max(luminance(a), 1e-6)))
    ev = float(np.clip(np.log2(max(lum, 1e-4) / LIGHT_REF), LIGHT_MIN_EV, 0.0))
    out = {**lighting, "exposure_ev": float(lighting["exposure_ev"]) + ev, "backplate_luminance": round(lum, 5),
           "backplate_exposure_ev": round(ev, 4)}
    if "night" in bg.get("tags", []):
        out["sun_intensity"] = float(lighting["sun_intensity"]) * NIGHT_SUN
        out["ambient_intensity"] = float(lighting["ambient_intensity"]) * NIGHT_AMBIENT
        out["color_temperature_k"] = min(float(lighting["color_temperature_k"]), NIGHT_KELVIN)
        out["sun_color_rgb"] = kelvin_to_rgb(out["color_temperature_k"])
    return out
