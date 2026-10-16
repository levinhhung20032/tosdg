"""Local HTTP server for the tosdg studio (stdlib only, bound to 127.0.0.1).

Guards: loopback bind, Host-header allow-list (DNS rebinding), POSTs must carry `X-Studio: 1`
(a cross-site form cannot set it), bounded request sizes, files served only from the workspace.

Contents:
  constants: STATIC, ALLOWED_HOSTS, MAX_JSON_BYTES, RANDOM_DIR, RANDOM_MAX_SAMPLES, ID_RE, PREVIEW_LIGHT, FINGER_NAMES, MAX_MODEL_UPLOAD_BYTES, MODEL_LAYER_MAX_W
  class ApiError
  class StudioApp
  make_handler
  class Presence — Which browser pages are open
  class StudioServer — HTTP server that owns its port alone
  WorkspaceInUse
  lock_workspace — One Studio per workspace
  create_server
  serve
  (+ 1 private helper)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from ..backgrounds import compose_backplate
from ..cli import COCO_EDGES
from ..gestures import GestureLibrary, pose_validity
from ..groundtruth import bbox_from_mask
from .. import baton as batonmod
from .. import podium as podiummod
from ..poses import PoseLibrary, merge, mirror_name, mirror_pose, save_pose
from .. import mesh_renderer
from ..camera import Camera
from ..lens import Lens, canvas_size, distort_plate, layer_with_lens, pinhole_horizon_row, render_with_lens
from ..render_proxy import render
from ..schema import SchemaError, load_json, repo_root
from ..skeleton import Skeleton
from ..uniform_mesh import check_against_skeleton, read_glb
from . import core
from .workspace import MAX_UPLOAD_BYTES, Workspace, WorkspaceError
from ..scene import CAMERA_OVERRIDE_KEYS

STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
MAX_JSON_BYTES = 4 * 1024 * 1024
RANDOM_DIR = repo_root() / "out" / "studio_random"   # batches of the Studio's "Random generate" panel (git-ignored)
RANDOM_MAX_SAMPLES = 500
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
PREVIEW_LIGHT = {"sun_direction": [0.6, 0.3, 0.74], "sun_color_rgb": [1, 1, 1], "sun_intensity": 0.9,
                 "ambient_intensity": 0.4, "exposure_ev": 0.0}
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
MAX_MODEL_UPLOAD_BYTES = 200 * 1024 * 1024    # a person model (GLB with its texture)
MODEL_LAYER_MAX_W = 960      # the 3D uniform layer of the camera view is rendered at most this wide and scaled up by the page


class ApiError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


class StudioApp:
    def __init__(self, workspace: str | Path | None = None, profile: str = "VN", character: str = "tripo_v1",
                 library: str | Path | None = None):
        self.presence = Presence()                      # open browser pages (for --exit-on-close)
        self.boot = time.time_ns() // 1_000_000         # id of this run (start time, ms): a page of an older run closes itself
        self.root = repo_root()
        self.library_dir = Path(library) if library else self.root / "data" / "backgrounds"
        self.ws = Workspace(workspace or self.root / "data" / "studio", self.library_dir)   # background photos live in the library
        self.sk = Skeleton.load(self.root / "characters" / character / "skeleton.json")
        self.poses = PoseLibrary(self.root / "poses")
        pdir = self.root / "profiles" / profile
        prof = load_json(pdir / "profile.json", "profile")
        files = prof["files"]
        self.profile, self.profile_dir, self.uniforms_file = prof, pdir, files["uniforms"]
        self.gestures = GestureLibrary(load_json(pdir / files["gestures"], "gestures"), self.poses)
        self.camera_profiles = load_json(pdir / files["camera_profiles"], "camera_profiles")["profiles"]
        self.uniforms = {u["id"]: u for u in load_json(pdir / files["uniforms"], "uniforms")["uniforms"]}
        self.lock = threading.Lock()
        self.random_job: dict | None = None             # the Studio's "Random generate" batch (one at a time)
        self._mesh_cache: dict = {}

    def mesh_for(self, uni: dict):
        """The skinned uniform mesh of a profile uniform (cached until the GLB changes), or None if it has no mesh."""
        if "mesh" not in uni:
            return None
        path = self.profile_dir / uni["mesh"]["asset"]
        if not path.exists():
            raise ApiError(f"mesh file {path.name} is missing; upload the model again", 500)
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self._mesh_cache:
            mesh = read_glb(path)
            errs = check_against_skeleton(mesh, self.sk)
            if errs:
                raise ApiError(f"mesh {path.name} does not fit the skeleton: {errs}", 500)
            self._mesh_cache = {k: v for k, v in self._mesh_cache.items() if k[0] != key[0]}     # drop an older version of this file, keep the other models
            self._mesh_cache[key] = mesh
        return self._mesh_cache[key]

    # ----------------------------------------------------------------------------- helpers
    def default_camera(self) -> dict:
        """The camera Studio starts from ("Default" in the camera list): core.DEFAULT_CAMERA, no distortion. The black border is on auto,
        so as soon as a lens is dialled in it follows k1, k2 and the focal length."""
        return {**core.DEFAULT_CAMERA, "lens_border_auto": 1.0}

    def default_pose(self) -> dict:
        return merge(self.poses.body_pose("neutral_stand"), self.poses.hand_pose("relaxed", "l"),
                     self.poses.hand_pose("relaxed", "r"))

    def _pose(self, body: dict) -> dict:
        pose = body.get("pose")
        if not isinstance(pose, dict):
            raise ApiError("'pose' must be an object")
        clean = {}
        for jn, dofs in pose.items():
            if jn not in self.sk.joints or not isinstance(dofs, dict):
                raise ApiError(f"unknown joint '{jn}'")
            clean[jn] = {}
            for dn, v in dofs.items():
                if self.sk.joints[jn].dof(dn) is None or not isinstance(v, (int, float)) or not np.isfinite(v):
                    raise ApiError(f"bad dof {jn}.{dn}")
                if abs(v) > 360:                  # past a joint limit is allowed (Pose warnings shows it); past a full turn is a broken client
                    raise ApiError(f"dof {jn}.{dn}={v} is not an angle in [-360, 360] degrees")
                clean[jn][dn] = float(v)
        return clean

    def _baton(self, body: dict):
        """(spec, mode) of the baton option, (None, None) when off."""
        b = body.get("baton")
        if not isinstance(b, dict) or not b.get("enabled"):
            return None, None
        try:
            spec = batonmod.normalise({k: b[k] for k in ("enabled", "hand", "length_cm", "radius_cm", "handle_cm") if k in b})
        except ValueError as e:
            raise ApiError(str(e)) from None
        mode = b.get("mode", "across")
        if mode not in batonmod.MODES:
            raise ApiError("baton mode must be 'across' or 'along'")
        return spec, mode

    def _uniform_for(self, body: dict) -> dict:
        """The profile uniform to draw: the requested one, else the first."""
        return self.uniforms.get(body.get("profile_uniform")) or next(iter(self.uniforms.values()))

    def baton_prop(self):
        """The profile's 3D baton model, or None (a plain cylinder is drawn then)."""
        pp = batonmod.prop_path(self.profile_dir)
        return batonmod.load_prop(pp) if pp.exists() else None

    def _with_podium(self, mesh, body: dict):
        """The mesh standing on the traffic podium when the Studio asks for it (preview only; the generator does not use it)."""
        if not body.get("podium"):
            return mesh
        pp = podiummod.prop_path(self.profile_dir)
        if not pp.exists():
            raise ApiError("this profile has no props/podium.glb", 404)
        return podiummod.mesh_with_podium(mesh, podiummod.load(pp), podiummod.sole_level(mesh, self.sk))

    def _gripped(self, pose: dict, body: dict):
        """(pose with the baton hand closed around the baton, spec incl. mode/closure or None)."""
        spec, mode = self._baton(body)
        if spec is None:
            return pose, None
        gp, info = batonmod.apply_grip(self.sk, pose, spec, mode, self.poses.hand_pose("fist", spec["hand"]), keep_wrist=True)   # a wrist turned in Studio stays turned
        return gp, {**spec, "mode": mode, "closure": info["closure"], "thumb_closure": info["thumb_closure"]}

    @staticmethod
    def _brightness(body) -> float:
        b = body.get("brightness", 1.0)
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not (0.3 <= b <= 2.0):
            raise ApiError("brightness must be a number between 0.3 and 2.0")
        return float(b)

    @staticmethod
    def _num_dict(d, defaults: dict, name: str) -> dict:
        d = d if isinstance(d, dict) else {}
        out = dict(defaults)
        for k, v in d.items():
            if k in defaults:
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    raise ApiError(f"{name}.{k} must be a finite number")
                out[k] = float(v)
        return out

    def podium_lift(self, uni: dict) -> float | None:
        """How high the officer stands above the ground on the podium for this model (None: no podium file or no 3D mesh)."""
        pp = podiummod.prop_path(self.profile_dir)
        mesh = self.mesh_for(uni) if pp.exists() else None
        return None if mesh is None else round(podiummod.lift_cm(podiummod.load(pp), podiummod.sole_level(mesh, self.sk)), 2)

    def _scene_args(self, body: dict):
        officer = self._num_dict(body.get("officer"), core.DEFAULT_OFFICER, "officer")
        officer["z"] = (self.podium_lift(self._uniform_for(body)) or 0.0) if body.get("podium") else 0.0     # decided here, not by the client
        cam = self._num_dict(body.get("camera"), core.DEFAULT_CAMERA, "camera")
        cam["width"], cam["height"] = int(cam["width"]), int(cam["height"])
        if not (16 <= cam["width"] <= 4096 and 16 <= cam["height"] <= 4096):
            raise ApiError("resolution must be 16..4096 px")
        if not (1.0 <= cam["focal_length_mm"] <= 1000.0 and 1.0 <= cam["sensor_width_mm"] <= 100.0):
            raise ApiError("focal length / sensor width out of range")
        if not (core.MIN_DISTANCE_CM <= cam["distance_cm"] <= core.MAX_DISTANCE_CM):
            raise ApiError(f"distance must be {core.MIN_DISTANCE_CM:g}..{core.MAX_DISTANCE_CM:g} cm")
        if not (-1.0 <= cam["lens_k1"] <= 1.0 and -1.0 <= cam["lens_k2"] <= 1.0 and 0.0 <= cam["lens_circle"] <= 1.5 and 0.0 <= cam["lens_border_auto"] <= 1.0):
            raise ApiError("lens: k1 and k2 must be within -1..1 and the image circle within 0..1.5")
        return officer, cam

    # ----------------------------------------------------------------------------- API
    def api_init(self, _body) -> dict:
        sk = self.sk
        gestures = [{"id": g["id"], "label": g["label"], "description": g.get("description", ""),
                     "variants": [{"id": v["id"], "pose": v["pose"], "hands": v.get("hands", {})}
                                  for v in g["pose_variants"]]} for g in self.gestures.gestures.values()]
        return {
            "skeleton": {
                "joints": [{"name": j.name, "parent": j.parent, "virtual": j.virtual,
                            "dofs": [{k: d[k] for k in ("name", "min", "max", "axis", "sign")} for d in j.dofs]}
                           for j in sk.joints.values()],
                "keypoint_sets": {k: [{"name": p["name"], "joint": p["joint"], "offset": p["offset"]} for p in v]
                                  for k, v in sk.keypoint_sets.items()},
                "coco_edges": COCO_EDGES,
            },
            "poses": {"body": sorted(self.poses.body), "hands": sorted(self.poses.hands)},
            "gestures": gestures,
            "camera_profiles": self.camera_profiles,
            "profile_uniforms": self.profile_uniform_list(),
            "models": self.api_models({})["models"],
            "podium": podiummod.size(podiummod.load(podiummod.prop_path(self.profile_dir))) if podiummod.prop_path(self.profile_dir).exists() else None,
            "defaults": {"camera": self.default_camera(), "officer": core.DEFAULT_OFFICER, "pose": self.default_pose()},
            "boot": self.boot,
            "backgrounds": self.ws.list_backgrounds(),
            "state": self.ws.load_state(),
            "library_dir": str(self.library_dir),
            "gpu": self.gpu_info(),
        }

    def gpu_info(self) -> dict:
        """{"available": bool, "name" | "reason"}: whether the GPU mesh renderer runs on this machine (asked once)."""
        if not hasattr(self, "_gpu"):
            try:
                from ..render_gl import available
                ok, what = available()
            except ImportError as e:
                ok, what = False, f"moderngl is not installed ({e})"
            self._gpu = {"available": ok, ("name" if ok else "reason"): what}
        return self._gpu

    def api_eval(self, body) -> dict:
        officer, cam = self._scene_args(body)
        pose = self._pose(body)
        pose, bspec = self._gripped(pose, body)
        out = core.evaluate(self.sk, pose, officer, cam, baton=bspec)
        out["gesture_matches"] = core.match_gestures(self.sk, self.gestures.gestures, pose, cam["azimuth_deg"])
        return out

    def api_pose_preset(self, body) -> dict:
        """Compose a pose from library entries: body pose + optional hand poses per side."""
        bid = body.get("body")
        if bid not in self.poses.body:
            raise ApiError(f"unknown body pose '{bid}'")
        parts = [self.poses.body_pose(bid)]
        for side in ("l", "r"):
            h = (body.get("hands") or {}).get(side, "relaxed")
            if h not in self.poses.hands:
                raise ApiError(f"unknown hand pose '{h}'")
            parts.append(self.poses.hand_pose(h, side))
        return {"pose": merge(*parts)}

    def api_drag(self, body) -> dict:
        officer, cam_params = self._scene_args(body)
        pose = self._pose(body)
        mode = body.get("mode")
        cam = core.make_camera(cam_params, officer)
        target = body.get("target") or {}
        if mode == "officer":  # move the officer on the ground plane
            if target.get("type") == "camera":
                g = core.ground_target(cam, target["uv"], officer["z"])    # the plane the officer stands on (the podium top)
                if g is None:
                    return {"officer": officer, "pose": pose, "unreachable": True}
                new = g
            else:
                new = np.array(target["point"], float)
            officer = {**officer, "x": float(new[0]), "y": float(new[1])}
            return {"officer": officer, "pose": pose}
        if mode == "angle":  # turn one joint DOF with the Alt+click lever; physics rules in core.set_joint_angle
            joint, dof, value = body.get("joint"), body.get("dof"), body.get("value")
            if joint not in self.sk.joints or self.sk.joints[joint].dof(str(dof)) is None:
                raise ApiError(f"unknown joint/dof '{joint}.{dof}'")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ApiError("value must be a number")
            new_pose, used, why = core.set_joint_angle(self.sk, pose, joint, str(dof), float(value))
            gp, bspec = self._gripped(new_pose, body)                   # evaluate what is drawn: the hand closed on the baton, if any
            ev = core.evaluate(self.sk, gp, officer, cam_params, baton=bspec)
            ev["gesture_matches"] = core.match_gestures(self.sk, self.gestures.gestures, new_pose, cam_params["azimuth_deg"])
            return {"pose": new_pose, "eval": ev, "angle": {"joint": joint, "dof": dof, "value": used, "limited_by": why}}
        if mode != "joint":
            raise ApiError("mode must be 'joint', 'angle' or 'officer'")
        joint = body.get("joint")
        if joint not in self.sk.joints:
            raise ApiError(f"unknown joint '{joint}'")
        offset = body.get("offset") or [0.0, 0.0, 0.0]
        R, t = core.officer_transform(officer)
        cur_world = R @ self.sk.fk(pose).point(joint, offset) + t
        if target.get("type") == "camera":
            uv = [float(target["uv"][0]), float(target["uv"][1])]
            if float(cam.to_camera(cur_world)[0, 0]) <= 1.0:
                return {"pose": pose, "unreachable": True}
            plane = target.get("plane")
            if plane is None:
                world_t = core.camera_plane_target(cam, cur_world, uv)
            else:                                                       # Ctrl+drag: stay on the plane across this axis of the officer
                if plane not in ("x", "y", "z"):
                    raise ApiError("target.plane must be 'x', 'y' or 'z'")
                world_t = core.axis_plane_target(cam, uv, cur_world, R @ np.eye(3)["xyz".index(plane)])
                if world_t is None:
                    return {"pose": pose, "unreachable": True}
        elif target.get("type") == "world":
            world_t = np.array([float(v) for v in target["point"]])
        else:
            raise ApiError("target.type must be 'camera' or 'world'")
        lock = "xyz".index(target["plane"]) if target.get("type") == "camera" and target.get("plane") else None
        new_pose, residual = core.solve_drag(self.sk, pose, joint, offset, core.to_char_frame(officer, world_t), lock_axis=lock)
        ev = core.evaluate(self.sk, new_pose, officer, cam_params)
        ev["gesture_matches"] = core.match_gestures(self.sk, self.gestures.gestures, new_pose, cam_params["azimuth_deg"])
        return {"pose": new_pose, "residual_cm": residual, "eval": ev}

    def api_backplate(self, body) -> tuple[bytes, str]:
        bg = self.ws.get_background(str(body.get("bg_id")))
        if bg is None:
            raise ApiError("unknown background", 404)
        officer, cam_params = self._scene_args(body)
        cam = core.make_camera(cam_params, officer)
        if cam.lens is not None and cam.lens.active:         # the photo is distorted by the lens like the officer
            return _lens_jpeg(str(self.ws.lib_dir), bg["file"], bg.get("horizon_y_frac"), _cam_key(cam)), "image/jpeg"
        return _compose_jpeg(str(self.ws.lib_dir), bg["file"], bg.get("horizon_y_frac"), cam.width, cam.height,
                             round(float(core.camera_summary(cam)["horizon_v"]), 1)), "image/jpeg"

    def _mesh_renderer(self, want):
        """(render, render_layer) for the page's renderer choice: 'gpu' = the GPU renderer, anything else the CPU one."""
        if want == "gpu":
            try:
                return mesh_renderer.get("gpu")
            except (RuntimeError, ImportError) as e:
                raise ApiError(f"GPU renderer unavailable: {e}", 503) from None
        return mesh_renderer.get("cpu")

    def api_model_layer(self, body) -> tuple[bytes, str]:
        """The 3D uniform alone (RGBA PNG, transparent background) for the page to draw over its own background in the camera view."""
        officer, cam_params = self._scene_args(body)
        pose = self._pose(body)
        uni = self._uniform_for(body)
        mesh = self.mesh_for(uni)
        if mesh is None:
            raise ApiError(f"uniform '{uni['id']}' has no 3D mesh", 404)
        pose, bspec = self._gripped(pose, body)
        mesh = batonmod.mesh_with_officer_baton(mesh, bspec, bspec["mode"] if bspec else "across", prop=self.baton_prop())
        mesh = self._with_podium(mesh, body)
        brightness = self._brightness(body)
        colors = dict(uni["proxy_colors"])
        scale = min(1.0, MODEL_LAYER_MAX_W / cam_params["width"])
        small = {**cam_params, "width": max(16, round(cam_params["width"] * scale)), "height": max(16, round(cam_params["height"] * scale))}
        cam = core.make_camera(small, officer)           # same field of view, fewer pixels
        R, t = core.officer_transform(officer)
        fk = self.sk.fk(pose).transformed(R, t)
        render_layer = self._mesh_renderer(body.get("renderer"))[1]
        draw = lambda c: render_layer(mesh, fk, c, colors, PREVIEW_LIGHT, brightness)
        rgba = layer_with_lens(draw, cam) if cam.lens is not None and cam.lens.active else draw(cam)       # the lens distorts the layer like the photo
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=1)
        return buf.getvalue(), "image/png"

    def api_render(self, body) -> dict:
        """Render with the project's proxy backend over the composed backplate: the generator's own pipeline."""
        officer, cam_params = self._scene_args(body)
        pose = self._pose(body)
        cam = core.make_camera(cam_params, officer)
        bg = self.ws.get_background(str(body.get("bg_id")))
        lensed = cam.lens is not None and cam.lens.active
        if bg is not None and lensed:
            plate = Image.open(io.BytesIO(_pin_jpeg(str(self.ws.lib_dir), bg["file"], bg.get("horizon_y_frac"), _cam_key(cam)))).convert("RGB")   # pinhole canvas, distorted below
        elif bg is not None:
            jpeg = _compose_jpeg(str(self.ws.lib_dir), bg["file"], bg.get("horizon_y_frac"), cam.width, cam.height,
                                 round(float(core.camera_summary(cam)["horizon_v"]), 1))
            plate = Image.open(io.BytesIO(jpeg)).convert("RGB")
        else:
            plate = Image.new("RGB", (cam.width, cam.height), (150, 150, 150))
        uni = self._uniform_for(body)
        colors = dict(uni["proxy_colors"])
        brightness = self._brightness(body)
        want = body.get("renderer") or "mesh"
        if want not in ("mesh", "gpu", "proxy"):
            raise ApiError("renderer must be 'mesh', 'gpu' or 'proxy'")
        render_mesh = self._mesh_renderer(want)[0] if want != "proxy" else None
        mesh = self.mesh_for(uni) if want != "proxy" else None
        pose, bspec = self._gripped(pose, body)
        if mesh is not None:
            mesh = batonmod.mesh_with_officer_baton(mesh, bspec, bspec["mode"] if bspec else "across", prop=self.baton_prop())
            mesh = self._with_podium(mesh, body)
        R, t = core.officer_transform(officer)
        fk = self.sk.fk(pose).transformed(R, t)
        extra = [(*batonmod.segment(self.sk, fk, bspec, bspec["mode"]), bspec["radius_cm"], "leather")] if bspec else None

        def draw(c, bg):
            if mesh is not None:
                return render_mesh(mesh, fk, c, bg, colors, PREVIEW_LIGHT, float(brightness))
            return render(self.sk, fk, c, bg, colors, PREVIEW_LIGHT, extra_capsules=extra)   # uniforms without a mesh (or an explicit request): capsule proxy
        rgb, mask = render_with_lens(draw, cam, plate) if lensed else draw(cam, plate)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "PNG")
        return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                "bbox": bbox_from_mask(mask), "uniform": uni["id"], "renderer": want if mesh is not None else "proxy",
                "brightness": float(brightness) if mesh is not None else 1.0}

    def api_save_pose(self, body) -> dict:
        pid, kind = str(body.get("id", "")), body.get("kind")
        if not ID_RE.match(pid):
            raise ApiError("pose id: letters, digits, _ and - only (max 64)")
        pose = self._pose(body)
        errs = self.sk.pose_errors(pose)
        if errs:
            raise ApiError("pose violates joint limits: " + "; ".join(errs[:3]))
        is_finger = lambda j: any(f in j for f in FINGER_NAMES)  # noqa: E731
        if kind == "body":
            joints = {j: d for j, d in pose.items() if not is_finger(j)}
            folder = "body"
        elif kind == "hand":
            side = body.get("side")
            if side not in ("l", "r"):
                raise ApiError("side must be 'l' or 'r' for a hand pose")
            joints = {j[: -len(side) - 1]: d for j, d in pose.items() if is_finger(j) and j.endswith("_" + side)}
            folder = "hands"
        else:
            raise ApiError("kind must be 'body' or 'hand'")
        path = self.root / "poses" / folder / f"{pid}.json"
        if path.exists() and not body.get("overwrite"):
            raise ApiError(f"poses/{folder}/{pid}.json already exists (tick overwrite to replace it)", 409)
        try:
            save_pose(path, pid, joints, kind=kind, description=str(body.get("description", ""))[:500])
        except SchemaError as e:
            raise ApiError(str(e)) from None
        self._reload_poses()
        return {"saved": str(path.relative_to(self.root)), "poses": {"body": sorted(self.poses.body), "hands": sorted(self.poses.hands)}}

    def _reload_poses(self):
        """Pose files changed on disk: reload them AND the gestures that refer to them (the gesture library keeps the pose library it
        was built with, so a pose overwritten without this kept its old values in Studio's gestures until a restart)."""
        self.poses = PoseLibrary(self.root / "poses")
        pdir = self.profile_dir
        self.gestures = GestureLibrary(load_json(pdir / self.profile["files"]["gestures"], "gestures"), self.poses)

    def api_save_gesture_pose(self, body) -> dict:
        """Overwrite the body pose of a gesture variant with the current pose (Studio: edit a gesture, then save it over the original).
        The pose must respect the joint limits and the gesture's own constraints (body['force'] saves anyway). The old file is kept in
        poses/body/.backup/; the mirror variant (stop_bent_l <-> stop_bent_r) gets the mirrored pose unless body['mirror'] is false."""
        gid, vid = str(body.get("gesture", "")), str(body.get("variant", ""))
        g = self.gestures.gestures.get(gid)
        if g is None:
            raise ApiError(f"unknown gesture '{gid}'")
        var = next((v for v in g["pose_variants"] if v["id"] == vid), None)
        if var is None:
            raise ApiError(f"gesture {gid} has no variant '{vid}'")
        pose = self._pose(body)
        errs = self.sk.pose_errors(pose)
        if errs:
            raise ApiError("pose violates joint limits: " + "; ".join(errs[:3]))
        problems, _ = pose_validity(self.sk, g, var, pose)
        if problems and not body.get("force"):
            raise ApiError(f"the pose is not a valid {g['label']} / {vid}: " + "; ".join(problems[:4]) + " (save anyway: force)", 422)
        is_finger = lambda j: any(f in j for f in FINGER_NAMES)  # noqa: E731
        joints = {j: {d: v for d, v in dofs.items()} for j, dofs in pose.items() if not is_finger(j) and dofs}
        pid = var["pose"]
        writes = [(pid, joints)]
        twin = mirror_name(pid)
        if body.get("mirror", True) and twin and twin in self.poses.body:
            mirrored = mirror_pose(joints)
            a, b = self.sk.fk(joints), self.sk.fk(mirrored)                # check: the two sides really are mirror images
            for jn in ("hand_l", "hand_r", "lowerarm_l", "lowerarm_r", "head"):
                tw = jn[:-2] + ("_r" if jn.endswith("_l") else "_l") if jn[-2:] in ("_l", "_r") else jn
                pa, pb = a.positions[jn], b.positions[tw]
                if abs(pa[0] - pb[0]) > 0.5 or abs(pa[1] + pb[1]) > 0.5 or abs(pa[2] - pb[2]) > 0.5:
                    raise ApiError(f"mirroring {pid} -> {twin} does not give a mirror image at {jn} (save with mirror off)")
            writes.append((twin, mirrored))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        saved = []
        for name, js in writes:
            path = self.root / "poses" / "body" / f"{name}.json"
            old = self.poses.body.get(name, {})
            if path.exists():
                bdir = path.parent / ".backup"
                bdir.mkdir(exist_ok=True)
                (bdir / f"{name}.{stamp}.json").write_bytes(path.read_bytes())
            try:
                save_pose(path, name, js, kind="body", description=old.get("description", "") or f"edited in tosdg studio ({gid} / {vid})")
            except SchemaError as e:
                raise ApiError(str(e)) from None
            saved.append(str(path.relative_to(self.root)).replace("\\", "/"))
        self._reload_poses()
        return {"saved": saved, "warnings": problems, "poses": {"body": sorted(self.poses.body), "hands": sorted(self.poses.hands)}}

    def api_save_scene(self, body) -> dict:
        officer, cam_params = self._scene_args(body)
        pose = self._pose(body)
        cam = core.make_camera(cam_params, officer)
        scene = {
            "format": "tosdg.studio_scene/1",
            "note": "Authoring snapshot from `tosdg studio`; UE world (X fwd, Y right, Z up, cm).",
            "officer": officer, "camera_params": cam_params, "camera": cam.to_metadata(), "pose": pose,
            "render": {"renderer": body.get("renderer") or "mesh", "uniform_brightness": body.get("brightness", 1.0)},
            "background": self.ws.get_background(str(body.get("bg_id"))),
            "gesture_matches": [m for m in core.match_gestures(self.sk, self.gestures.gestures, pose,
                                                               cam_params["azimuth_deg"]) if m["ok"]],
        }
        name = str(body.get("name") or "scene")
        p = self.ws.save_scene(name, scene)
        return {"saved": str(p)}

    def api_state(self, body) -> dict:
        self.ws.save_state(body)
        return {"ok": True}

    def upload_model(self, name: str, data: bytes) -> dict:
        """Register an uploaded GLB (a person in uniform, without a baton of its own) as the profile's active model; the skeleton in the
        file is used when it has one."""
        import tempfile
        from ..models import register_model
        if data[:4] != b"glTF":
            raise ApiError("not a GLB file (binary glTF 2.0 expected)")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "upload.glb"
            tmp.write_bytes(data)
            try:
                r = register_model(self.root, self.profile["country"], tmp, name, profile_dir=self.profile_dir)
            except (ValueError, KeyError, OSError) as e:
                raise ApiError(f"cannot use this model: {e}") from None
        doc = load_json(self.profile_dir / self.uniforms_file, "uniforms")["uniforms"]
        self.uniforms = {u["id"]: u for u in doc}
        self._mesh_cache = {}
        from ..models import list_models
        return {"model": r, "models": list_models(self.profile_dir), "profile_uniforms": self.profile_uniform_list()}

    def profile_uniform_list(self) -> list:
        return [{"id": u["id"], "description": u["description"], "proxy_colors": u["proxy_colors"], "has_mesh": "mesh" in u}
                for u in self.uniforms.values()]

    def api_models(self, body) -> dict:
        from ..models import list_models
        return {"models": list_models(self.profile_dir)}

    def api_model_activate(self, body) -> dict:
        """Make a saved model (including the default one) the active model; re-imports it when it is not registered yet."""
        from ..models import activate_model, list_models
        try:
            r = activate_model(self.root, self.profile["country"], str(body.get("id", "")), profile_dir=self.profile_dir)
        except (ValueError, KeyError, OSError) as e:
            raise ApiError(f"cannot use this model: {e}") from None
        doc = load_json(self.profile_dir / self.uniforms_file, "uniforms")["uniforms"]
        self.uniforms = {u["id"]: u for u in doc}
        self._mesh_cache = {}
        return {"model": r, "models": list_models(self.profile_dir), "profile_uniforms": self.profile_uniform_list()}

    def api_background_delete(self, body) -> dict:
        self.ws.delete_background(str(body.get("id")))
        _pin_jpeg.cache_clear()
        _lens_jpeg.cache_clear()
        _compose_jpeg.cache_clear()  # a later upload may reuse the same file name; never serve the old composition
        return {"deleted": str(body.get("id")), "backgrounds": self.ws.list_backgrounds()}

    # ------------------------------------------------------------- random generate (a batch with some values locked)
    def random_config(self, body: dict) -> dict:
        """A generation config from the Studio's "Random generate" panel: what is locked comes from the Studio scene, the
        rest is drawn by the generator as usual (camera profile ranges, gestures, photos, ...)."""
        def pick(key, allowed):
            v = body.get(key)
            if v in (None, "", "random"):
                return None
            if v not in allowed:
                raise ApiError(f"{key}: unknown value {v!r}")
            return v
        n, seed = body.get("num_samples", 20), body.get("seed", 0)
        if not (isinstance(n, int) and 1 <= n <= RANDOM_MAX_SAMPLES) or not (isinstance(seed, int) and seed >= 0):
            raise ApiError(f"num_samples must be 1..{RANDOM_MAX_SAMPLES} and seed a non-negative integer")
        res = body.get("resolution", [1280, 720])
        if not (isinstance(res, list) and len(res) == 2 and all(isinstance(v, int) and 64 <= v <= 4096 for v in res)):
            raise ApiError("resolution must be [width, height], 64..4096 px")
        gestures = list(self.gestures.gestures)
        profiles = [c["id"] for c in self.camera_profiles]
        cfg = {"schema_version": "1.6", "name": "studio_random", "profile": self.profile_dir.name, "character": self.sk.id,
               "backend": "mesh", "master_seed": seed, "num_samples": n, "resolution": res, "image_format": "jpg",
               "background_library": str(self.library_dir.resolve()), "uniform_brightness_range": [0.9, 1.1],
               "max_attempts_per_sample": 60, "max_pose_tries": 200}
        g = pick("gesture", gestures)
        if g:
            cfg["gesture_weights"] = {k: (1.0 if k == g else 0.0) for k in gestures}
        c = pick("camera_type", profiles)
        if c:
            cfg["camera_profile_weights"] = {k: (1.0 if k == c else 0.0) for k in profiles}
        cam = body.get("camera_lock")
        if cam:                                         # the Studio view: distance, height, azimuth, focal length, ...
            ov = {}
            for k in CAMERA_OVERRIDE_KEYS:
                v = cam.get(k)
                if not isinstance(v, (int, float)):
                    raise ApiError(f"camera_lock.{k} must be a number")
                ov[k] = [float(v), float(v)]
            cfg["camera_overrides"] = ov
        b = pick("background", [x["id"] for x in self.ws.list_backgrounds()])
        if b:
            cfg["background_ids"] = [b]
        if body.get("baton"):
            cfg["baton"] = {"enabled": True, "hand": "r"}
        pod = body.get("podium_probability", 0.0)
        if not (isinstance(pod, (int, float)) and 0.0 <= pod <= 1.0):
            raise ApiError("podium_probability must be 0..1")
        if pod > 0:
            cfg["podium"] = {"probability": float(pod)}
        turn = body.get("officer_turn", "junction")
        if turn in ("road", "junction"):
            cfg["officer_yaw_mode"] = turn
        elif turn != "studio" or not cam:
            raise ApiError("officer_turn must be 'junction', 'road', or 'studio' together with a locked camera")
        if turn == "junction" and not body.get("size_from_people", True):
            raise ApiError("officer_turn 'junction' needs the size from the people in the photo")
        cfg["match_backplate_light"] = True                     # night / dusk photos: the officer as dark as the photo
        if body.get("size_from_people", True):
            cfg["size_reference"] = "auto"
            if body.get("min_officer_height_px") and not cam:    # a locked camera keeps its distance
                cfg["min_officer_height_px"] = int(body["min_officer_height_px"])
        if body.get("lens") is False:
            cfg["lens_distortion"] = False
        if body.get("renderer", "gpu") == "gpu" and self.gpu_info().get("available"):
            cfg["mesh_renderer"] = "gpu"
        return cfg

    def api_random_generate(self, body) -> dict:
        """Start a batch in the background (one at a time); poll /api/random_status."""
        from ..generate import run
        from ..scene import GenerationContext
        job = self.random_job
        if job and job.get("running"):
            raise ApiError("a random generation is already running", 409)
        cfg = self.random_config(body)
        name = time.strftime("%Y%m%d-%H%M%S")
        RANDOM_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = RANDOM_DIR / f"{name}.config.json"
        cfg_path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        try:
            ctx = GenerationContext(cfg_path)                # config errors are reported now, not in the thread
        except (ValueError, OSError, SchemaError) as e:
            cfg_path.unlink(missing_ok=True)
            raise ApiError(str(e)) from None
        job = self.random_job = {"running": True, "name": name, "total": cfg["num_samples"], "done": 0, "failed": 0,
                                  "log": [], "error": None, "samples": [], "config": cfg}

        def log(line: str):
            job["log"] = (job["log"] + [line])[-8:]
            if line.startswith("["):
                job["done"] += 1
                job["failed"] += "FAILED" in line

        def work():
            try:
                man = run(ctx, RANDOM_DIR / name, log=log)
                job["samples"] = [f"{name}/{json.loads((RANDOM_DIR / name / x['metadata']).read_text())['files']['rgb']}"
                                  for x in man["samples"]]
            except Exception as e:                    # reported to the page; the server keeps running
                job["error"] = str(e)
            finally:
                job["running"] = False
        threading.Thread(target=work, daemon=True).start()
        return self.api_random_status({})

    def api_random_status(self, _body) -> dict:
        job = self.random_job
        if not job:
            return {"running": False, "name": None}
        return {k: job[k] for k in ("running", "name", "total", "done", "failed", "log", "error", "samples")} | {
            "out_dir": str(RANDOM_DIR / job["name"])}

    def api_background_update(self, body) -> dict:
        fields = {k: body[k] for k in ("horizon_y_frac", "tags", "road_yaw_deg", "camera_height_cm") if k in body}
        return {"background": self.ws.update_background(str(body.get("id")), **fields)}

    POST_JSON = {
        "/api/eval": api_eval, "/api/drag": api_drag, "/api/render": api_render, "/api/pose_preset": api_pose_preset,
        "/api/save_pose": api_save_pose, "/api/save_gesture_pose": api_save_gesture_pose, "/api/save_scene": api_save_scene,
        "/api/state": api_state,
        "/api/background_update": api_background_update, "/api/background_delete": api_background_delete,
        "/api/random_generate": api_random_generate, "/api/random_status": api_random_status,
        "/api/models": api_models, "/api/model_activate": api_model_activate,
    }


@lru_cache(maxsize=6)
def _compose_jpeg(ws_root: str, rel: str, horizon_frac, width: int, height: int, horizon_v: float) -> bytes:
    entry = {"horizon_y_frac": horizon_frac}
    with Image.open(Path(ws_root) / rel) as im:
        plate, _ = compose_backplate(im, entry, horizon_v, width, height, np.random.default_rng(0), best_effort=True)
    buf = io.BytesIO()
    plate.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _cam_key(cam) -> tuple:
    """Hashable essentials of a lens camera (what the backplate composition depends on)."""
    return (cam.width, cam.height, cam.focal_length_mm, cam.sensor_width_mm, round(cam.pitch, 3), cam.lens.k1, cam.lens.k2, cam.lens.image_circle)


def _cam_from_key(key: tuple) -> Camera:
    w, h, f, sensor, pitch, k1, k2, circle = key
    return Camera("plate", (0.0, 0.0, 0.0), pitch, 0.0, 0.0, f, sensor, w, h, lens=Lens(k1, k2, circle))


@lru_cache(maxsize=16)
def _pin_jpeg(ws_root: str, rel: str, horizon_frac, key: tuple) -> bytes:
    """The background composed for the lens camera's pinhole canvas (larger than the frame), so a renderer can draw on it."""
    cam = _cam_from_key(key)
    Wr, Hr = canvas_size(cam)
    with Image.open(Path(ws_root) / rel) as im:
        plate, _ = compose_backplate(im, {"horizon_y_frac": horizon_frac}, pinhole_horizon_row(cam, Hr), Wr, Hr, np.random.default_rng(0), best_effort=True)
    buf = io.BytesIO()
    plate.save(buf, "JPEG", quality=92)
    return buf.getvalue()


@lru_cache(maxsize=16)
def _lens_jpeg(ws_root: str, rel: str, horizon_frac, key: tuple) -> bytes:
    """The same background as the lens camera sees it (distorted, black outside the image circle), at the frame size."""
    cam = _cam_from_key(key)
    plate = Image.open(io.BytesIO(_pin_jpeg(ws_root, rel, horizon_frac, key))).convert("RGB")
    buf = io.BytesIO()
    distort_plate(plate, cam).save(buf, "JPEG", quality=88)
    return buf.getvalue()


def make_handler(app: StudioApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tosdg-studio"

        def log_message(self, fmt, *args):  # quiet
            pass

        # -- plumbing
        def _host_ok(self) -> bool:
            host = self.headers.get("Host") or ""
            name = host[: host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]
            return name in ALLOWED_HOSTS

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _read_body(self, limit: int) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                raise ApiError("request too large", 413)
            return self.rfile.read(n)

        def _json_body(self) -> dict:
            try:
                body = json.loads(self._read_body(MAX_JSON_BYTES) or b"{}")
            except ValueError:
                raise ApiError("invalid JSON") from None
            if not isinstance(body, dict):
                raise ApiError("JSON body must be an object")
            return body

        # -- verbs
        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8",
                                      {"Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                                       "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"})
                if u.path == "/api/init":
                    with app.lock:
                        return self._json(200, app.api_init({}))
                if u.path.startswith("/bg/"):
                    p = app.ws.resolve_bg(u.path[len("/bg/"):].replace("%20", " "))
                    ctype = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}.get(p.suffix.lower()) if p else None
                    if ctype is None:
                        return self._json(404, {"error": "not found"})
                    return self._send(200, p.read_bytes(), ctype)
                if u.path.startswith("/gen/"):                     # images of a Studio random generation
                    p = (RANDOM_DIR / u.path[len("/gen/"):].replace("%20", " ")).resolve()
                    if not (p.is_file() and RANDOM_DIR.resolve() in p.parents and p.suffix.lower() in (".jpg", ".png")):
                        return self._json(404, {"error": "not found"})
                    return self._send(200, p.read_bytes(), "image/jpeg" if p.suffix.lower() == ".jpg" else "image/png")
                self._json(404, {"error": "not found"})
            except ApiError as e:
                self._json(e.status, {"error": str(e)})
            except OSError as e:
                self._json(404, {"error": f"not found: {e.strerror}"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            if self.headers.get("X-Studio") != "1":
                return self._json(403, {"error": "missing X-Studio header"})
            u = urlparse(self.path)
            try:
                if u.path in ("/api/ping", "/api/bye"):          # which pages are open (see Presence)
                    cid = str(self._json_body().get("id") or "")[:64]
                    if cid:
                        (app.presence.ping if u.path == "/api/ping" else app.presence.bye)(cid)
                    return self._json(200, {"ok": True})
                if u.path == "/api/upload_model":
                    q = parse_qs(u.query)
                    data = self._read_body(MAX_MODEL_UPLOAD_BYTES + 1024)
                    with app.lock:
                        return self._json(200, app.upload_model((q.get("name") or ["model.glb"])[0], data))
                if u.path == "/api/upload_background":
                    q = parse_qs(u.query)
                    data = self._read_body(MAX_UPLOAD_BYTES + 1024)
                    name = (q.get("name") or ["upload"])[0]
                    with app.lock:
                        return self._json(200, {"background": app.ws.add_background(name, data)})
                if u.path == "/api/backplate":
                    body = self._json_body()
                    with app.lock:
                        data, ctype = app.api_backplate(body)
                    return self._send(200, data, ctype)
                if u.path == "/api/model_layer":
                    data, ctype = app.api_model_layer(self._json_body())      # read-only, so it does not take the lock
                    return self._send(200, data, ctype)
                fn = app.POST_JSON.get(u.path)
                if fn is None:
                    return self._json(404, {"error": "not found"})
                body = self._json_body()
                with app.lock:
                    result = fn(app, body)
                self._json(200, result)
            except ApiError as e:
                self._json(e.status, {"error": str(e)})
            except WorkspaceError as e:
                self._json(400, {"error": str(e)})
            except (KeyError, TypeError, IndexError, ValueError) as e:
                self._json(400, {"error": f"malformed request: {e!r}"})

    return Handler


class Presence:
    """Which browser pages are open. Pages ping every few seconds and say bye when they are closed; a refresh says bye and pings again
    right away, so it only counts as "everyone left" when nobody is back within `grace` seconds. A page that went silent for `stale`
    seconds (crash, sleep) counts as gone. Used to stop the server when the last tab is closed (--exit-on-close)."""

    def __init__(self, grace: float = 4.0, stale: float = 90.0):
        self.grace, self.stale = grace, stale
        self.seen: dict[str, float] = {}
        self.ever = False
        self.empty_since: float | None = None
        self.lock = threading.Lock()

    def ping(self, cid: str) -> None:
        with self.lock:
            self.seen[cid] = time.monotonic()
            self.ever, self.empty_since = True, None

    def bye(self, cid: str) -> None:
        with self.lock:
            self.seen.pop(cid, None)

    def should_exit(self) -> bool:
        now = time.monotonic()
        with self.lock:
            self.seen = {k: t for k, t in self.seen.items() if now - t < self.stale}
            if not self.ever or self.seen:
                self.empty_since = None
                return False
            if self.empty_since is None:
                self.empty_since = now
            return now - self.empty_since >= self.grace


class StudioServer(ThreadingHTTPServer):
    """Owns its port alone. http.server sets SO_REUSEADDR, which on Windows lets a second server bind a port that is already
    listening: two Studios then answer at random and one of them can serve stale state. Windows gets SO_EXCLUSIVEADDRUSE instead,
    so a busy port is an error; elsewhere SO_REUSEADDR only skips TIME_WAIT and stays."""
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class WorkspaceInUse(RuntimeError):
    pass


def lock_workspace(root: Path):
    """Hold <workspace>/.studio.lock for the life of the process: two Studios on one workspace overwrite each other's state.json.
    The OS drops the lock when the process ends (also when it is killed), so a stale lock file never blocks a restart.
    Returns the open lock file (keep a reference to it)."""
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / ".studio.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(2**31 - 2)                             # lock a byte far past the text: readers (a second Studio, an editor) still see who holds it
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            f.seek(0)
            holder = f.read().strip() or "another process"
        except OSError:                                   # Windows: the holder's locked byte cannot be read
            holder = "another process"
        f.close()
        raise WorkspaceInUse(f"workspace {root} is in use by another Studio ({holder}); close it, "
                             "or start this one with --workspace <other dir>") from None
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}")
    f.flush()
    return f


def create_server(app: StudioApp, port: int = 8765) -> ThreadingHTTPServer:
    return StudioServer(("127.0.0.1", port), make_handler(app))


def serve(port: int = 8765, workspace=None, profile: str = "VN", library=None, open_browser: bool = True,
          exit_on_close: bool = False) -> int:
    app = StudioApp(workspace, profile, library=library)
    try:
        lock = lock_workspace(Path(app.ws.root))        # held until the process ends
    except WorkspaceInUse as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        srv = create_server(app, port)
    except OSError as e:
        print(f"error: port {port} is in use ({e.strerror or e}); close the program using it (start_studio.bat does), "
              "or pass --port <another port>", file=sys.stderr)
        return 1
    lock.truncate(0)
    lock.write(f"pid {os.getpid()}, port {srv.server_address[1]}")
    lock.flush()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"tosdg studio on {url}   workspace: {app.ws.root}\n" + ("Closes when the browser tab is closed. Ctrl+C also stops it" if exit_on_close else "Ctrl+C to stop"))
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    if exit_on_close:
        def watch():
            while not app.presence.should_exit():
                time.sleep(1.0)
            print("browser closed: stopping")
            srv.shutdown()
        threading.Thread(target=watch, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(serve())
