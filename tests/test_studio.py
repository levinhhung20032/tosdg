"""Studio: IK/drag, projection parity with the generator camera, workspace safety, HTTP API.

Contents:
  default_pose()
  image_bytes()
  CoreTests (13 tests)
  WorkspaceTests (8 tests)
  HttpTests (15 tests)
  PresenceTests (1 tests)
  ServerTests (3 tests)
  GestureEditTests (3 tests)
  PageScriptTests (1 tests)
"""
import time
import base64
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import ROOT, poses, skeleton, vn_gestures
from tosdg.backgrounds import BackgroundLibrary
from tosdg.poses import merge
from tosdg.studio import core
from tosdg.studio.server import StudioApp, WorkspaceInUse, create_server, lock_workspace
from tosdg.studio.workspace import Workspace, WorkspaceError


def default_pose():
    p = poses()
    return merge(p.body_pose("neutral_stand"), p.hand_pose("relaxed", "l"), p.hand_pose("relaxed", "r"))


def image_bytes(w=64, h=48, color=(10, 120, 200), fmt="PNG"):
    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, fmt)
    return b.getvalue()


class CoreTests(unittest.TestCase):
    def test_eval_matches_generator_projection(self):
        sk, pose = skeleton(), default_pose()
        officer = {"x": 30.0, "y": -20.0, "yaw_deg": 25.0}
        cam_params = {"distance_cm": 900, "azimuth_deg": 40, "height_cm": 150, "focal_length_mm": 28}
        out = core.evaluate(sk, pose, officer, cam_params)
        cam = core.make_camera(cam_params, officer)
        R, t = core.officer_transform(officer)
        _, pts = sk.keypoints(sk.fk(pose).transformed(R, t), "coco17")
        uv, _ = cam.project(pts)
        got = np.array([k["uv"] for k in out["keypoints"]["coco17"]])
        np.testing.assert_allclose(got, uv, atol=1e-9)
        self.assertAlmostEqual(out["camera"]["hfov_deg"], cam.hfov_deg)

    def test_view_with_the_officer_partly_outside_the_frame_is_accepted(self):
        sk, pose = skeleton(), default_pose()
        officer = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        cam_params = {"distance_cm": 500, "azimuth_deg": 0, "height_cm": 150, "focal_length_mm": 28, "sensor_width_mm": 36.0, "yaw_offset_deg": 30}
        out = core.evaluate(sk, pose, officer, cam_params)
        bb = out["bbox"]
        self.assertTrue(bb[0] < 0 or bb[2] > out["camera"]["width"])        # really partly outside
        self.assertTrue(out["checks"]["bbox_edge_ok"])
        strict = core.evaluate(sk, pose, officer, cam_params, {"max_bbox_edge_touch": False, "min_keypoints_in_frame_frac": 1.0})
        self.assertFalse(strict["generator_would_accept_view"])      # the strict v2 rules still reject it
        self.assertTrue(out["generator_would_accept_view"])

    def test_eval_with_a_baton_projects_it_and_closes_the_hand(self):
        sk, pose = skeleton(), default_pose()
        pose = {**pose, "upperarm_r": {"flexion": 90.0}}
        officer = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        cam_params = {"distance_cm": 500, "azimuth_deg": 90, "height_cm": 150, "focal_length_mm": 28}
        off = core.evaluate(sk, pose, officer, cam_params)
        self.assertIsNone(off["baton"])
        from tosdg import baton
        from tosdg.poses import PoseLibrary
        spec = baton.normalise({"enabled": True, "hand": "r"})
        gp, info = baton.apply_grip(sk, pose, spec, "across", PoseLibrary(ROOT / "poses").hand_pose("fist", "r"))
        on = core.evaluate(sk, gp, officer, cam_params, baton={**spec, "mode": "across", **info})
        (ax, ay), (bx, by) = on["baton"]["uv"]
        self.assertLess(by, ay)                                             # the tip is above the handle in the image: pointing up
        self.assertEqual(on["baton"]["collisions"], [])
        self.assertLessEqual(on["bbox"][1], off["bbox"][1])                 # the baton extends the bbox upwards

    def test_joint_angle_rules_limits_wrist_ellipse_and_collisions(self):
        sk, pose = skeleton(), default_pose()
        _, v, why = core.set_joint_angle(sk, pose, "calf_r", "flexion", 200.0)                 # knee: the joint limit
        self.assertEqual((v, why), (sk.joints["calf_r"].dof("flexion")["max"], "joint limit"))
        _, v, why = core.set_joint_angle(sk, pose, "calf_r", "flexion", 70.0)
        self.assertEqual((v, why), (70.0, None))
        _, v, why = core.set_joint_angle(sk, pose, "hand_r", "flexion", -500.0)                # wrist: limited, not beyond its own range
        self.assertEqual((v, why), (-90.0, "wrist range"))
        flexed = {**pose, "hand_r": {"flexion": 60.0}}                                         # the wrist moves in an ellipse: flexed -> less deviation
        _, free, _ = core.set_joint_angle(sk, pose, "hand_r", "deviation", 30.0)
        _, tight, why = core.set_joint_angle(sk, flexed, "hand_r", "deviation", 30.0)
        self.assertEqual(free, 30.0)
        self.assertLess(tight, 30.0)
        self.assertEqual(why, "wrist range")
        # self-collision: the left arm cannot be swung into the body; it stops at the last collision-free angle
        new, v, why = core.set_joint_angle(sk, pose, "upperarm_l", "abduction", -30.0)
        self.assertTrue(why and why.startswith("collision"), why)
        self.assertGreater(v, -30.0)
        self.assertEqual(sk.collisions(sk.fk(new)), sk.collisions(sk.fk(pose)))               # no new collision
        with self.assertRaises(ValueError):
            core.set_joint_angle(sk, pose, "calf_r", "nope", 1.0)

    def test_dof_axes_are_the_right_handed_world_axes_of_each_dof(self):
        sk, pose = skeleton(), {**default_pose(), "lowerarm_r": {"flexion": 40.0, "pronation": 30.0}, "hand_r": {"flexion": 20.0, "deviation": 10.0}}
        officer = {"x": 5.0, "y": -3.0, "yaw_deg": 25.0}
        R, t = core.officer_transform(officer)
        fk = sk.fk(pose).transformed(R, t)
        axes = core.dof_axes(sk, pose, fk)
        for jn, dn, child in (("lowerarm_r", "pronation", "middle_01_r"), ("hand_r", "flexion", "middle_01_r"),
                              ("hand_r", "deviation", "middle_01_r"), ("calf_r", "flexion", "foot_r")):
            d = sk.joints[jn].dof(dn)
            p2 = {k: dict(v) for k, v in pose.items()}
            p2.setdefault(jn, {})[dn] = p2.get(jn, {}).get(dn, 0.0) + 1.0
            moved = sk.fk(p2).transformed(R, t).positions[child] - fk.positions[child]
            pred = np.cross(np.array(axes[jn][dn]) * np.radians(d["sign"]), fk.positions[child] - fk.positions[jn])
            self.assertAlmostEqual(float(moved @ pred / np.linalg.norm(moved) / np.linalg.norm(pred)), 1.0, places=3, msg=f"{jn}.{dn}")
        self.assertIn("dof_axes", core.evaluate(sk, pose, officer, {"distance_cm": 500, "azimuth_deg": 40, "height_cm": 150, "focal_length_mm": 28}))

    def test_camera_looks_at_aim_point(self):
        officer = {"x": 0, "y": 0, "yaw_deg": 0}
        cam = core.make_camera({"distance_cm": 1000, "azimuth_deg": 90, "height_cm": 200, "aim_height_cm": 100}, officer)
        uv, depth = cam.project(np.array([[0.0, 0.0, 100.0]]))
        np.testing.assert_allclose(uv[0], [cam.cx, cam.cy], atol=1e-6)
        self.assertGreater(depth[0], 0)

    def test_drag_reaches_reachable_target_within_limits(self):
        sk, pose = skeleton(), default_pose()
        start = sk.fk(pose).point("hand_r", [0, 0, 0])
        new, resid = core.solve_drag(sk, pose, "hand_r", [0, 0, 0], start + np.array([10.0, 25.0, 40.0]))
        self.assertLess(resid, 0.5)
        self.assertEqual(sk.pose_errors(new), [])

    def test_drag_unreachable_target_stays_within_limits(self):
        sk, pose = skeleton(), default_pose()
        new, resid = core.solve_drag(sk, pose, "hand_r", [0, 0, 0], np.array([0.0, 400.0, 100.0]))
        self.assertEqual(sk.pose_errors(new), [])
        self.assertGreater(resid, 100)

    def test_drag_leaves_other_limb_alone(self):
        sk, pose = skeleton(), default_pose()
        tgt = sk.fk(pose).point("hand_r", [0, 0, 0]) + np.array([0, 20.0, 50.0])
        new, _ = core.solve_drag(sk, pose, "hand_r", [0, 0, 0], tgt)
        for j, dofs in new.items():
            if j.endswith("_l"):
                self.assertEqual(dofs, pose.get(j, {}), j)

    def test_fingertip_drag_moves_finger_not_arm(self):
        sk, pose = skeleton(), default_pose()
        kp = sk.keypoint_sets["hand21_r"][8]
        p0 = sk.fk(pose).point(kp["joint"], kp["offset"])
        new, _ = core.solve_drag(sk, pose, kp["joint"], kp["offset"], p0 + np.array([0.0, 0.0, 3.0]))
        for dn, v in new.get("upperarm_r", {}).items():
            self.assertAlmostEqual(v, pose.get("upperarm_r", {}).get(dn, 0.0), delta=2.0)

    def test_camera_plane_target_roundtrip(self):
        officer = {"x": 0, "y": 0, "yaw_deg": 0}
        cam = core.make_camera({"distance_cm": 800, "azimuth_deg": 30}, officer)
        p = np.array([5.0, -10.0, 120.0])
        uv, _ = cam.project(p[None])
        # a point at the same camera depth as p, offset sideways in the screen plane
        right = cam.R[:, 1]
        back = core.camera_plane_target(cam, p + right * 30.0, uv[0])
        np.testing.assert_allclose(back, p, atol=1e-6)

    def test_ground_target_hits_ground(self):
        officer = {"x": 0, "y": 0, "yaw_deg": 0}
        cam = core.make_camera({"distance_cm": 800, "azimuth_deg": 0}, officer)
        uv, _ = cam.project(np.array([[10.0, 5.0, 0.0]]))
        np.testing.assert_allclose(core.ground_target(cam, uv[0]), [10.0, 5.0, 0.0], atol=1e-6)

    def test_gesture_matching_follows_pose(self):
        arm_up = merge(poses().body_pose("stop_bent_r"), poses().hand_pose("open_flat", "r"),
                       poses().hand_pose("relaxed", "l"))
        m = core.match_gestures(skeleton(), vn_gestures().gestures, arm_up, 90.0)      # seen from the officer's right
        self.assertTrue(m[0]["ok"])
        self.assertEqual(m[0]["gesture"], "VN_STOP_RIGHT")
        self.assertEqual(m[0]["instruction_for_side"], "STOP")


class WorkspaceTests(unittest.TestCase):
    """Background photos live in the generator's library (one folder for the Studio and `generate`)."""
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.ws = Workspace(Path(self._td.name) / "ws", Path(self._td.name) / "lib")

    def tearDown(self):
        self._td.cleanup()

    def add(self, name, data=None):
        return self.ws.add_background(name, data or image_bytes(), detect_ground=False)

    def test_background_upload_and_horizon(self):
        e = self.add("../../evil name.exe")
        self.assertTrue(e["file"].startswith("images/") and e["file"].endswith(".png"))
        self.assertEqual(e["tags"], ["day", "ground_level"])
        self.ws.update_background(e["id"], horizon_y_frac=0.4)
        lib = BackgroundLibrary(self.ws.lib_dir)                  # what `generate` reads
        self.assertEqual(lib.by_id[e["id"]]["horizon_y_frac"], 0.4)
        self.assertEqual(lib.integrity_errors(), [])
        with self.assertRaises(WorkspaceError):
            self.ws.update_background(e["id"], horizon_y_frac=1.5)

    def test_tags_are_saved_in_the_library(self):
        e = self.add("x")
        self.ws.update_background(e["id"], tags=["elevated", "Night!"])
        b = BackgroundLibrary(self.ws.lib_dir).by_id[e["id"]]
        self.assertEqual(b["tags"], ["Night", "elevated"])
        self.assertNotIn("scene", b)

    def test_delete_background_removes_file_and_entry(self):
        keep = self.add("keep")
        gone = self.add("gone", image_bytes(color=(200, 10, 10)))
        path = self.ws.lib_dir / gone["file"]
        self.assertTrue(path.exists())
        self.ws.delete_background(gone["id"])
        self.assertFalse(path.exists())
        self.assertEqual([b["id"] for b in BackgroundLibrary(self.ws.lib_dir).index["backgrounds"]], [keep["id"]])
        self.assertTrue((self.ws.lib_dir / keep["file"]).exists())
        with self.assertRaises(WorkspaceError):
            self.ws.delete_background(gone["id"])           # already gone
        with self.assertRaises(WorkspaceError):
            self.ws.delete_background("../../etc/passwd")   # not an id in the list

    def test_delete_never_leaves_the_library(self):
        e = self.add("evil")
        victim = Path(self._td.name) / "tosdg_victim.txt"
        victim.write_text("do not delete")
        idx_path = self.ws.lib_dir / "index.json"
        idx = json.loads(idx_path.read_text())
        idx["backgrounds"][0]["file"] = "../tosdg_victim.txt"   # a tampered index entry
        idx_path.write_text(json.dumps(idx))
        self.ws.delete_background(e["id"])
        self.assertTrue(victim.exists())

    def test_rejects_non_images_and_unsupported_formats(self):
        with self.assertRaises(WorkspaceError):
            self.add("x", b"not an image")
        with self.assertRaises(WorkspaceError):
            self.add("x", image_bytes(fmt="PPM"))
        e = self.add("x", image_bytes(fmt="BMP"))      # BMP / WebP / GIF / TIFF are accepted and stored as JPG
        self.assertTrue(e["file"].endswith(".jpg"))

    def test_an_old_workspace_folder_is_moved_into_the_library(self):
        td = Path(self._td.name)
        lib = BackgroundLibrary(td / "lib2")
        (td / "shared.png").write_bytes(image_bytes(color=(1, 2, 3)))
        lib.add(td / "shared.png", ["ground_level"], bg_id="shared")
        lib.save()
        old = td / "ws2" / "backgrounds"
        old.mkdir(parents=True)
        (old / "shared.png").write_bytes(image_bytes(color=(1, 2, 4)))      # the same photo re-encoded (same id and size): not copied again
        (old / "other.png").write_bytes(image_bytes(80, 60))                 # same id as a library photo, other size: a new id
        (old / "own.png").write_bytes(image_bytes(color=(200, 0, 0)))
        (old / "index.json").write_text(json.dumps({"backgrounds": [
            {"id": "shared", "file": "backgrounds/shared.png", "width": 64, "height": 48, "tags": ["day"]},
            {"id": "shared", "file": "backgrounds/other.png", "width": 80, "height": 60, "tags": ["day"]},
            {"id": "own", "file": "backgrounds/own.png", "width": 64, "height": 48, "horizon_y_frac": 0.3, "tags": ["day", "ground_level"]}]}))
        ws = Workspace(td / "ws2", td / "lib2")
        self.assertEqual(ws.migrated, ["shared_2", "own"])
        self.assertFalse(old.exists())
        lib = BackgroundLibrary(td / "lib2")
        self.assertEqual(sorted(lib.by_id), ["own", "shared", "shared_2"])
        self.assertEqual(lib.by_id["own"]["horizon_y_frac"], 0.3)
        self.assertEqual(lib.integrity_errors(), [])

    def test_resolve_blocks_traversal(self):
        e = self.add("ok")
        self.assertIsNotNone(self.ws.resolve_bg(e["file"]))
        self.assertIsNone(self.ws.resolve_bg("../../../etc/passwd"))
        self.assertIsNone(self.ws.resolve_bg("images/../../x"))


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.test_pose_file = ROOT / "poses" / "body" / "_studio_test_pose.json"
        cls.app = StudioApp(Path(cls._td.name) / "ws", library=Path(cls._td.name) / "lib")
        cls.srv = create_server(cls.app, 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls._td.cleanup()
        cls.test_pose_file.unlink(missing_ok=True)

    def post(self, path, body=None, raw=None, headers=None):
        h = {"X-Studio": "1", "Content-Type": "application/json", **(headers or {})}
        data = raw if raw is not None else json.dumps(body or {}).encode()
        req = urllib.request.Request(self.base + path, data=data, headers=h, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def test_index_and_init(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"tosdg studio", r.read().lower())
        with urllib.request.urlopen(self.base + "/api/init") as r:
            d = json.loads(r.read())
        self.assertIn("coco17", d["skeleton"]["keypoint_sets"])
        self.assertIn("neutral_stand", d["poses"]["body"])
        self.assertEqual(d["boot"], self.app.boot)   # pages of an older run compare it and close (start_studio.bat restart)
        self.assertIsInstance(d["boot"], int)

    def test_post_requires_header_and_host(self):
        req = urllib.request.Request(self.base + "/api/eval", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 403)
        st, _, _ = self.post("/api/eval", {"pose": {}}, headers={"Host": "evil.example.com"})
        self.assertEqual(st, 403)

    def test_eval_drag_flow(self):
        pose = self.app.default_pose()
        st, body, _ = self.post("/api/eval", {"pose": pose})
        self.assertEqual(st, 200)
        wrist = next(k for k in json.loads(body)["keypoints"]["coco17"] if k["name"] == "right_wrist")
        uv = [wrist["uv"][0], wrist["uv"][1] - 60]
        st, body, _ = self.post("/api/drag", {"pose": pose, "mode": "joint", "joint": wrist["joint"],
                                              "offset": wrist["offset"], "target": {"type": "camera", "uv": uv}})
        self.assertEqual(st, 200)
        st, body, _ = self.post("/api/eval", {"pose": json.loads(body)["pose"]})
        w2 = next(k for k in json.loads(body)["keypoints"]["coco17"] if k["name"] == "right_wrist")
        self.assertLess(w2["uv"][1], wrist["uv"][1] - 20)  # the wrist moved up on screen

    def test_officer_drag_moves_on_ground(self):
        pose = self.app.default_pose()
        st, body, _ = self.post("/api/drag", {"pose": pose, "mode": "officer",
                                              "target": {"type": "world", "point": [120.0, -40.0, 0.0]}})
        self.assertEqual(st, 200)
        o = json.loads(body)["officer"]
        self.assertEqual((o["x"], o["y"]), (120.0, -40.0))

    def test_validation_errors_are_400_not_500(self):
        self.assertEqual(self.post("/api/eval", {"pose": {"nope": {"x": 1}}})[0], 400)
        self.assertEqual(self.post("/api/eval", {"pose": {}, "camera": {"focal_length_mm": 0}})[0], 400)
        self.assertEqual(self.post("/api/eval", raw=b"{bad")[0], 400)
        self.assertEqual(self.post("/api/drag", {"pose": {}, "mode": "joint", "joint": "x", "target": {}})[0], 400)

    def test_background_flow_and_render(self):
        st, body, _ = self.post("/api/upload_background?name=street.png", raw=image_bytes(320, 240),
                                headers={"Content-Type": "image/png"})
        self.assertEqual(st, 200)
        bg = json.loads(body)["background"]
        self.assertEqual(self.post("/api/background_update", {"id": bg["id"], "horizon_y_frac": 0.45})[0], 200)
        st, _, h = self.post("/api/backplate", {"bg_id": bg["id"], "camera": {"width": 320, "height": 180}})
        self.assertEqual((st, h["Content-Type"]), (200, "image/jpeg"))
        with urllib.request.urlopen(self.base + "/bg/" + bg["file"]) as r:
            self.assertEqual(r.status, 200)
        st, body, _ = self.post("/api/render", {"pose": self.app.default_pose(), "bg_id": bg["id"],
                                                "camera": {"width": 320, "height": 180, "distance_cm": 600}})
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(body)["image"].startswith("data:image/png;base64,"))
        self.assertEqual(self.post("/api/background_update", {"id": bg["id"], "tags": ["ground_level", "day"]})[0], 200)
        lib = BackgroundLibrary(self.app.library_dir)          # the uploaded photo IS in the generator's library
        self.assertEqual(lib.integrity_errors(), [])
        self.assertEqual(lib.by_id[bg["id"]]["horizon_y_frac"], 0.45)
        self.assertEqual(lib.by_id[bg["id"]]["tags"], ["day", "ground_level"])

    def test_delete_background_and_reupload_same_name(self):
        def upload(color):
            st, body, _ = self.post("/api/upload_background?name=samename.png", raw=image_bytes(320, 240, color),
                                    headers={"Content-Type": "image/png"})
            self.assertEqual(st, 200)
            return json.loads(body)["background"]
        req = {"camera": {"width": 160, "height": 90}}
        red = upload((220, 20, 20))
        _, jpg_red, _ = self.post("/api/backplate", {"bg_id": red["id"], **req})
        st, body, _ = self.post("/api/background_delete", {"id": red["id"]})
        self.assertEqual(st, 200)
        self.assertNotIn(red["id"], [b["id"] for b in json.loads(body)["backgrounds"]])
        self.assertNotIn(red["id"], BackgroundLibrary(self.app.library_dir).by_id)     # `generate` no longer sees it
        self.assertEqual(self.post("/api/backplate", {"bg_id": red["id"], **req})[0], 404)
        self.assertEqual(self.post("/api/background_delete", {"id": red["id"]})[0], 400)
        blue = upload((20, 20, 220))                       # same file name, new content
        self.assertEqual(blue["id"], red["id"])
        _, jpg_blue, _ = self.post("/api/backplate", {"bg_id": blue["id"], **req})
        px_red = np.asarray(Image.open(io.BytesIO(jpg_red)).convert("RGB")).mean(axis=(0, 1))
        px_blue = np.asarray(Image.open(io.BytesIO(jpg_blue)).convert("RGB")).mean(axis=(0, 1))
        self.assertGreater(px_red[0], 150)                  # first upload was red...
        self.assertGreater(px_blue[2], 150)                 # ...the re-upload must NOT come back as the cached red image

    def test_ctrl_drag_keeps_the_joint_on_the_plane_across_the_chosen_axis(self):
        """Ctrl+drag with an axis aligned: the target is where the pointer ray meets the plane through the joint across that axis of the
        officer, so the joint keeps that coordinate (a camera looking at an angle no longer pushes it towards / away from itself)."""
        pose = self.app.default_pose()
        cam = {"width": 640, "height": 360, "distance_cm": 300, "azimuth_deg": 35, "height_cm": 190, "aim_height_cm": 120}
        officer = {"x": 0, "y": 0, "yaw_deg": 20, "z": 0}
        ev = json.loads(self.post("/api/eval", {"pose": pose, "camera": cam, "officer": officer})[1])
        u, v = ev["joints"]["hand_r"]["uv"]
        for axis in ("x", "y", "z"):
            body = {"pose": pose, "camera": cam, "officer": officer, "mode": "joint", "joint": "hand_r", "offset": [0, 0, 0],
                    "target": {"type": "camera", "uv": [u + 25, v - 30], "plane": axis}}
            st, raw, _ = self.post("/api/drag", body)
            self.assertEqual(st, 200)
            r = json.loads(raw)
            R, t = core.officer_transform(officer)
            before = R.T @ (np.array(ev["joints"]["hand_r"]["w"]) - t)
            after = R.T @ (np.array(r["eval"]["joints"]["hand_r"]["w"]) - t)
            i = "xyz".index(axis)
            self.assertLess(abs(after[i] - before[i]), 1.0 + r["residual_cm"], axis)          # that coordinate stays
            self.assertGreater(float(np.linalg.norm(after - before)), 3.0, axis)             # and the hand did move
        bad = {**body, "target": {**body["target"], "plane": "w"}}
        self.assertEqual(self.post("/api/drag", bad)[0], 400)

    def test_a_wrist_turned_in_studio_stays_turned_while_holding_the_baton(self):
        """Alt+click on the wrist turns its flexion / deviation; with the baton in that hand the grip used to straighten the wrist again,
        so the 3D render did not move while the capsules did. Studio keeps the wrist (the generator still straightens it)."""
        def png(d):
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(d["image"].split(",", 1)[1]))).convert("RGB")).astype(float)
        pose = self.app.default_pose()
        bent = {**pose, "hand_r": {**pose.get("hand_r", {}), "flexion": 50.0}}
        base = {"camera": {"width": 320, "height": 180, "distance_cm": 300}, "baton": {"enabled": True, "hand": "r", "mode": "across"}}
        a = json.loads(self.post("/api/render", {**base, "pose": pose})[1])
        b = json.loads(self.post("/api/render", {**base, "pose": bent})[1])
        self.assertGreater(np.abs(png(a) - png(b)).max(), 40)
        ev = json.loads(self.post("/api/drag", {**base, "pose": pose, "mode": "angle", "joint": "hand_r", "dof": "flexion", "value": 40})[1])
        self.assertAlmostEqual(ev["pose"]["hand_r"]["flexion"], 40.0, places=3)
        self.assertIsNotNone(ev["eval"].get("baton"))                      # the drag's evaluation draws the baton too

    def test_render_preview_uses_the_3d_uniform_and_the_brightness(self):
        def png(d):
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(d["image"].split(",", 1)[1]))).convert("RGB")).astype(float)
        base = {"pose": self.app.default_pose(), "camera": {"width": 320, "height": 180, "distance_cm": 380}}
        st, body, _ = self.post("/api/render", base)
        mesh = json.loads(body)
        self.assertEqual((st, mesh["renderer"], mesh["uniform"], mesh["brightness"]), (200, "mesh", "beta", 1.0))
        dark = json.loads(self.post("/api/render", {**base, "brightness": 0.6})[1])
        self.assertEqual(dark["brightness"], 0.6)
        a, b = png(mesh), png(dark)
        x0, y0, x1, y1 = mesh["bbox"]["xyxy"]
        self.assertLess(b[y0:y1, x0:x1].mean(), a[y0:y1, x0:x1].mean() * 0.95)          # darker cloth, same silhouette
        self.assertEqual(dark["bbox"]["xyxy"], mesh["bbox"]["xyxy"])
        proxy = json.loads(self.post("/api/render", {**base, "renderer": "proxy"})[1])
        self.assertEqual(proxy["renderer"], "proxy")
        self.assertGreater(np.abs(png(proxy) - a).mean(), 0.1)
        self.assertEqual(self.post("/api/render", {**base, "brightness": 9})[0], 400)
        self.assertEqual(self.post("/api/render", {**base, "brightness": "bright"})[0], 400)
        self.assertEqual(self.post("/api/render", {**base, "renderer": "unreal"})[0], 400)
        with urllib.request.urlopen(self.base + "/api/init") as r:
            uniforms = {u["id"]: u for u in json.loads(r.read())["profile_uniforms"]}
        self.assertEqual(list(uniforms)[0], "beta")                       # the active model comes first
        self.assertTrue(uniforms["beta"]["has_mesh"])

    def test_render_with_or_without_a_baton_uses_the_same_model(self):
        base = {"pose": self.app.default_pose(), "camera": {"width": 320, "height": 180, "distance_cm": 380}}
        plain = json.loads(self.post("/api/render", base)[1])
        held = json.loads(self.post("/api/render", {**base, "baton": {"enabled": True, "hand": "r"}})[1])
        self.assertEqual(plain["uniform"], "beta")
        self.assertEqual(held["uniform"], plain["uniform"])        # no baton-free variants: the baton is a prop, in the hand or on the belt
        self.assertTrue(plain["renderer"] == held["renderer"] == "mesh")
        self.assertEqual(self.post("/api/render", {**base, "baton": {"enabled": True, "hand": "x"}})[0], 400)

    def test_drag_mode_angle_turns_one_joint_within_the_physics_rules(self):
        base = {"pose": self.app.default_pose(), "camera": {"width": 320, "height": 180, "distance_cm": 380}}
        st, body, _ = self.post("/api/drag", {**base, "mode": "angle", "joint": "calf_r", "dof": "flexion", "value": 60})
        d = json.loads(body)
        self.assertEqual((st, d["pose"]["calf_r"]["flexion"], d["angle"]["limited_by"]), (200, 60.0, None))
        self.assertIn("dof_axes", d["eval"])
        d = json.loads(self.post("/api/drag", {**base, "mode": "angle", "joint": "calf_r", "dof": "flexion", "value": 999})[1])
        self.assertEqual(d["angle"]["limited_by"], "joint limit")
        for bad in ({"joint": "nope", "dof": "flexion", "value": 1}, {"joint": "calf_r", "dof": "x", "value": 1}, {"joint": "calf_r", "dof": "flexion", "value": "a"}):
            self.assertEqual(self.post("/api/drag", {**base, "mode": "angle", **bad})[0], 400)
        init = json.loads(urllib.request.urlopen(self.base + "/api/init").read())
        self.assertTrue(all("axis" in dd and "sign" in dd for j in init["skeleton"]["joints"] for dd in j["dofs"]))

    def test_model_layer_is_a_transparent_png_of_the_3d_uniform(self):
        base = {"pose": self.app.default_pose(), "camera": {"width": 1920, "height": 1080, "distance_cm": 380}}
        st, data, headers = self.post("/api/model_layer", base)
        self.assertEqual((st, headers["Content-Type"]), (200, "image/png"))
        im = Image.open(io.BytesIO(data))
        self.assertEqual((im.mode, im.size), ("RGBA", (960, 540)))                       # same aspect ratio, capped width
        a = np.asarray(im)[..., 3]
        self.assertTrue(0 < (a > 0).mean() < 0.3 and a.max() == 255)                      # only the officer is opaque
        small = Image.open(io.BytesIO(self.post("/api/model_layer", {**base, "camera": {"width": 400, "height": 300, "distance_cm": 380}})[1]))
        self.assertEqual(small.size, (400, 300))                                          # never scaled up
        dark = np.asarray(Image.open(io.BytesIO(self.post("/api/model_layer", {**base, "brightness": 0.6})[1])))
        lit = np.asarray(im)
        self.assertLess(dark[..., :3][a > 250].mean(), lit[..., :3][a > 250].mean())     # brightness reaches the layer
        np.testing.assert_array_equal(dark[..., 3], lit[..., 3])
        self.assertEqual(self.post("/api/model_layer", {**base, "brightness": 5})[0], 400)
        self.assertEqual(self.post("/api/model_layer", {"pose": {"nope": {"x": 1}}})[0], 400)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(urllib.request.Request(self.base + "/api/model_layer", data=b"{}", method="POST"))        # needs the X-Studio header
        self.assertEqual(cm.exception.code, 403)

    def test_workspace_files_are_not_served(self):
        for path in ("/files/../state.json", "/files/%2e%2e/%2e%2e/pyproject.toml", "/files/state.json", "/bg/../state.json"):
            try:
                urllib.request.urlopen(self.base + path)
                self.fail(f"{path} should not be served")
            except urllib.error.HTTPError as e:
                self.assertIn(e.code, (403, 404))

    def test_save_pose_refuses_overwrite_and_bad_ids(self):
        pose = self.app.default_pose()
        self.assertEqual(self.post("/api/save_pose", {"id": "../x", "kind": "body", "pose": pose})[0], 400)
        self.assertEqual(self.post("/api/save_pose", {"id": "_studio_test_pose", "kind": "body", "pose": pose})[0], 200)
        self.assertEqual(self.post("/api/save_pose", {"id": "_studio_test_pose", "kind": "body", "pose": pose})[0], 409)
        self.assertEqual(self.post("/api/save_pose", {"id": "_studio_test_pose", "kind": "body", "pose": pose,
                                                      "overwrite": True})[0], 200)
        bad = {**pose, "lowerarm_r": {"flexion": 999}}
        self.assertEqual(self.post("/api/save_pose", {"id": "_studio_x", "kind": "body", "pose": bad})[0], 400)


if __name__ == "__main__":
    unittest.main()


class PresenceTests(unittest.TestCase):
    def test_server_stops_only_after_the_last_page_has_been_gone_for_the_grace_period(self):
        from tosdg.studio.server import Presence
        p = Presence(grace=0.05, stale=60)
        self.assertFalse(p.should_exit())                 # nobody has connected yet: keep waiting
        p.ping("a"); p.ping("b")
        p.bye("a")
        self.assertFalse(p.should_exit())                 # another tab is still open
        p.bye("b")
        self.assertFalse(p.should_exit())                 # just left: a refresh may be about to come back
        p.ping("b2")
        time.sleep(0.1)
        self.assertFalse(p.should_exit())                 # the refreshed page is back
        p.bye("b2")
        p.should_exit()
        time.sleep(0.1)
        self.assertTrue(p.should_exit())                  # nobody came back


class ServerTests(unittest.TestCase):
    def test_a_second_server_cannot_take_a_port_in_use(self):
        """On Windows, http.server's SO_REUSEADDR let a second Studio bind a listening port and answer with stale state."""
        with tempfile.TemporaryDirectory() as td:
            app = StudioApp(Path(td) / "ws", library=Path(td) / "lib")
            first = create_server(app, 0)
            try:
                with self.assertRaises(OSError):
                    create_server(app, first.server_address[1])
            finally:
                first.server_close()

    def test_one_studio_per_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ws"
            held = lock_workspace(ws)
            try:
                with self.assertRaisesRegex(WorkspaceInUse, "in use by another Studio"):
                    lock_workspace(ws)
            finally:
                held.close()
            lock_workspace(ws).close()                  # released with its holder: a restart is never blocked by a stale file

    def test_render_refuses_angles_past_a_full_turn(self):
        with tempfile.TemporaryDirectory() as td:
            app = StudioApp(Path(td) / "ws", library=Path(td) / "lib")
            with self.assertRaisesRegex(Exception, r"not an angle in \[-360, 360\]"):
                app._pose({"pose": {"upperarm_r": {"flexion": 1e9}}})
            self.assertEqual(app._pose({"pose": {"upperarm_r": {"flexion": 200.0}}}), {"upperarm_r": {"flexion": 200.0}})   # past the limit: a warning, not an error


class GestureEditTests(unittest.TestCase):
    """Studio: load a gesture variant, change the pose, save it over the gesture's own pose file (with its mirror twin)."""
    @classmethod
    def setUpClass(cls):
        import shutil
        cls._td = tempfile.TemporaryDirectory()
        td = Path(cls._td.name)
        shutil.copytree(ROOT / "poses", td / "poses")                     # never touch the real pose files
        cls.app = StudioApp(td / "ws", library=td / "lib")
        cls.app.root = td
        cls.app._reload_poses()

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def call(self, body):
        from tosdg.studio.server import ApiError
        try:
            return 200, self.app.api_save_gesture_pose(body)
        except ApiError as e:
            return e.status, str(e)

    def pose_of(self, name):
        p = poses()
        return merge(self.app.poses.body_pose(name), p.hand_pose("open_flat", "r"), p.hand_pose("relaxed", "l"))

    def test_a_valid_edit_overwrites_the_pose_and_its_mirror_and_keeps_a_backup(self):
        from tosdg.poses import PoseLibrary
        pose = self.pose_of("stop_bent_r")
        pose["upperarm_r"] = {**pose["upperarm_r"], "abduction": pose["upperarm_r"].get("abduction", 0) + 6.0}
        st, r = self.call({"gesture": "VN_STOP_RIGHT", "variant": "bent", "pose": pose})
        self.assertEqual(st, 200, r)
        self.assertEqual(sorted(r["saved"]), ["poses/body/stop_bent_l.json", "poses/body/stop_bent_r.json"])
        lib = PoseLibrary(self.app.root / "poses")
        self.assertAlmostEqual(lib.body_pose("stop_bent_r")["upperarm_r"]["abduction"], pose["upperarm_r"]["abduction"], places=3)
        self.assertAlmostEqual(lib.body_pose("stop_bent_l")["upperarm_l"]["abduction"], pose["upperarm_r"]["abduction"], places=3)
        sk = skeleton("tripo_v1")
        fl, fr = sk.fk(lib.body_pose("stop_bent_l")), sk.fk(lib.body_pose("stop_bent_r"))
        np.testing.assert_allclose(fl.positions["hand_l"] * [1, -1, 1], fr.positions["hand_r"], atol=0.5)      # still mirror images
        self.assertEqual(len(list((self.app.root / "poses" / "body" / ".backup").glob("stop_bent_*.json"))), 2)
        # the gesture library follows the new file at once (it used to keep the old pose until a restart)
        self.assertAlmostEqual(self.app.gestures.poses.body_pose("stop_bent_r")["upperarm_r"]["abduction"], pose["upperarm_r"]["abduction"], places=3)

    def test_an_edit_that_is_no_longer_the_gesture_is_refused_unless_forced(self):
        pose = self.pose_of("neutral_stand")                               # arms down: not a STOP with the right arm
        st, msg = self.call({"gesture": "VN_STOP_RIGHT", "variant": "hand_up", "pose": pose})
        self.assertEqual(st, 422)
        self.assertIn("save anyway", msg)
        st, r = self.call({"gesture": "VN_STOP_RIGHT", "variant": "hand_up", "pose": pose, "force": True, "mirror": False})
        self.assertEqual(st, 200)
        self.assertEqual(r["saved"], ["poses/body/stop_handup_r.json"])     # mirror off: the left twin is left alone
        self.assertTrue(r["warnings"])
        self.assertEqual(self.call({"gesture": "VN_NOPE", "variant": "x", "pose": pose})[0], 400)
        self.assertEqual(self.call({"gesture": "VN_STOP_RIGHT", "variant": "nope", "pose": pose})[0], 400)

    def test_mirror_pose_swaps_sides_and_negates_midline_turns(self):
        from tosdg.poses import mirror_name, mirror_pose
        m = mirror_pose({"upperarm_r": {"flexion": 30.0}, "spine_02": {"flexion": 5.0, "lateral": 4.0, "twist": -7.0}})
        self.assertEqual(m, {"upperarm_l": {"flexion": 30.0}, "spine_02": {"flexion": 5.0, "lateral": -4.0, "twist": 7.0}})
        self.assertEqual((mirror_name("hello_l"), mirror_name("hello_r"), mirror_name("arms_horizontal")), ("hello_r", "hello_l", None))


class PageScriptTests(unittest.TestCase):
    def test_the_page_script_parses(self):
        """A syntax error anywhere in the page's one big script stops the whole Studio from loading (a '//' comment pasted into the
        middle of a one-line statement did once). Checked with Node when it is installed."""
        import re
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        html = (ROOT / "src" / "tosdg" / "studio" / "static" / "index.html").read_text(encoding="utf-8")
        scripts = re.findall(r"<script>(.*?)</script>", html.split("-->", 1)[1], re.S)             # skip the contents comment at the top
        self.assertTrue(scripts)
        with tempfile.TemporaryDirectory() as td:
            for i, js in enumerate(scripts):
                f = Path(td) / f"s{i}.js"
                f.write_text(js, encoding="utf-8")
                r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[:800])

