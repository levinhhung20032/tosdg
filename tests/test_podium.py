"""The traffic podium prop (Studio preview): sits at the officer's feet, stays put when the officer is posed.

Contents:
  default_mesh()
  PodiumTests (4 tests)
  GeneratorTests (3 tests)
  StudioTests (1 tests)
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import ROOT, make_config, poses, skeleton
from tosdg import podium
from tosdg.generate import run
from tosdg.render_mesh import skin_mesh
from tosdg.scene import GenerationContext, build_scene
from tosdg.uniform_mesh import read_glb
from tosdg.validate import validate_dataset

VN = ROOT / "profiles" / "VN"


def default_mesh():
    uniforms = json.loads((VN / "uniforms.json").read_text(encoding="utf-8"))["uniforms"]
    uni = next(u for u in uniforms if u.get("mesh") and u.get("weight", 0) > 0)
    return read_glb(VN / uni["mesh"]["asset"])


class PodiumTests(unittest.TestCase):
    def test_the_rim_is_a_clean_ring(self):
        """The model's rim was a crown of jagged triangles; above the standing surface there is now only the rebuilt ring."""
        p = podium.load(podium.prop_path(VN))
        used = np.unique(p["tris"])
        self.assertEqual(len(used), len(p["pos"]))                             # no left-over vertices
        top = p["pos"][:, 2] > podium.RIM_TOL
        r = np.hypot(p["pos"][top, 0], p["pos"][top, 1])
        self.assertLess(float(np.ptp(p["pos"][top, 2])), 1e-6)                 # one flat top edge
        self.assertEqual(len(set(np.round(r, 3))), 2)                          # an inner and an outer wall, nothing else
        self.assertAlmostEqual(podium.size(p)["rim_cm"], float(p["pos"][top, 2].max()), places=3)

    def test_podium_is_centred_under_the_feet_with_its_standing_surface_at_zero(self):
        p = podium.load(podium.prop_path(VN))
        self.assertAlmostEqual(float(np.ptp(p["pos"][:, :2], axis=0).max()), podium.DIAMETER_CM, delta=1.0)
        self.assertLess(abs(float(p["pos"][:, 0].min() + p["pos"][:, 0].max())), 1.0)
        self.assertLess(float(p["pos"][:, 2].min()), -20.0)            # the body of the podium is below the feet
        self.assertGreater(float(np.linalg.norm(p["nrm"], axis=1).min()), 0.99)

    def test_the_shoes_rest_on_the_standing_surface(self):
        sk, mesh = skeleton(), default_mesh()
        prop = podium.load(podium.prop_path(VN))
        sole = podium.sole_level(mesh, sk)
        both = podium.mesh_with_podium(mesh, prop, sole)
        n0 = len(mesh.positions)
        surface = podium.size(prop)["depth_cm"]                                   # height of the standing surface above the ground
        lift = podium.lift_cm(prop, sole)                                         # the officer's root is lifted by this
        pos, _ = skin_mesh(both, sk.fk({}).transformed(np.eye(3), np.array([0.0, 0.0, lift])))
        self.assertAlmostEqual(float(pos[:n0, 2].min()), surface, places=2)       # the lowest point of the shoes touches the surface
        self.assertAlmostEqual(float(pos[n0:, 2].min()), 0.0, places=3)           # the podium itself stands on the ground
        top = pos[n0:][np.hypot(pos[n0:, 0], pos[n0:, 1]) < 15]                  # the middle of the platform
        self.assertAlmostEqual(float(top[:, 2].max()), surface, delta=1.0)

    def test_podium_does_not_move_with_the_pose_and_the_officer_is_unchanged(self):
        sk, pl = skeleton(), poses()
        mesh = default_mesh()
        both = podium.mesh_with_podium(mesh, podium.load(podium.prop_path(VN)))
        n0 = len(mesh.positions)
        self.assertEqual(len(both.triangles), len(mesh.triangles) + len(podium.load(podium.prop_path(VN))["tris"]))
        for pose in ({}, pl.body_pose("stop_bent_r")):
            pos, _ = skin_mesh(both, sk.fk(pose))
            ref, _ = skin_mesh(mesh, sk.fk(pose))
            np.testing.assert_allclose(pos[:n0], ref, atol=1e-6)                               # the officer is the same
            np.testing.assert_allclose(pos[n0:], skin_mesh(both, sk.fk({}))[0][n0:], atol=1e-6)  # the podium never moves


class GeneratorTests(unittest.TestCase):
    def test_podium_is_off_by_default_and_changes_no_other_draw(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = build_scene(GenerationContext(make_config(td, num_samples=1)), 1234)
            (td / "p").mkdir()
            on = build_scene(GenerationContext(make_config(td / "p", num_samples=1, podium={"probability": 1.0})), 1234)
            self.assertIsNone(base.podium)
            self.assertIsNotNone(on.podium)
            self.assertEqual((base.gesture["id"], base.sampled_pose.pose), (on.gesture["id"], on.sampled_pose.pose))   # the other streams are untouched
            np.testing.assert_allclose(base.camera.position, on.camera.position)
            self.assertAlmostEqual(float(on.officer_position[2]), on.podium["lift_cm"])                    # the officer stands on top of the podium
            self.assertAlmostEqual(float(on.fk_world.positions["foot_l"][2] - base.fk_world.positions["foot_l"][2]), on.podium["lift_cm"], places=3)

    def test_generation_records_the_podium_and_masks_only_the_officer(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            man = run(GenerationContext(make_config(td, num_samples=2, podium={"probability": 1.0}, max_attempts_per_sample=50)), td / "out", log=lambda *_: None)
            self.assertEqual(validate_dataset(td / "out"), [])
            (td / "n").mkdir()
            man0 = run(GenerationContext(make_config(td / "n", num_samples=2, max_attempts_per_sample=50)), td / "out0", log=lambda *_: None)
            for s, s0 in zip(man["samples"], man0["samples"]):
                meta = json.loads((td / "out" / s["metadata"]).read_text())
                self.assertEqual(meta["schema_version"], "1.8")
                self.assertTrue(meta["officer"]["podium"]["enabled"])
                self.assertAlmostEqual(meta["officer"]["world_position"][2], meta["officer"]["podium"]["lift_cm"], places=2)
                self.assertEqual(json.loads((td / "out0" / s0["metadata"]).read_text())["officer"]["podium"], {"enabled": False})
                mask = np.asarray(Image.open(td / "out" / meta["files"]["mask"]))
                rgb = np.asarray(Image.open(td / "out" / meta["files"]["rgb"]))
                self.assertGreater(int((mask > 0).sum()), 0)
                rgb0 = np.asarray(Image.open(td / "out0" / s0["files"]["rgb"] if "files" in s0 else td / "out0" / meta["files"]["rgb"]))
                self.assertGreater(int((rgb != rgb0).any(axis=-1).sum()), int((mask > 0).sum()) // 20)   # the podium is really drawn

    def test_podium_needs_the_mesh_backend(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                GenerationContext(make_config(Path(td), backend="proxy", podium={"probability": 0.5}))


class StudioTests(unittest.TestCase):
    def test_officer_on_the_podium_is_lifted_and_dragging_uses_that_plane(self):
        from tosdg.studio import core
        sk = skeleton()
        off0 = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "z": 0.0}
        off1 = {**off0, "z": 61.2}
        z0 = core.evaluate(sk, {}, off0, core.DEFAULT_CAMERA)["joints"]["foot_l"]["w"][2]
        z1 = core.evaluate(sk, {}, off1, core.DEFAULT_CAMERA)["joints"]["foot_l"]["w"][2]
        self.assertAlmostEqual(z1 - z0, 61.2, places=3)
        cam = core.make_camera(core.DEFAULT_CAMERA, off1)                 # the camera is measured from the ground: the officer's z does not move it
        self.assertAlmostEqual(cam.position[2], core.DEFAULT_CAMERA["height_cm"])
        uv = cam.project(np.array([[30.0, -20.0, 61.2]]))[0][0]
        np.testing.assert_allclose(core.ground_target(cam, uv, 61.2), [30.0, -20.0, 61.2], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
