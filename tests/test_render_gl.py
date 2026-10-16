"""The GPU mesh renderer (render_gl, mesh_renderer "gpu"): same geometry and mask as the CPU renderer, a better picture.

Skipped where no OpenGL 4.3 context can be made (no moderngl, no GPU / driver).

Contents:
  GpuRendererTests (7 tests)
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import ROOT, make_config, poses, skeleton
from tosdg.camera import Camera, look_at_rotator
from tosdg.generate import run
from tosdg.poses import merge
from tosdg.render_mesh import render as render_cpu
from tosdg.scene import GenerationContext
from tosdg.schema import load_json
from tosdg.uniform_mesh import read_glb
from tosdg.validate import validate_dataset

try:
    from tosdg import render_gl
    GPU_OK, GPU_WHY = render_gl.available()
except ImportError as e:                     # moderngl is not installed
    GPU_OK, GPU_WHY = False, str(e)

PROFILE = ROOT / "profiles" / "VN"
LIGHT = {"sun_direction": [0.3, -0.2, 0.93], "sun_color_rgb": [1, 1, 1], "sun_intensity": 0.8, "ambient_intensity": 0.4, "exposure_ev": 0.0}


def cam(az=20.0, dist=420.0, w=320, h=480):
    eye = np.array([dist * np.cos(np.radians(az)), dist * np.sin(np.radians(az)), 120.0])
    p, y = look_at_rotator(eye, [0, 0, 100])
    return Camera("t", tuple(eye), p, y, 0, 35, 36, w, h)


def pose_of(name, hand="relaxed"):
    pl = poses()
    return merge(pl.body_pose(name), pl.hand_pose(hand, "l"), pl.hand_pose(hand, "r"))


@unittest.skipUnless(GPU_OK, f"GPU renderer unavailable: {GPU_WHY}")
class GpuRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sk = skeleton("tripo_v1")
        cls.uni = next(u for u in load_json(PROFILE / "uniforms.json", "uniforms")["uniforms"] if u["id"] == "beta")
        cls.mesh = read_glb(PROFILE / cls.uni["mesh"]["asset"])
        cls.bg = Image.new("RGB", (320, 480), (120, 125, 130))

    def draw(self, fn, pose, c, **kw):
        return fn(self.mesh, self.sk.fk(pose), c, self.bg, self.uni["proxy_colors"], LIGHT, **kw)

    def test_the_mask_is_the_same_silhouette_as_the_cpu_renderer(self):
        """Same skinning, same projection: the instance masks agree (only edge pixels may differ by the rasteriser's tie rule)."""
        for name, az in (("neutral_stand", 0), ("stop_bent_r", 40), ("arms_horizontal", 120)):
            with self.subTest(pose=name):
                c = cam(az)
                _, m_cpu = self.draw(render_cpu, pose_of(name), c)
                _, m_gpu = self.draw(render_gl.render, pose_of(name), c)
                iou = (m_cpu & m_gpu).sum() / max((m_cpu | m_gpu).sum(), 1)
                self.assertGreater(m_cpu.sum(), 500)
                self.assertGreater(iou, 0.97)

    def test_deterministic(self):
        a = self.draw(render_gl.render, pose_of("stop_bent_r"), cam())
        b = self.draw(render_gl.render, pose_of("stop_bent_r"), cam())
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_the_officer_is_drawn_with_the_texture_and_the_background_is_kept(self):
        rgb, mask = self.draw(render_gl.render, pose_of("neutral_stand"), cam(0))
        px = rgb[mask].astype(float)
        self.assertGreater(px.std(axis=0).mean(), 15)                 # textured and shaded, not a flat silhouette
        self.assertGreater(px[:, 0].mean(), px[:, 2].mean() + 20)      # the khaki uniform: red over blue
        far = rgb[:40, :40].astype(int)
        np.testing.assert_array_equal(far, np.broadcast_to(np.array([120, 125, 130]), far.shape))   # untouched far from the officer

    def test_the_officer_casts_a_shadow_on_the_ground_outside_the_mask(self):
        rgb, mask = self.draw(render_gl.render, pose_of("neutral_stand"), cam(0))
        layer = render_gl.render_layer(self.mesh, self.sk.fk(pose_of("neutral_stand")), cam(0), self.uni["proxy_colors"], LIGHT)
        outside = layer[..., 3] == 0                                     # not even a fringe of the officer
        darker = (rgb.astype(int).sum(axis=2) < 3 * 120 - 15) & outside
        self.assertGreater(int(darker.sum()), 50)
        self.assertTrue((rgb[outside].astype(int) <= np.array([120, 125, 130]) + 1).all())   # the shadow only darkens

    def test_layer_composited_over_the_background_matches_the_render_without_ground_shadow(self):
        c = cam(30)
        old = render_gl.GROUND_SHADOW
        render_gl.GROUND_SHADOW = 0.0
        try:
            rgb, _ = self.draw(render_gl.render, pose_of("hello_r", "open_flat"), c)
        finally:
            render_gl.GROUND_SHADOW = old
        layer = render_gl.render_layer(self.mesh, self.sk.fk(pose_of("hello_r", "open_flat")), c, self.uni["proxy_colors"], LIGHT, backplate=self.bg)
        a = layer[..., 3:4].astype(float) / 255.0
        comp = layer[..., :3].astype(float) * a + np.asarray(self.bg, float) * (1 - a)
        self.assertLess(float(np.abs(comp - rgb.astype(float)).max()), 6.0)

    def test_generation_with_the_gpu_renderer_validates(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = GenerationContext(make_config(Path(td), backend="mesh", num_samples=2, mesh_renderer="gpu"))
            man = run(ctx, Path(td) / "out", log=lambda *_: None)
            self.assertEqual((len(man["samples"]), len(man["failures"])), (2, 0))
            self.assertEqual(validate_dataset(Path(td) / "out"), [])

    def test_mesh_renderer_needs_the_mesh_backend_and_a_known_name(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "mesh_renderer"):
                GenerationContext(make_config(Path(td), backend="proxy", mesh_renderer="gpu", uniform_brightness_range=[1.0, 1.0]))
            cfg = json.loads(make_config(Path(td)).read_text())
            from tosdg.schema import schema_errors
            self.assertTrue(schema_errors({**cfg, "mesh_renderer": "vulkan"}, "generation_config"))
            self.assertFalse(schema_errors({**cfg, "mesh_renderer": "gpu"}, "generation_config"))


if __name__ == "__main__":
    unittest.main()
