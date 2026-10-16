"""Which program draws the skinned mesh: "cpu" (render_mesh, numpy, runs anywhere) or "gpu" (render_gl, OpenGL 4.3 on the graphics card,
PBR with the model's normal/roughness maps, soft shadows, image-based light, ambient occlusion). Same inputs, same geometry and ground truth.

Contents:
  constants: NAMES
  get — (render, render_layer) of a mesh renderer by name
  default_name — "gpu" when a GPU context can be made here, else "cpu"
"""
from __future__ import annotations

NAMES = ("cpu", "gpu")


def get(name: str):
    """(render, render_layer) functions of the mesh renderer `name`. "gpu" raises RuntimeError when no OpenGL 4.3 context can be made."""
    if name == "cpu":
        from .render_mesh import render, render_layer
        return render, render_layer
    if name == "gpu":
        from . import render_gl
        ok, why = render_gl.available()
        if not ok:
            raise RuntimeError(why)
        return render_gl.render, render_gl.render_layer
    raise ValueError(f"mesh renderer must be one of {list(NAMES)}, not '{name}'")


def default_name() -> str:
    try:
        from . import render_gl
        return "gpu" if render_gl.available()[0] else "cpu"
    except ImportError:                   # moderngl is not installed
        return "cpu"
