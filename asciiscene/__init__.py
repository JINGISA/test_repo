"""ASCII scene toolkit: noise, reusable layers, an ASCII renderer and outputs."""

from .ascii import AsciiRenderer
from .output import run_window, save_png, save_video
from .scenes import SCENES, Scene


def make_frame_fn(scene_name, width=1920, height=1080, cell=(5, 8), **params):
    """Build frame_fn(t) -> uint8 RGB pixels for a named scene."""
    renderer = AsciiRenderer(*cell)
    scene = SCENES[scene_name](*renderer.grid(width, height), **params)
    return lambda t: renderer.render(scene.image(t))


__all__ = ["AsciiRenderer", "Scene", "SCENES", "make_frame_fn",
           "save_png", "save_video", "run_window"]
