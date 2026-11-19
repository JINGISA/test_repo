"""Turn a float colour image (one pixel per character cell) into ASCII art pixels."""

import glob
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_LUMA = np.array([0.25, 0.6, 0.15], np.float32)


def _load_font(size):
    fonts = (glob.glob("/usr/share/fonts/**/DejaVuSansMono-Bold.ttf", recursive=True)
             + glob.glob("/usr/share/fonts/**/*Mono*.ttf", recursive=True)
             + glob.glob("C:/Windows/Fonts/consola*.ttf")
             + glob.glob("/System/Library/Fonts/Menlo.ttc"))
    for path in fonts:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class AsciiRenderer:
    """Stamps every cell with a glyph chosen by brightness, tinted with the cell colour.

    cell_w, cell_h: pixels per character. Smaller cells give a finer picture.
    background:     how much of the cell colour shows behind the glyph (0 = black).
    """

    def __init__(self, cell_w=5, cell_h=8, levels=18, background=0.30):
        self.cell_w, self.cell_h, self.background = cell_w, cell_h, background
        font = _load_font(cell_h)
        glyphs = []
        for ch in string.printable[:94]:
            img = Image.new("L", (cell_w, cell_h), 0)
            ImageDraw.Draw(img).text((cell_w / 2, cell_h / 2), ch, 255, font=font, anchor="mm")
            glyphs.append(np.asarray(img, np.float32) / 255)
        glyphs.sort(key=lambda g: g.sum())                 # sparse to dense
        picks = np.linspace(0, len(glyphs) - 1, levels - 1).astype(int)
        blank = np.zeros((cell_h, cell_w), np.float32)
        self.atlas = np.stack([blank] + [glyphs[i] for i in picks])

    def grid(self, width, height):
        """How many (columns, rows) of cells fit in a width x height pixel frame."""
        return width // self.cell_w, height // self.cell_h

    def render(self, img):
        """img: float array (rows, cols, 3) in 0..1. Returns uint8 RGB pixels."""
        rows, cols = img.shape[:2]
        lum = img @ _LUMA
        idx = np.clip(lum ** 0.6 * len(self.atlas), 0, len(self.atlas) - 1).astype(int)
        mask = self.atlas[idx]                             # rows, cols, cell_h, cell_w
        px = (img ** 0.8)[:, :, None, None, :] * (self.background + mask[..., None])
        px = px.transpose(0, 2, 1, 3, 4).reshape(rows * self.cell_h, cols * self.cell_w, 3)
        return (np.clip(px, 0, 1) * 255).astype(np.uint8)
