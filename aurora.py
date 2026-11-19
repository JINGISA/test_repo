"""ASCII aurora over a frozen lake, drawn with numpy.

    python aurora.py                  # live window (needs pygame)
    python aurora.py --out aurora.mp4 # render a video (needs ffmpeg)
    python aurora.py --png frame.png  # one still frame (needs Pillow only)

Every frame is computed as whole-array numpy operations on a grid of
character cells, then each cell is stamped with a glyph picked by brightness.
"""

import argparse
import glob
import string
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 5, 8          # pixels per character cell
HORIZON = 0.55                 # fraction of the screen that is sky
BACKGROUND = 0.30              # how much colour shows behind each glyph

rng = np.random.default_rng(7)
TAB1 = rng.random(256).astype(np.float32)
TAB2 = rng.random((256, 256)).astype(np.float32)


# --------------------------------------------------------------- noise ----
def noise1(x):
    """Smooth 1D value noise in 0..1."""
    i = np.floor(x).astype(np.int64)
    f = (x - i).astype(np.float32)
    f = f * f * (3 - 2 * f)
    a, b = TAB1[i & 255], TAB1[(i + 1) & 255]
    return a + (b - a) * f


def fbm1(x):
    return (noise1(x) + 0.5 * noise1(x * 2.03 + 19.1) + 0.25 * noise1(x * 4.01 + 7.7)) / 1.75


def noise2(x, y):
    """Smooth 2D value noise in 0..1."""
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - xi).astype(np.float32), (y - yi).astype(np.float32)
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x0, x1, y0, y1 = xi & 255, (xi + 1) & 255, yi & 255, (yi + 1) & 255
    top = TAB2[y0, x0] + (TAB2[y0, x1] - TAB2[y0, x0]) * fx
    bot = TAB2[y1, x0] + (TAB2[y1, x1] - TAB2[y1, x0]) * fx
    return top + (bot - top) * fy


def fbm2(x, y):
    return (noise2(x, y) + 0.5 * noise2(x * 2.1 + 31, y * 2.1 + 17)
            + 0.25 * noise2(x * 4.3 + 5, y * 4.3 + 53)) / 1.75


def smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------- glyphs ---
def build_atlas(levels=18):
    """Render printable ASCII, sort by ink coverage, keep an even ramp."""
    fonts = glob.glob("/usr/share/fonts/**/DejaVuSansMono-Bold.ttf", recursive=True) \
        + glob.glob("/usr/share/fonts/**/*Mono*.ttf", recursive=True)
    try:
        font = ImageFont.truetype(fonts[0], CELL_H) if fonts else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    glyphs = []
    for ch in string.printable[:94]:
        img = Image.new("L", (CELL_W, CELL_H), 0)
        ImageDraw.Draw(img).text((CELL_W / 2, CELL_H / 2), ch, 255, font=font, anchor="mm")
        glyphs.append(np.asarray(img, np.float32) / 255)
    glyphs.sort(key=lambda g: g.sum())
    picks = np.linspace(0, len(glyphs) - 1, levels - 1).astype(int)
    blank = np.zeros((CELL_H, CELL_W), np.float32)
    return np.stack([blank] + [glyphs[i] for i in picks])


# --------------------------------------------------------------- scene ----
#           base  amp   freq  speed height gain  seed
CURTAINS = [(0.48, 0.26, 2.0, 0.090, 0.50, 1.25, 0.0),
            (0.66, 0.20, 2.9, -0.070, 0.38, 0.95, 41.3),
            (0.84, 0.10, 4.2, 0.110, 0.24, 0.70, 87.9)]

GREEN = np.array([0.10, 1.00, 0.55], np.float32)
BLUE = np.array([0.08, 0.45, 1.00], np.float32)
PINK = np.array([1.00, 0.22, 0.55], np.float32)
SKY_TOP = np.array([0.008, 0.016, 0.060], np.float32)
SKY_LOW = np.array([0.020, 0.060, 0.150], np.float32)
SNOW = np.array([0.55, 0.72, 0.90], np.float32)


class Scene:
    def __init__(self, width, height):
        self.W, self.H = width // CELL_W, height // CELL_H
        self.Hs = int(self.H * HORIZON)          # sky rows
        self.Hl = self.H - self.Hs               # lake rows
        W, Hs, Hl = self.W, self.Hs, self.Hl
        self.atlas = build_atlas()

        self.x = np.linspace(0, 3.2, W, dtype=np.float32)            # sky x
        self.y = (np.arange(Hs, dtype=np.float32) / Hs)[:, None]     # 0 top, 1 horizon
        self.sky_base = SKY_TOP + (SKY_LOW - SKY_TOP) * self.y[..., None] ** 1.5

        n = W * Hs // 28
        self.star_x = rng.integers(0, W, n)
        self.star_y = (rng.random(n) ** 1.4 * (Hs - 2)).astype(int)
        self.star_b = rng.random(n).astype(np.float32) ** 3 * 0.5 + 0.08
        self.star_f = rng.uniform(0.6, 3.0, n).astype(np.float32)
        self.star_p = rng.uniform(0, 6.28, n).astype(np.float32)

        # pine silhouette along the far shore
        cols = np.arange(W)
        self.trees = (0.6 + 2.6 * fbm1(cols * 0.09) * (0.4 + 0.6 * noise1(cols * 1.7)) ** 2
                      + 1.5 * smooth(fbm1(cols * 0.02 + 50), 0.45, 0.7))

        # lake: perspective-project each cell onto the ice plane
        v = ((np.arange(Hl, dtype=np.float32) + 0.5) / Hl)[:, None]  # 0 horizon, 1 near
        depth = 1 / (v + 0.07)
        self.v = v
        self.u = (np.arange(W, dtype=np.float32)[None, :] - W / 2) / W * depth * 7
        self.w = depth * 4.5
        self.snow = smooth(fbm2(self.u * 0.55 + 9, self.w * 0.55), 0.50, 0.60)[..., None]
        crack = noise2(self.u * 1.7 + 70, self.w * 1.7 + 3)
        self.crack = (np.exp(-((crack - 0.5) / 0.012) ** 2) * (1 - self.snow[..., 0]))[..., None]
        self.cols = np.arange(W)[None, :]
        self.rows = np.arange(Hl)

    def aurora(self, t):
        x, y = self.x, self.y
        out = np.zeros((self.Hs, self.W, 3), np.float32)
        for base, amp, freq, speed, height, gain, seed in CURTAINS:
            # the curtain's lower hem wanders left/right and up/down over time
            hem = base + amp * 2 * (fbm1(x * freq + t * speed + seed) - 0.5)
            tall = height * (0.55 + 0.9 * noise1(x * 2.3 - t * 0.11 + seed))
            h = (hem[None, :] - y) / tall[None, :]                   # 0 at hem, 1 at top
            above = h >= 0
            # vertical rays: fine stripes drifting sideways along the curtain
            rays = 0.12 + 0.88 * (0.6 * noise1(x * 9 + t * 0.5 + seed)
                                  + 0.4 * noise1(x * 23 - t * 0.9 + seed)) ** 1.6
            act = 0.35 + 0.65 * smooth(noise1(x * 1.3 + t * 0.17 + seed * 2), 0.2, 0.8)
            body = np.where(above, np.exp(-2.0 * np.maximum(h, 0)), np.exp(40 * np.minimum(h, 0)))
            edge = np.exp(-(h / 0.05) ** 2)
            glow = gain * act * (rays * body + 0.35 * edge)
            hc = np.clip(h, 0, 2)[..., None]
            colour = GREEN * np.exp(-2.2 * hc) + BLUE * 1.6 * hc * np.exp(-1.6 * hc)
            out += glow[..., None] * colour
            # occasional crimson fringe at the top of the brightest folds
            crim = smooth(noise1(x * 0.9 + t * 0.08 + seed * 3), 0.6, 0.95)
            fringe = gain * crim * act * rays * np.exp(-((h - 0.95) / 0.45) ** 2) * above
            out += 0.9 * fringe[..., None] * PINK
        return out

    def frame(self, t):
        W, Hs, Hl = self.W, self.Hs, self.Hl
        aur = self.aurora(t)
        ambient = aur.mean(axis=(0, 1))                              # colour cast on the land
        sky = self.sky_base + aur + 0.25 * ambient * self.y[..., None] ** 3

        tw = self.star_b * (0.65 + 0.35 * np.sin(t * self.star_f + self.star_p))
        hidden = np.clip(aur[self.star_y, self.star_x].sum(-1), 0, 1)
        np.add.at(sky, (self.star_y, self.star_x), (tw * (1 - hidden))[:, None])

        # lake = mirrored sky, smeared a little, broken up by snow and cracks
        v = self.v
        ripple = (noise1(self.rows * 0.9 + t * 0.6) - 0.5) * 5 * v[:, 0]
        xs = np.clip(self.cols + ripple[:, None], 0, W - 1).astype(int)
        refl = np.zeros((Hl, W, 3), np.float32)
        for k, wgt in ((-2, 0.25), (0, 0.5), (3, 0.25)):
            ry = np.clip(Hs - 1 - self.rows * (Hs / Hl) * 0.95 + k, 0, Hs - 1).astype(int)
            refl += wgt * sky[ry[:, None], xs]
        ice = refl * (0.9 - 0.3 * v[..., None]) + 0.015
        ice += self.crack * (0.10 + 1.5 * ambient)
        snow = (SNOW * (0.16 + 1.5 * ambient.mean()) + 0.8 * ambient) * (0.6 + 0.4 * v[..., None])
        lake = ice * (1 - 0.9 * self.snow) + snow * self.snow
        haze = np.exp(-v[..., None] * 9)
        lake = lake * (1 - haze) + (SKY_LOW + 0.3 * ambient) * haze

        img = np.concatenate([sky, lake])
        rows = np.arange(self.H)[:, None]
        tree = (rows <= Hs) & (rows > Hs - self.trees[None, :])
        img[tree] = 0.006 + 0.03 * ambient

        img = 1 - np.exp(-1.7 * img)                                 # tone map
        return self.to_ascii(img)

    def to_ascii(self, img):
        lum = img @ np.array([0.25, 0.6, 0.15], np.float32)
        idx = np.clip(lum ** 0.6 * len(self.atlas), 0, len(self.atlas) - 1).astype(int)
        mask = self.atlas[idx]                                       # H, W, ch, cw
        px = (img ** 0.8)[:, :, None, None, :] * (BACKGROUND + mask[..., None])
        px = px.transpose(0, 2, 1, 3, 4).reshape(self.H * CELL_H, self.W * CELL_W, 3)
        return (np.clip(px, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------- output ---
def render_video(scene, path, seconds, fps):
    h, w = scene.H * CELL_H, scene.W * CELL_W
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(seconds * fps)):
        proc.stdin.write(scene.frame(i / fps).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit("ffmpeg failed")


def run_window(scene, fps):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((scene.W * CELL_W, scene.H * CELL_H))
    pygame.display.set_caption("aurora.py")
    clock, t = pygame.time.Clock(), 0.0
    while not any(e.type == pygame.QUIT for e in pygame.event.get()):
        pygame.surfarray.blit_array(screen, scene.frame(t).swapaxes(0, 1))
        pygame.display.flip()
        t += clock.tick(fps) / 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write an mp4 instead of opening a window")
    ap.add_argument("--png", help="write a single still frame")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--time", type=float, default=3.0, help="scene time for --png")
    args = ap.parse_args()
    width, height = map(int, args.size.lower().split("x"))
    scene = Scene(width, height)
    if args.png:
        Image.fromarray(scene.frame(args.time)).save(args.png)
    elif args.out:
        render_video(scene, args.out, args.seconds, args.fps)
    else:
        run_window(scene, args.fps)


if __name__ == "__main__":
    main()
