"""Reusable scene pieces. Colour images are float arrays (rows, cols, 3)."""

import numpy as np

from .noise import fbm1, fbm2, noise1, noise2, smooth


def gradient(rows, cols, top, bottom, power=1.5):
    """Vertical colour gradient from `top` to `bottom`."""
    y = (np.arange(rows, dtype=np.float32) / max(rows, 1))[:, None, None]
    top, bottom = np.asarray(top, np.float32), np.asarray(bottom, np.float32)
    return np.broadcast_to(top + (bottom - top) * y ** power, (rows, cols, 3)).copy()


def tone_map(img, exposure=1.7):
    """Squash unbounded brightness into 0..1 without hard clipping."""
    return 1 - np.exp(-exposure * img)


class Stars:
    """Twinkling stars. `add(img, t, glow)` dims stars where `glow` is bright."""

    def __init__(self, rows, cols, density=1 / 28, brightness=0.5, seed=1):
        rng = np.random.default_rng(seed)
        n = int(rows * cols * density)
        self.x = rng.integers(0, cols, n)
        self.y = (rng.random(n) ** 1.4 * max(rows - 2, 1)).astype(int)
        self.b = rng.random(n).astype(np.float32) ** 3 * brightness + 0.08
        self.f = rng.uniform(0.6, 3.0, n).astype(np.float32)
        self.p = rng.uniform(0, 6.28, n).astype(np.float32)

    def add(self, img, t, glow=None):
        tw = self.b * (0.65 + 0.35 * np.sin(t * self.f + self.p))
        if glow is not None:
            tw = tw * (1 - np.clip(glow[self.y, self.x].sum(-1), 0, 1))
        np.add.at(img, (self.y, self.x), tw[:, None])
        return img


class Treeline:
    """Dark pine silhouette sitting on a horizon row of the full image."""

    def __init__(self, cols, height=1.0):
        c = np.arange(cols)
        self.h = height * (0.6 + 2.6 * fbm1(c * 0.09) * (0.4 + 0.6 * noise1(c * 1.7)) ** 2
                           + 1.5 * smooth(fbm1(c * 0.02 + 50), 0.45, 0.7))

    def draw(self, img, horizon_row, colour=0.006):
        rows = np.arange(img.shape[0])[:, None]
        img[(rows <= horizon_row) & (rows > horizon_row - self.h[None, :])] = colour
        return img


class FrozenLake:
    """Ground plane seen in perspective: mirrors the sky, with snow patches and cracks.

    render(sky, t, ambient) takes the sky image above the horizon and returns the lake image.
    """

    def __init__(self, rows, cols, snow_colour=(0.55, 0.72, 0.90), snow_amount=0.5,
                 haze_colour=(0.02, 0.06, 0.15)):
        self.rows, self.cols = rows, cols
        self.snow_colour = np.asarray(snow_colour, np.float32)
        self.haze_colour = np.asarray(haze_colour, np.float32)
        v = ((np.arange(rows, dtype=np.float32) + 0.5) / rows)[:, None]   # 0 far, 1 near
        depth = 1 / (v + 0.07)
        u = (np.arange(cols, dtype=np.float32)[None, :] - cols / 2) / cols * depth * 7
        w = depth * 4.5
        self.v = v[..., None]
        lo = 1.0 - snow_amount                                            # more snow = lower cut
        self.snow = smooth(fbm2(u * 0.55 + 9, w * 0.55), lo, lo + 0.10)[..., None]
        crack = noise2(u * 1.7 + 70, w * 1.7 + 3)
        self.crack = (np.exp(-((crack - 0.5) / 0.012) ** 2) * (1 - self.snow[..., 0]))[..., None]
        self.col_idx = np.arange(cols)[None, :]
        self.row_idx = np.arange(rows)

    def render(self, sky, t, ambient):
        """ambient: RGB colour cast from the sky, e.g. sky_glow.mean(axis=(0, 1))."""
        sky_rows, v = sky.shape[0], self.v
        ripple = (noise1(self.row_idx * 0.9 + t * 0.6) - 0.5) * 5 * v[:, 0, 0]
        xs = np.clip(self.col_idx + ripple[:, None], 0, self.cols - 1).astype(int)
        refl = np.zeros((self.rows, self.cols, 3), np.float32)
        for k, wgt in ((-2, 0.25), (0, 0.5), (3, 0.25)):                  # vertical smear
            ry = sky_rows - 1 - self.row_idx * (sky_rows / self.rows) * 0.95 + k
            refl += wgt * sky[np.clip(ry, 0, sky_rows - 1).astype(int)[:, None], xs]
        ice = refl * (0.9 - 0.3 * v) + 0.015 + self.crack * (0.10 + 1.5 * ambient)
        snow = (self.snow_colour * (0.16 + 1.5 * ambient.mean()) + 0.8 * ambient) * (0.6 + 0.4 * v)
        lake = ice * (1 - 0.9 * self.snow) + snow * self.snow
        haze = np.exp(-v * 9)
        return lake * (1 - haze) + (self.haze_colour + 0.3 * ambient) * haze
