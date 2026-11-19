"""Aurora over a frozen lake."""

import numpy as np

from ..layers import FrozenLake, Stars, Treeline, gradient, tone_map
from ..noise import fbm1, noise1, smooth
from . import Scene


class Aurora(Scene):
    PARAMS = dict(
        horizon=0.55,                      # fraction of the frame that is sky
        speed=1.0,                         # overall animation speed
        brightness=1.0,                    # aurora strength
        pink=0.9,                          # strength of the crimson top fringe (0 = off)
        colour_low=(0.10, 1.00, 0.55),     # hem colour
        colour_high=(0.08, 0.45, 1.00),    # upper colour
        colour_fringe=(1.00, 0.22, 0.55),
        sky_top=(0.008, 0.016, 0.060),
        sky_low=(0.020, 0.060, 0.150),
        snow_amount=0.5,                   # 0 = bare ice, 1 = all snow
        star_density=1 / 28,
        trees=1.0,                         # treeline height scale (0 = none)
        #          base  amp   freq  drift  height gain  seed
        curtains=((0.48, 0.26, 2.0, 0.090, 0.50, 1.25, 0.0),
                  (0.66, 0.20, 2.9, -0.070, 0.38, 0.95, 41.3),
                  (0.84, 0.10, 4.2, 0.110, 0.24, 0.70, 87.9)),
    )

    def setup(self):
        p = self.p
        self.sky_rows = int(self.rows * p["horizon"])
        self.x = np.linspace(0, 3.2 * self.cols / 384, self.cols, dtype=np.float32)
        self.y = (np.arange(self.sky_rows, dtype=np.float32) / self.sky_rows)[:, None]
        self.sky_base = gradient(self.sky_rows, self.cols, p["sky_top"], p["sky_low"])
        self.stars = Stars(self.sky_rows, self.cols, p["star_density"])
        self.trees = Treeline(self.cols, p["trees"])
        self.lake = FrozenLake(self.rows - self.sky_rows, self.cols,
                               snow_amount=p["snow_amount"], haze_colour=p["sky_low"])
        self.low, self.high, self.fringe = (np.asarray(p[k], np.float32) for k in
                                            ("colour_low", "colour_high", "colour_fringe"))

    def aurora(self, t):
        x, y, p = self.x, self.y, self.p
        out = np.zeros((self.sky_rows, self.cols, 3), np.float32)
        for base, amp, freq, drift, height, gain, seed in p["curtains"]:
            gain *= p["brightness"]
            # the curtain's lower hem wanders sideways and up/down over time
            hem = base + amp * 2 * (fbm1(x * freq + t * drift + seed) - 0.5)
            tall = height * (0.55 + 0.9 * noise1(x * 2.3 - t * 0.11 + seed))
            h = (hem[None, :] - y) / tall[None, :]                   # 0 at hem, 1 at top
            above = h >= 0
            # vertical rays: fine stripes drifting along the curtain
            rays = 0.12 + 0.88 * (0.6 * noise1(x * 9 + t * 0.5 + seed)
                                  + 0.4 * noise1(x * 23 - t * 0.9 + seed)) ** 1.6
            act = 0.35 + 0.65 * smooth(noise1(x * 1.3 + t * 0.17 + seed * 2), 0.2, 0.8)
            body = np.where(above, np.exp(-2.0 * np.maximum(h, 0)), np.exp(40 * np.minimum(h, 0)))
            edge = np.exp(-(h / 0.05) ** 2)
            glow = gain * act * (rays * body + 0.35 * edge)
            hc = np.clip(h, 0, 2)[..., None]
            colour = self.low * np.exp(-2.2 * hc) + self.high * 1.6 * hc * np.exp(-1.6 * hc)
            out += glow[..., None] * colour
            # occasional crimson fringe on top of the brightest folds
            crim = smooth(noise1(x * 0.9 + t * 0.08 + seed * 3), 0.6, 0.95)
            top = gain * crim * act * rays * np.exp(-((h - 0.95) / 0.45) ** 2) * above
            out += p["pink"] * top[..., None] * self.fringe
        return out

    def image(self, t):
        t = t * self.p["speed"]
        glow = self.aurora(t)
        ambient = glow.mean(axis=(0, 1))                             # colour cast on the land
        sky = self.sky_base + glow + 0.25 * ambient * self.y[..., None] ** 3
        self.stars.add(sky, t, glow)
        img = np.concatenate([sky, self.lake.render(sky, t, ambient)])
        if self.p["trees"] > 0:
            self.trees.draw(img, self.sky_rows, 0.006 + 0.03 * ambient)
        return tone_map(img)
