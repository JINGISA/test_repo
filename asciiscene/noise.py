"""Smooth value noise on numpy arrays. All functions return values in 0..1."""

import numpy as np

_rng = np.random.default_rng(7)
_TAB1 = _rng.random(256).astype(np.float32)
_TAB2 = _rng.random((256, 256)).astype(np.float32)


def noise1(x):
    """1D noise. Period is 256 units of x."""
    x = np.asarray(x, np.float32)
    i = np.floor(x).astype(np.int64)
    f = x - i
    f = f * f * (3 - 2 * f)
    a, b = _TAB1[i & 255], _TAB1[(i + 1) & 255]
    return a + (b - a) * f


def fbm1(x):
    """1D noise with three octaves of detail."""
    x = np.asarray(x, np.float32)
    return (noise1(x) + 0.5 * noise1(x * 2.03 + 19.1) + 0.25 * noise1(x * 4.01 + 7.7)) / 1.75


def noise2(x, y):
    """2D noise. x and y must broadcast to the same shape."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32))
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - xi, y - yi
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x0, x1, y0, y1 = xi & 255, (xi + 1) & 255, yi & 255, (yi + 1) & 255
    top = _TAB2[y0, x0] + (_TAB2[y0, x1] - _TAB2[y0, x0]) * fx
    bot = _TAB2[y1, x0] + (_TAB2[y1, x1] - _TAB2[y1, x0]) * fx
    return top + (bot - top) * fy


def fbm2(x, y):
    """2D noise with three octaves of detail."""
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    return (noise2(x, y) + 0.5 * noise2(x * 2.1 + 31, y * 2.1 + 17)
            + 0.25 * noise2(x * 4.3 + 5, y * 4.3 + 53)) / 1.75


def smooth(x, lo, hi):
    """Smoothstep: 0 below lo, 1 above hi, eased in between."""
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)
