"""Relief surfaces: height fields lit like painted bas-relief (pebbles, strata, craters, curls)."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lib import H, W, clip, from_pil

RNG = np.random.default_rng


def _box(a, k, axis):
    if k < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (k + 1, k)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * k + 1, 2 * k + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * k + 1)).astype(np.float32)


def blur(a, r):
    """Gaussian-like blur: three box passes per axis."""
    k = max(1, int(round(r * 0.8)))
    a = a.astype(np.float32)
    for _ in range(3):
        a = _box(_box(a, k, 0), k, 1)
    return a


def mask(draw_fn, w=W, h=H, r=0):
    m = Image.new("L", (w, h), 0)
    draw_fn(ImageDraw.Draw(m))
    if r:
        m = m.filter(ImageFilter.GaussianBlur(r))
    return from_pil(m)


def pebbles(seed, n, rmin, rmax, w=W, h=H, squash=1.0, height=0.8):
    """Soft rounded stones packed together (max of domes): crevices between them read as relief."""
    r = RNG(seed)
    hf = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = r.uniform(-30, w + 30), r.uniform(-30, h + 30)
        rx = r.uniform(rmin, rmax)
        ry = rx * r.uniform(0.6, 1.1) * squash
        a = r.uniform(0, math.pi)
        R = int(max(rx, ry)) + 2
        x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R)
        y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
        v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
        dome = np.sqrt(clip(1 - (u / rx) ** 2 - (v / ry) ** 2)) * min(rx, ry) * height
        hf[y0:y1, x0:x1] = np.maximum(hf[y0:y1, x0:x1], dome)
    return hf


def craters(seed, n, rmin, rmax, w=W, h=H):
    r = RNG(seed)
    hf = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy, rr = r.uniform(0, w), r.uniform(0, h), r.uniform(rmin, rmax)
        R = int(rr * 1.6) + 2
        x0, x1, y0, y1 = max(0, int(cx) - R), min(w, int(cx) + R), max(0, int(cy) - R), min(h, int(cy) + R)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.6) ** 2) / rr
        hf[y0:y1, x0:x1] += rr * 0.35 * (np.exp(-((d - 1) / 0.22) ** 2) - 0.8 * np.exp(-(d / 0.7) ** 2))
    return hf


def shade(hf, lo, mid, hi, light=(-0.45, -0.75, 0.55), strength=1.0, cav=8.0, spec=0.22, cav_r=6):
    """Light a height field and colour it with a three-stop ramp; crevices darken."""
    gy, gx = np.gradient(hf)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(hf)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / n, ny / n, nz / n
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    diff = clip(nx * L[0] + ny * L[1] + nz * L[2])
    hv = L + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    sp = clip(nx * hv[0] + ny * hv[1] + nz * hv[2]) ** 30
    c = clip(-(hf - blur(hf, cav_r)) / cav)
    t = clip(diff * (1 - 0.65 * c))[..., None]
    lo, mid, hi = (np.array(x, np.float32) for x in (lo, mid, hi))
    rgb = np.where(t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5))
    return clip(rgb + spec * sp[..., None])
