"""Shared drawing helpers: textures, masks, colour utilities."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 960          # internal render size (2x output)
OW, OH = 640, 480         # output size, 4:3
FPS = 24
BOIL = 3                  # texture variants that cycle, like hand-drawn line boil

Y, X = np.mgrid[0:H, 0:W].astype(np.float32)


def clip(a, lo=0.0, hi=1.0):
    return np.clip(a, lo, hi)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    """0 before a, 1 after b, eased in between."""
    return ease((t - a) / (b - a))


def lerp(a, b, k):
    return a + (b - a) * k


def col(c, k=1.0):
    return tuple(int(255 * min(1.0, max(0.0, x * k))) for x in c)


def cola(c, a, k=1.0):
    return col(c, k) + (int(255 * min(1.0, max(0.0, a))),)


def vnoise(seed, cells, octaves=4, w=W, h=H):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cw = cells * 2 ** o
        ch = max(2, int(cw * h / w))
        a = (rng.random((ch, cw)) * 255).astype(np.uint8)
        im = Image.fromarray(a).resize((w, h), Image.BICUBIC)
        acc += amp * np.asarray(im, np.float32) / 255
        tot += amp
        amp *= 0.5
    return acc / tot


def stroke_mod(seed, n, length, angle, jit, width=2, w=W, h=H):
    """Field of short pastel strokes, as modulation in [-1, 1]."""
    rng = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 128)
    d = ImageDraw.Draw(im)
    xs, ys = rng.uniform(-20, w, n), rng.uniform(-20, h, n)
    ang = angle + rng.normal(0, jit, n)
    ln = rng.uniform(0.4, 1.0, n) * length
    vals = rng.integers(70, 190, n)
    for x, y, a, l, v in zip(xs, ys, ang, ln, vals):
        d.line([(x, y), (x + math.cos(a) * l, y + math.sin(a) * l)], fill=int(v), width=width)
    return (np.asarray(im, np.float32) - 128) / 128


def stipple(seed, density, w=W, h=H):
    rng = np.random.default_rng(seed)
    dots = (rng.random((h, w)) < density).astype(np.uint8) * 255
    im = Image.fromarray(dots).filter(ImageFilter.GaussianBlur(0.8))
    return np.asarray(im, np.float32) / 255


def over(dst, src, m):
    m = m[..., None]
    return dst * (1 - m) + src * m


def to_pil(a):
    return Image.fromarray((clip(a) * 255).astype(np.uint8))


def from_pil(im):
    return np.asarray(im, np.float32) / 255


def vgrad(top, bot, y0=0, y1=H, power=1.0):
    k = clip((Y[:, :1] - y0) / max(1, y1 - y0)) ** power
    g = np.array(top, np.float32) * (1 - k[..., None]) + np.array(bot, np.float32) * k[..., None]
    return np.broadcast_to(g, (H, W, 3)).copy()


def blur_l(im, r):
    return from_pil(im.filter(ImageFilter.GaussianBlur(r)))


def add_glow(f, mask_im, color, radius, strength):
    g = blur_l(mask_im, radius)
    return f + np.array(color, np.float32) * (g * strength)[..., None]


def font(size, bold=False):
    name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", int(size))
    except OSError:
        return ImageFont.load_default()


def almond_pts(cx, cy, hw, hh, n=24):
    top = [(cx - hw + 2 * hw * i / n, cy - hh * math.sin(math.pi * i / n)) for i in range(n + 1)]
    bot = [(cx + hw - 2 * hw * i / n, cy + hh * math.sin(math.pi * i / n)) for i in range(n + 1)]
    return top + bot
