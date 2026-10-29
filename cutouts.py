"""Turn museum cut-outs into Balasa-style painted pieces: toned stone palette, torn edges, pastel grain."""
import os

import numpy as np
from PIL import Image, ImageFilter

from lib import clip

ASSETS = os.environ.get("BALASA_ASSETS", "/home/user/balasa_assets/assets/cut")
_cache = {}


def load(key):
    """key is the museum id, e.g. '152487' (Cleveland) — returns RGBA PIL image."""
    if key not in _cache:
        name = next(f for f in os.listdir(ASSETS) if f.endswith(f"_{key}.png"))
        _cache[key] = Image.open(os.path.join(ASSETS, name)).convert("RGBA")
    return _cache[key]


def tone(img, lo=(0.04, 0.06, 0.13), hi=(0.84, 0.84, 0.80), keep=0.3, gamma=0.95, seed=0, torn=True, grain=0.12):
    a = np.asarray(img, np.float32) / 255
    rgb, al = a[..., :3], a[..., 3]
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    m = al > 0.5
    if m.any():
        p2, p98 = np.percentile(lum[m], 2), np.percentile(lum[m], 98)
        lum = clip((lum - p2) / max(1e-3, p98 - p2))
    base = np.array(lo, np.float32) + (np.array(hi, np.float32) - np.array(lo, np.float32)) * lum[..., None] ** gamma
    out = base + (rgb - rgb.mean(-1, keepdims=True)) * keep * 1.6
    r = np.random.default_rng(seed)
    h, w = al.shape
    out *= (1 - grain / 2 + grain * r.random((h, w), np.float32))[..., None]
    dots = (r.random((h, w)) < 0.04).astype(np.float32)
    out += 0.10 * dots[..., None] * (1 - lum[..., None])
    if torn:
        n = r.random((max(2, h // 6), max(2, w // 6))).astype(np.float32)
        n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
        ab = np.asarray(Image.fromarray((al * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
        al = clip((ab - 0.5 + 0.35 * (n - 0.5)) * 6)
    res = np.dstack([clip(out), al])
    return Image.fromarray((res * 255).astype(np.uint8), "RGBA")


def scaled(img, height):
    s = height / img.height
    return img.resize((max(1, int(img.width * s)), max(1, int(height))), Image.LANCZOS)


def halo(img, color=(0.55, 0.65, 0.85), radius=40, strength=0.5):
    """Soft glow from the piece's silhouette, for an ethereal presence."""
    pad = radius * 2
    al = Image.new("L", (img.width + 2 * pad, img.height + 2 * pad), 0)
    al.paste(img.getchannel("A"), (pad, pad))
    g = np.asarray(al.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255 * strength
    rgba = np.dstack([np.broadcast_to(np.array(color, np.float32), g.shape + (3,)), g])
    return Image.fromarray((clip(rgba) * 255).astype(np.uint8), "RGBA"), pad
