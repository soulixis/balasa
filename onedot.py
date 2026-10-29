"""ONE DOT — renders the reaction-diffusion run (rd_sim.py) as lit relief on toned paper.

Usage: python3 onedot.py <states.npy> <frames_dir> [frame numbers...]
"""
import math
import sys

import numpy as np
from PIL import Image

import engine
import relief as rl
from birds import texture
from engine import A
from lib import FPS, H, W, X, Y, clip, ease, lerp

TOTAL = 56 * FPS

# palette keyframes: time, background, surface low, surface high
KEYS = [
    (0, (0.04, 0.05, 0.11), (0.62, 0.60, 0.55), (0.96, 0.94, 0.87)),
    (21, (0.04, 0.05, 0.11), (0.62, 0.60, 0.55), (0.96, 0.94, 0.87)),
    (26, (0.02, 0.08, 0.12), (0.58, 0.14, 0.12), (0.98, 0.56, 0.42)),
    (30, (0.02, 0.08, 0.12), (0.58, 0.14, 0.12), (0.98, 0.56, 0.42)),
    (33, (0.08, 0.065, 0.055), (0.52, 0.37, 0.16), (0.98, 0.84, 0.52)),
    (40, (0.08, 0.065, 0.055), (0.52, 0.37, 0.16), (0.98, 0.84, 0.52)),
    (43, (0.30, 0.17, 0.07), (0.70, 0.68, 0.62), (1.0, 0.98, 0.92)),
    (47, (0.30, 0.17, 0.07), (0.70, 0.68, 0.62), (1.0, 0.98, 0.92)),
    (50, (0.04, 0.05, 0.11), (0.62, 0.60, 0.55), (0.96, 0.94, 0.87)),
    (56, (0.04, 0.05, 0.11), (0.62, 0.60, 0.55), (0.96, 0.94, 0.87)),
]


def palette(t):
    for (t0, *a), (t1, *b) in zip(KEYS[:-1], KEYS[1:]):
        if t0 <= t <= t1:
            k = ease((t - t0) / max(1e-6, t1 - t0))
            return [lerp(np.array(x, np.float32), np.array(y, np.float32), k) for x, y in zip(a, b)]
    return [np.array(x, np.float32) for x in KEYS[-1][1:]]


def build():
    A["rd"] = np.load(STATES, mmap_mode="r")
    A["paper"] = texture((1, 1, 1), 90, angle=-0.6, length=20, ss=0.12)
    A["glow"] = np.exp(-(((X - 640) / 520) ** 2 + ((Y - 480) / 420) ** 2))[..., None].astype(np.float32)


def camera(t):
    z = lerp(2.6, 1.0, ease(t / 7.0))
    z = lerp(z, 2.4, ease((t - 49) / 6.5))
    return (640 + 14 * math.sin(t * 0.21), 480 + 10 * math.sin(t * 0.17), z)


def shot(t):
    i = min(int(round(t * FPS)), A["rd"].shape[0] - 1)
    v = np.asarray(Image.fromarray(np.ascontiguousarray(A["rd"][i])).resize((W, H), Image.BICUBIC), np.float32) / 255
    v = rl.blur(v, 1.2)
    on = clip((v - 0.25) / 0.2)
    hf = v * 60
    gy, gx = np.gradient(hf)
    n = np.sqrt(gx * gx + gy * gy + 1)
    L = np.array([-0.45, -0.75, 0.55], np.float32)
    L /= np.linalg.norm(L)
    diff = (-gx * L[0] - gy * L[1] + L[2]) / n
    hv = L + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    spec = clip((-gx * hv[0] - gy * hv[1] + hv[2]) / n) ** 28
    lum = clip(1 + 1.9 * (diff - L[2]), 0.2, 1.6)
    bg, lo, hi = palette(t)
    surf = lo + (hi - lo) * clip(v * 1.3)[..., None]
    surf = surf * lum[..., None] + 0.25 * spec[..., None]
    shadow = rl.blur(np.roll(np.roll(on, 9, 0), 7, 1), 5)
    back = bg * A["paper"] * (1 - 0.55 * shadow[..., None])
    lace = clip((t - 41) / 2) * (1 - clip((t - 47) / 2.5))
    if lace > 0:  # warm light shining up through the holes
        back = back + np.array([0.55, 0.32, 0.12]) * A["glow"] * lace * (1 - 0.6 * shadow[..., None])
    f = back * (1 - on[..., None]) + surf * on[..., None]
    w = 1 - clip((t - 55) / 1.0)
    return [(clip(f), camera(t), w, None)]


if __name__ == "__main__":
    STATES = sys.argv[1]
    frames = [int(x) for x in sys.argv[3:]]
    engine.run([(shot, 0, 56, 0)], TOTAL, sys.argv[2], frames, [build], "pastel", boil=2)
