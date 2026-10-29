"""ONE DOT — reaction-diffusion (Gray-Scott) simulation, steered by chemistry schedules and image maps.

Writes every frame's pattern to an .npy file (uint8, frames x 300 x 400) for onedot.py to render.
Usage: python3 rd_sim.py <out.npy> [frames] [--preview dir]
"""
import math
import sys

import numpy as np
from PIL import Image

import cutouts as cu

GW, GH = 400, 300
FPS = 24
TOTAL = 56 * FPS
ITERS = 40

# (time s, feed, kill) — named Gray-Scott regimes (Sims' units: Da=1, Db=0.5)
MITOSIS, MAZE, CORAL, HOLES = (0.0367, 0.0649), (0.029, 0.057), (0.0545, 0.062), (0.039, 0.058)
SPOTS = (0.030, 0.0625)
DSCALE = 0.6


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def grid_map(img, height, cy=0.5):
    """Luminance*alpha of an RGBA image, scaled to `height` grid cells and centred; 0..1."""
    s = height / img.height
    im = img.resize((max(1, int(img.width * s)), int(height)), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    lum = (a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)) * a[..., 3]
    out = np.zeros((GH, GW), np.float32)
    y0, x0 = int(GH * cy - im.height / 2), int(GW / 2 - im.width / 2)
    ys, xs = slice(max(0, y0), min(GH, y0 + im.height)), slice(max(0, x0), min(GW, x0 + im.width))
    out[ys, xs] = lum[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    inside = out > 0.02
    m = out[inside]
    if m.size:
        lo, hi = np.percentile(m, 5), np.percentile(m, 98)
        out = np.where(inside, np.clip((out - lo) / (hi - lo + 1e-6), 0, 1), 0)
        bl = __import__("relief").blur(out, 10)
        out = np.where(inside, np.clip(0.5 + 2.2 * (out - bl) + 0.5 * (out - 0.5), 0, 1), 0)
    return out.astype(np.float32)


def maps():
    face = grid_map(cu.load("102361"), 290, 0.52)
    moth = cu.load("8785324")
    moth = moth.crop((int(moth.width * 0.06), 0, moth.width, int(moth.height * 0.82)))
    moth = grid_map(moth.crop(moth.getbbox()), 215, 0.5)
    return face, moth


def schedule(t, face, moth, yy, xx):
    """Feed and kill fields for time t."""
    def mix(a, b, w):
        return a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w

    f, k = MITOSIS
    target, tk = None, 0.0
    if t >= 7:
        w = smooth((t - 7) / 2.0)
        f, k = mix(MITOSIS, MAZE, w)
        target, tk = face, 0.015 * w
    if t >= 22:
        w = smooth((t - 22) / 4.0)
        f, k = mix(MAZE, CORAL, w)
        tk = 0.015 * (1 - w)
    if t >= 30:
        w = smooth((t - 30) / 2.0)
        f, k = mix(CORAL, MAZE, w)
        target, tk = moth, 0.015 * w
    if t >= 40:
        w = smooth((t - 40) / 2.0)
        f, k = mix(MAZE, HOLES, w)
    F = np.full((GH, GW), f, np.float32)
    K = np.full((GH, GW), k, np.float32)
    if target is not None and tk > 0:
        w = tk / 0.015
        tone = target
        wm = np.clip((tone - 0.42) / 0.25, 0, 1)
        bright = HOLES if t >= 40 else MAZE
        bf = np.where(True, SPOTS[0] + (bright[0] - SPOTS[0]) * wm, 0)
        bk = SPOTS[1] + (bright[1] - SPOTS[1]) * wm
        F = F * (1 - w) + bf * w
        K = K * (1 - w) + bk * w
        K += 0.022 * w * np.clip((0.24 - tone) / 0.12, 0, 1)
    if t >= 47:  # starvation from the edges inward
        r = np.sqrt(((xx - GW / 2) / (GW / 2)) ** 2 + ((yy - GH / 2) / (GH / 2)) ** 2)
        front = (t - 47) / 8.0
        K += 0.02 * np.clip((r - (1.2 - 1.3 * front)) / 0.15, 0, 1) + 0.006 * smooth((t - 50) / 4)
    return F, K


def lap(a):
    return (0.2 * (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
            + 0.05 * (np.roll(np.roll(a, 1, 0), 1, 1) + np.roll(np.roll(a, 1, 0), -1, 1)
                      + np.roll(np.roll(a, -1, 0), 1, 1) + np.roll(np.roll(a, -1, 0), -1, 1)) - a)


def run(out_path, frames=TOTAL, preview=None):
    face, moth = maps()
    yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
    U = np.ones((GH, GW), np.float32)
    V = np.zeros((GH, GW), np.float32)
    d = (xx - GW / 2) ** 2 + (yy - GH / 2) ** 2 < 16
    U[d], V[d] = 0.5, 0.25
    rng = np.random.default_rng(1)
    V += 0.01 * rng.random((GH, GW)).astype(np.float32) * d
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(frames, GH, GW))
    for i in range(frames):
        t = i / FPS
        F, K = schedule(t, face, moth, yy, xx)
        for _ in range(ITERS + (30 if t < 7 else 0)):
            uvv = U * V * V
            U += lap(U) * (1.0 * DSCALE) - uvv + F * (1 - U)
            V += lap(V) * (0.5 * DSCALE) + uvv - (F + K) * V
        np.clip(U, 0, 1, out=U)
        np.clip(V, 0, 1, out=V)
        out[i] = np.clip(V * 600, 0, 255).astype(np.uint8)
        if preview and i % 48 == 0:
            Image.fromarray(out[i]).save(f"{preview}/rd_{i:04d}.png")
        if i % 120 == 0:
            print("frame", i, flush=True)
    out.flush()


if __name__ == "__main__":
    args = sys.argv[1:]
    prev = None
    if "--preview" in args:
        j = args.index("--preview")
        prev = args[j + 1]
        args = args[:j] + args[j + 2:]
    run(args[0], int(args[1]) if len(args) > 1 else TOTAL, prev)
