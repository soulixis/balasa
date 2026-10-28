"""Shared render engine: camera moves, dissolves, paint texture, film finishing, parallel render.

A film is a list of shots: (fn, start_s, duration_s, dissolve_in_s).
fn(t) returns a list of layers: (frame HxWx3 float, (cx, cy, zoom), weight, post)
where post(draw, map_xy, zoom, layer_image) draws sharp details after the camera move.
"""
import math
import multiprocessing as mp
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lib import BOIL, FPS, H, OH, OW, W, X, Y, clip, ease, from_pil, over, stipple, stroke_mod, to_pil, vnoise

A = {}
CFG = {}


def overlay(f, draw_fn):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(lay), lay)
    return composite(f, lay)


def composite(f, lay):
    a = np.asarray(lay, np.float32) / 255
    return over(f, a[..., :3], a[..., 3])


def build_finish(style="pastel"):
    A["warp"], A["paint"], A["tooth"], A["speck"] = [], [], [], []
    for i in range(BOIL):
        amp = 8 if style == "pastel" else 5
        dx = (vnoise(300 + i, 12, 2) - 0.5) * amp
        dy = (vnoise(310 + i, 12, 2) - 0.5) * amp
        A["warp"].append((np.clip(Y + dy, 0, H - 1).astype(np.int32), np.clip(X + dx, 0, W - 1).astype(np.int32)))
        if style == "oil":
            sm = stroke_mod(340 + i, 9000, 34, -0.7, 0.6, width=5)
            sm = from_pil(Image.fromarray(((sm + 1) * 127).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))) * 2 - 1
            relief = sm - np.roll(sm, 3, axis=0)          # impasto ridges lit from above
            weave = np.sin(X * 1.9) * np.sin(Y * 1.9)
            paint = 1 + 0.07 * sm + 0.12 * relief + 0.16 * (vnoise(350 + i, 5) - 0.5) + 0.025 * weave
        else:
            paint = (1 + 0.13 * stroke_mod(340 + i, 16000, 18, -1.0, 0.35) + 0.22 * (vnoise(350 + i, 5) - 0.5)
                     + 0.25 * (stipple(360 + i, 0.05) - 0.05))
        A["paint"].append(paint.astype(np.float32))
        g = np.random.default_rng(320 + i).random((H, W)).astype(np.float32)
        tooth_amp = 0.4 if style == "pastel" else 0.22
        A["tooth"].append(1 - tooth_amp / 2 + tooth_amp * (from_pil(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9))) * 2 - 0.5))
        A["speck"].append(stipple(330 + i, 0.03))
    oy, ox = np.mgrid[0:OH, 0:OW].astype(np.float32)
    A["vig"] = 1 - 0.35 * clip(((ox - OW / 2) / (OW / 2)) ** 2 + ((oy - OH / 2) / (OH / 2)) ** 2) ** 1.5
    CFG["style"] = style


def camera(frame, cam, v, post):
    yi, xi = A["warp"][v]
    f = frame[yi, xi] * A["paint"][v][..., None]
    cx, cy, z = cam
    z = max(z, 1.0)
    cw, chh = W / z, H / z
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - chh / 2, 0), H - chh)
    im = to_pil(f)
    if z > 1.001:
        im = im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + chh))
    if post:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        post(ImageDraw.Draw(lay), lambda x, y: ((x - x0) * z, (y - y0) * z), z, lay)
        im = Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")
    return from_pil(im)


def layers_at(T):
    shots = CFG["shots"]
    out = []
    for k, (fn, st, dur, xin) in enumerate(shots):
        nxt = shots[k + 1][3] if k + 1 < len(shots) else 0
        if not (st <= T < st + dur + nxt):
            continue
        w = 1.0
        if xin and k > 0 and T < st + xin:
            w = ease((T - st) / xin)
        if nxt and T >= st + dur:
            w = 1 - ease((T - st - dur) / nxt)
        for f, cam, lw, post in fn(T - st):
            out.append((f, cam, w * lw, post))
    return out


def render(i):
    T = i / FPS
    v = (i // 3) % BOIL
    acc = np.zeros((H, W, 3), np.float32)
    lays = layers_at(T)
    for f, cam, w, post in lays:
        if w > 0.001:
            acc += camera(f, cam, v, post) * w
    f = acc * A["tooth"][v][..., None] + 0.04 * A["speck"][v][..., None] * (1 - acc.mean(-1, keepdims=True))
    f *= min(1.0, T / 1.0)
    rng = np.random.default_rng(5000 + i)
    f *= 1 + 0.015 * math.sin(i * 0.7) + rng.normal(0, 0.006)
    if CFG["style"] == "oil":  # faded vintage stock: lifted blacks, warm cast
        f = f * np.array([0.93, 0.90, 0.84]) + np.array([0.035, 0.03, 0.02])
    im = to_pil(f).resize((OW, OH), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    if f.max() > 0.04:
        a += rng.normal(0, 5, (OH, OW, 1))
        a *= A["vig"][..., None]
        if rng.random() < 0.25:
            a = np.roll(a, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), (0, 1))
        for _ in range(rng.poisson(0.5)):
            yy, xx = rng.integers(0, OH), rng.integers(0, OW)
            a[max(0, yy - 1):yy + 1, max(0, xx - 1):xx + 1] = 190
        if CFG["style"] == "oil" and rng.random() < 0.06:   # a print scratch
            x = int(rng.integers(20, OW - 20))
            a[:, x] = a[:, x] * 0.6 + 80
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(CFG["outdir"], f"f_{i:04d}.png"), compress_level=1)
    return i


def run(shots, total, outdir, frames, builders, style):
    CFG["shots"], CFG["outdir"] = shots, outdir
    os.makedirs(outdir, exist_ok=True)
    build_finish(style)
    for b in builders:
        b()
        print("built", b.__name__, flush=True)
    frames = frames or list(range(total))
    with mp.get_context("fork").Pool(4) as p:
        for n, _ in enumerate(p.imap_unordered(render, frames, chunksize=2)):
            if n % 120 == 0:
                print("frame", n, flush=True)
    print("done")
