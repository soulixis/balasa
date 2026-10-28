"""Procedural short in the style of Sabin Balasa's painted animation (1960s).

Renders frames at 2x resolution, then downsamples for a soft, hand-made look.
Usage: python3 render.py <frames_dir>
"""
import math
import multiprocessing as mp
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1280, 960          # internal render size
OW, OH = 640, 480         # output size (4:3)
FPS = 24
TOTAL = 20 * FPS          # 480 frames
BOIL = 3                  # texture variants that cycle, like hand-drawn line boil

Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
A = {}                    # precomputed assets, shared with workers via fork


# ---------- helpers ----------
def clip(a, lo=0.0, hi=1.0):
    return np.clip(a, lo, hi)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


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
    """Field of short pastel/pencil strokes, returned as modulation in [-1, 1]."""
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


def texture(base, seed, n=14000, length=16, angle=-1.0, jit=0.3, sn=0.22, ss=0.16,
            st=0.0, grain=0.05, w=W, h=H):
    rng = np.random.default_rng(seed + 99)
    f = (1 + (vnoise(seed, 5, w=w, h=h) - 0.5) * 2 * sn
         + stroke_mod(seed + 1, n, length, angle, jit, w=w, h=h) * ss
         + rng.normal(0, grain, (h, w)).astype(np.float32))
    if st:
        f += stipple(seed + 2, 0.06, w=w, h=h) * st
    return clip(np.array(base, np.float32)[None, None, :] * f[..., None])


def over(dst, src, m):
    m = m[..., None]
    return dst * (1 - m) + src * m


def poly_mask(pts, blur=1.5, w=W, h=H):
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im, np.float32) / 255


def almond(cx, cy, hw, hh, xx=X, yy=Y):
    u = (xx - cx) / hw
    hgt = hh * clip(1 - u * u)
    return clip((hgt - np.abs(yy - cy)) / 1.5 + 0.5) * (np.abs(u) < 1)


def to_pil(a):
    return Image.fromarray((clip(a) * 255).astype(np.uint8))


def from_pil(im):
    return np.asarray(im, np.float32) / 255


# ---------- scene 1: egg opens into a crimson eye ----------
S1 = dict(cx=640, cy=470, rx=300, ry=235)


def build_s1():
    cx, cy, rx, ry = S1["cx"], S1["cy"], S1["rx"], S1["ry"]
    A["bg1"] = [texture((0.13, 0.16, 0.15), 10 + i, angle=-1.1) for i in range(BOIL)]
    nz = vnoise(7, 8)
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) + (nz - 0.5) * 0.015
    A["eggmask"] = clip((1 - d) * ry / 2 + 0.5)
    shade = (1.08 - 0.32 * clip((Y - (cy - ry)) / (2 * ry))) * (1 - 0.28 * clip(d) ** 2)
    eggs = []
    for i in range(BOIL):
        rng = np.random.default_rng(30 + i)
        im = Image.new("L", (W, H), 128)
        dr = ImageDraw.Draw(im)
        for r in np.linspace(0.02, 1.0, 110):   # concentric hatching, as in the film
            jx, jy = rng.normal(0, 2, 2)
            dr.ellipse([cx - r * rx + jx, cy - r * ry + jy, cx + r * rx + jx, cy + r * ry + jy],
                       outline=int(rng.integers(70, 200)), width=2)
        ring = (np.asarray(im, np.float32) - 128) / 128
        t = texture((0.50, 0.58, 0.53), 40 + i, n=6000, ss=0.08, st=0.15)
        eggs.append(clip(t * (1 + 0.22 * ring[..., None]) * shade[..., None]))
    A["egg"] = eggs
    r2 = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / (ry * 0.62)) ** 2)
    k = clip(r2) ** 0.8
    c0, c1 = np.array([0.55, 0.06, 0.10]), np.array([0.07, 0.01, 0.03])
    eye = c0[None, None] * (1 - k[..., None]) + c1[None, None] * k[..., None]
    rings = 0.5 + 0.5 * np.sin(r2 * 70 + vnoise(8, 6) * 6)
    A["eye1"] = clip(eye * (0.85 + 0.25 * rings[..., None]) * (0.9 + 0.2 * stipple(9, 0.08)[..., None]))


def scene1(t, v):
    cx, cy, rx, ry = S1["cx"], S1["cy"], S1["rx"], S1["ry"]
    em = A["eggmask"]
    s = ease((t - 2.4) / 0.8)
    amp = 0.62 * ease((t - 3.2) / 2.0)
    dil = ease((t - 4.4) / 1.6)
    u = (X - cx) / rx
    base = clip(1 - u * u)
    h = amp * base * ry
    opening = clip((h - np.abs(Y - cy)) / 1.5 + 0.5) * em * (h > 0.5)
    dist = np.abs(Y - cy) - h
    lid = np.exp(-clip(dist, 0, None) / 18) * em * min(1.0, amp * 8)
    seam = s * np.exp(-((Y - cy) / 2.5) ** 2) * np.sqrt(base) * em
    egg = A["egg"][v] * (1 - 0.6 * lid[..., None]) * (1 - 0.85 * seam[..., None])
    f = over(A["bg1"][v], egg, em)
    # pupil: thin slit that dilates into a glowing magenta core
    px, py = 3 + 75 * dil, 150 - 60 * dil
    rr = ((X - cx) / px) ** 2 + ((Y - cy) / py) ** 2
    pm = clip((1 - rr) * 6)
    glow = np.exp(-rr * 2.5) * dil
    pupil = np.stack([0.02 + 0.6 * glow, 0.0 + 0.18 * glow, 0.01 + 0.42 * glow], -1)
    eye = over(A["eye1"], pupil, pm)
    return over(f, eye, opening)


# ---------- scene 2: stone profile, hair turns to fire, floating cobalt eye ----------
PROFILE = [(470, 205), (560, 180), (640, 220), (668, 290), (676, 335), (672, 352),
           (722, 420), (684, 440), (690, 458), (676, 474), (688, 490), (672, 530),
           (630, 562), (618, 600), (624, 960), (512, 960), (528, 600), (474, 470),
           (452, 330)]


def build_s2():
    A["bg2"] = [texture((0.13, 0.12, 0.115), 50 + i, angle=-0.9) for i in range(BOIL)]
    A["facemask"] = poly_mask(PROFILE, 1.5)
    shade = 0.55 + 0.5 * clip((X - 470) / 230)
    shade = shade * (1 - 0.28 * clip((Y - 520) / 120)) * (1 - 0.3 * clip((Y - 600) / 360))
    ink = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(ink)
    d.arc([598, 328, 662, 366], 15, 165, fill=200, width=4)          # closed eye
    d.arc([594, 296, 670, 346], 200, 300, fill=90, width=3)          # brow
    d.line([(662, 475), (684, 476)], fill=170, width=3)               # lips
    d.arc([672, 424, 696, 444], 180, 300, fill=120, width=3)          # nostril
    d.line(PROFILE[2:14], fill=110, width=3)                         # contour
    inkf = from_pil(ink.filter(ImageFilter.GaussianBlur(1.0)))
    A["face2"] = [clip(texture((0.73, 0.73, 0.71), 60 + i, n=12000, ss=0.18, st=0.6, grain=0.1)
                       * shade[..., None] * (1 - 0.6 * inkf[..., None])) for i in range(BOIL)]
    rng = np.random.default_rng(70)
    hair = []
    for _ in range(220):
        th = math.radians(rng.uniform(-215, -55))
        r = rng.uniform(0.85, 0.98)
        x0 = 560 + 108 * r * math.cos(th)
        y0 = 330 + 150 * r * math.sin(th)
        hair.append(dict(x0=x0, y0=y0, L=rng.uniform(330, 620), amp=rng.uniform(18, 45),
                         fr=rng.uniform(1.2, 2.4), ph=rng.uniform(0, 6.28), sp=rng.uniform(1.2, 2.2),
                         g=rng.uniform(0.7, 1.15), fire=[(0.96, 0.40, 0.16), (0.82, 0.12, 0.07),
                                                         (1.0, 0.58, 0.22)][rng.integers(0, 3)],
                         rnd=rng.uniform(0, 0.25)))
    A["hair"] = hair
    A["eyetex"] = 0.9 + 0.2 * stipple(80, 0.1)
    ang = np.arctan2(Y - 360, X - 1000)
    A["irisstreak"] = 0.8 + 0.4 * (0.5 + 0.5 * np.sin(ang * 23 + vnoise(81, 4) * 8))


def scene2(t, v):
    f = over(A["bg2"][v], A["face2"][v], A["facemask"])
    ig = ease((t - 2.0) / 3.5)
    im = to_pil(f).convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for s in A["hair"]:
        n = 32
        pts = []
        for k in range(n + 1):
            q = k / n
            x = s["x0"] - s["L"] * q
            y = (s["y0"] + (s["y0"] - 330) * 0.55 * q
                 + s["amp"] * q ** 1.3 * math.sin(2 * math.pi * q * s["fr"] + s["ph"] - t * s["sp"])
                 + 9 * q * math.sin(2 * math.pi * q * s["fr"] * 4 + s["ph"] * 2 - t * s["sp"] * 2.5))
            pts.append((x, y))
        grey = np.array([0.60, 0.61, 0.64]) * s["g"]
        fire = np.array(s["fire"])
        for k in range(n):
            q = k / n
            fz = min(max(ig * 1.9 - (1 - q) * 0.95 + s["rnd"], 0.0), 1.0)
            c = grey * (1 - fz) + fire * fz
            w = 6 if q < 0.3 else (4 if q < 0.7 else 3)
            d.line([pts[k], pts[k + 1]], fill=tuple(int(255 * min(1, x)) for x in c) + (int(255 * (1 - 0.5 * q)),), width=w)
    f = from_pil(Image.alpha_composite(im, lay).convert("RGB"))
    # floating eye
    eo = ease((t - 1.0) / 2.0)
    blink = 1 - math.sin(math.pi * min(max((t - 6.0) / 0.45, 0), 1))
    ex, ey, ew, eh = 1000, 360, 110, 46 * max(blink, 0.04)
    rim = almond(ex, ey, ew + 8, eh + 9)
    inner = almond(ex, ey, ew, eh)
    f = over(f, np.full_like(f, 0.06), rim * eo)
    white = np.stack([0.80, 0.83, 0.86]) * A["eyetex"][..., None]
    rr = np.sqrt((X - ex) ** 2 + (Y - ey) ** 2)
    k = clip(rr / 40)[..., None]
    iris = (np.array([0.35, 0.6, 1.0]) * (1 - k) + np.array([0.06, 0.16, 0.55]) * k) * A["irisstreak"][..., None]
    eye = over(white, iris, clip((40 - rr) / 1.5))
    eye = over(eye, np.full_like(eye, 0.02), clip((14 - rr) / 1.5))
    eye = over(eye, np.full_like(eye, 0.95), clip((5 - np.sqrt((X - ex + 12) ** 2 + (Y - ey + 12) ** 2)) / 1.5))
    return over(f, eye, inner * eo)


# ---------- scene 3: night sea, a stone face rises and opens glowing eyes ----------
HOR = 470
PW, PH = 320, 460


def build_s3():
    grad = clip((Y - 0) / H)
    bgs = []
    for i in range(BOIL):
        sky = texture((0.05, 0.07, 0.15), 90 + i, angle=-0.2, jit=0.15, length=24)
        sea = texture((0.06, 0.08, 0.17), 95 + i, angle=0.0, jit=0.05, length=30, ss=0.22)
        b = np.where((Y < HOR)[..., None], sky * (1 + 0.3 * grad[..., None]), sea)
        moon = np.sqrt((X - 640) ** 2 + (Y - 140) ** 2)
        b = b + np.array([0.12, 0.14, 0.2]) * np.exp(-moon / 60)[..., None]
        b = over(b, np.array([0.72, 0.75, 0.84]) * (0.9 + 0.15 * stipple(97 + i, 0.2)[..., None]),
                 clip((20 - moon) / 1.5))
        bgs.append(clip(b))
    A["bg3"] = bgs
    py, px = np.mgrid[0:PH, 0:PW].astype(np.float32)
    cx, cy, rx, ry = PW / 2, PH / 2, 120, 175
    v = (py - cy) / ry
    rxx = rx * (1 - 0.28 * clip(v))
    d = np.sqrt(((px - cx) / rxx) ** 2 + v ** 2)
    A["f3mask"] = clip((1 - d) * 80 + 0.5)
    shade = 1.0 - 0.45 * clip(d) ** 2
    ink = Image.new("L", (PW, PH), 0)
    dr = ImageDraw.Draw(ink)
    dr.line([(cx, cy - 10), (cx - 4, cy + 55)], fill=90, width=3)       # nose ridge
    dr.arc([cx - 22, cy + 45, cx + 22, cy + 70], 20, 160, fill=130, width=3)
    dr.line([(cx - 30, cy + 98), (cx + 30, cy + 98)], fill=170, width=4)  # lips
    dr.arc([cx - 28, cy + 92, cx + 28, cy + 112], 20, 160, fill=90, width=3)
    inkf = from_pil(ink.filter(ImageFilter.GaussianBlur(1.0)))
    A["f3"] = [clip(texture((0.42, 0.45, 0.56), 100 + i, n=1500, ss=0.1, st=0.4, grain=0.09, w=PW, h=PH)
                    * shade[..., None] * (1 - 0.55 * inkf[..., None])) for i in range(BOIL)]
    A["f3xy"] = (px, py)


def scene3(t, v):
    f = A["bg3"][v].copy()
    rise = ease(t / 5.0)
    fcy = int(790 - 180 * rise)
    eopen = ease((t - 3.6) / 1.4)
    px, py = A["f3xy"]
    face = A["f3"][v].copy()
    cx, cy = PW / 2, PH / 2
    for sx in (-1, 1):
        ex, ey = cx + sx * 50, cy - 20
        lidline = almond(ex, ey, 34, 2.0, px, py)
        eye = almond(ex, ey, 34, 13 * eopen + 0.01, px, py)
        rr = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        col = np.array([0.3, 0.45, 1.0]) * (0.8 + 0.5 * np.exp(-rr / 10))[..., None]
        face = over(face, np.full_like(face, 0.05), lidline * (1 - eopen))
        face = over(face, col, eye)
        face = face + np.array([0.05, 0.1, 0.35]) * (np.exp(-rr / 22) * eopen)[..., None]
    # paste face patch
    y0, x0 = fcy - PH // 2, 640 - PW // 2
    y1 = min(y0 + PH, H)
    wl = 772 + 5 * np.sin(X[0] * 0.012 + t * 1.3)          # waterline per column
    region = f[y0:y1, x0:x0 + PW]
    m = A["f3mask"][: y1 - y0] * (1 - 0.25 * (1 - rise))
    rows = Y[y0:y1, x0:x0 + PW]
    m = m * clip((wl[None, x0:x0 + PW] - rows) / 3)
    f[y0:y1, x0:x0 + PW] = over(region, clip(face[: y1 - y0]), m)
    facealpha = np.zeros((H, W), np.float32)
    facealpha[y0:y1, x0:x0 + PW] = m
    # wave hatching
    wim = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(wim)
    rng = np.random.default_rng(200)
    y = HOR + 4.0
    xs = np.arange(-20, W + 40, 20, dtype=np.float32)
    while y < H + 10:
        depth = (y - HOR) / (H - HOR)
        amp = 2 + 14 * depth
        ph = rng.uniform(0, 6.28)
        fr = rng.uniform(0.006, 0.012)
        ys = y + amp * np.sin(xs * fr + ph + t * (0.8 + depth))
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=int(rng.integers(80, 230)), width=3)
        y += 3 + 10 * depth
    wa = from_pil(wim) * 0.7 * (1 - facealpha)
    return over(f, np.array([0.46, 0.53, 0.68]) * np.ones_like(f), wa)


# ---------- timeline + finishing ----------
SCENES = [(scene1, 0, 156), (scene2, 132, 348), (scene3, 324, TOTAL)]
XF = 24


def render(i):
    v = (i // 3) % BOIL
    out, wsum = None, 0.0
    for fn, a, b in SCENES:
        if a <= i < b:
            w = 1.0
            if i < a + XF and a > 0:
                w = (i - a) / XF
            if i >= b - XF and b < TOTAL:
                w = min(w, (b - i) / XF)
            img = fn((i - a) / FPS, v)
            out = img * w if out is None else out + img * w
            wsum += w
    f = out / max(wsum, 1e-6)
    yi, xi = A["warp"][v]
    f = f[yi, xi]
    lum = f.mean(-1, keepdims=True)
    f = f * (0.8 + 0.4 * A["tooth"][v][..., None]) + 0.05 * A["speck"][v][..., None] * (1 - lum)
    f *= min(1.0, i / 24) * min(1.0, (TOTAL - 1 - i) / 30)
    rng = np.random.default_rng(1000 + i)
    f *= 1 + 0.015 * math.sin(i * 0.7) + rng.normal(0, 0.006)
    im = to_pil(f).resize((OW, OH), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    a += rng.normal(0, 5, (OH, OW, 1))
    a *= A["vig"][..., None]
    if rng.random() < 0.25:
        a = np.roll(a, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), (0, 1))
    for _ in range(rng.poisson(0.6)):                      # occasional dust specks
        yy, xx = rng.integers(0, OH), rng.integers(0, OW)
        a[max(0, yy - 1):yy + 1, max(0, xx - 1):xx + 1] = 200
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(
        os.path.join(OUTDIR, f"f_{i:04d}.png"), compress_level=1)
    return i


if __name__ == "__main__":
    OUTDIR = sys.argv[1]
    os.makedirs(OUTDIR, exist_ok=True)
    oy, ox = np.mgrid[0:OH, 0:OW].astype(np.float32)
    A["vig"] = 1 - 0.35 * clip(((ox - OW / 2) / (OW / 2)) ** 2 + ((oy - OH / 2) / (OH / 2)) ** 2) ** 1.5
    print("building assets...", flush=True)
    build_s1(); build_s2(); build_s3()
    A["warp"], A["tooth"], A["speck"] = [], [], []
    for i in range(BOIL):
        dx = (vnoise(300 + i, 12, 2) - 0.5) * 9
        dy = (vnoise(310 + i, 12, 2) - 0.5) * 9
        A["warp"].append((np.clip(Y + dy, 0, H - 1).astype(np.int32), np.clip(X + dx, 0, W - 1).astype(np.int32)))
        g = np.random.default_rng(320 + i).random((H, W)).astype(np.float32)
        A["tooth"].append(from_pil(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9))) * 2 - 0.5)
        A["speck"].append(stipple(330 + i, 0.03))
    only = [int(x) for x in sys.argv[2:]]
    frames = only or list(range(TOTAL))
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as p:
        for n, _ in enumerate(p.imap_unordered(render, frames, chunksize=4)):
            if n % 60 == 0:
                print(n, flush=True)
    print("done")
