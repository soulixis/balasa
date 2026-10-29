"""SOFT GEOLOGY — silent stop-motion film: surreal relief surfaces, chalk engraving, line bundles.

Usage: python3 soft.py <frames_dir> [frame numbers...]
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import cutouts as cu
import engine
import relief as rl
from birds import step, texture
from engine import A
from lib import FPS, H, W, X, Y, clip, col, ease, from_pil, lerp, ramp, stipple, to_pil, vgrad, vnoise

RNG = np.random.default_rng
RAW = os.path.join(os.path.dirname(cu.ASSETS), "raw")


def ell(d, cx, cy, rx, ry, fill):
    d.ellipse([cx - abs(rx), cy - abs(ry), cx + abs(rx), cy + abs(ry)], fill=fill)


def frame(f, w=1.0, cam=(640, 480, 1.0), post=None):
    return [(f, cam, w, post)]


# =====================================================================================
# V1 — pink matter: a woman rises out of a wall of soft pebbles
# =====================================================================================
PINK = ((0.50, 0.20, 0.26), (0.86, 0.50, 0.52), (0.99, 0.84, 0.82))
PALE = ((0.62, 0.48, 0.50), (0.93, 0.84, 0.82), (1.0, 0.97, 0.95))


def rounded(m):
    return (rl.blur(m, 6) + rl.blur(m, 16) + rl.blur(m, 40)) / 3


def build_v1():
    hf = rl.pebbles(1, 1500, 14, 50) + 4 * vnoise(2, 40, 2)
    A["v1_peb"] = hf
    A["v1_phase"] = vnoise(3, 3, 2) * 6.28
    cx, top = 560, 150
    head = rl.mask(lambda d: ell(d, cx, top + 95, 58, 80, 255))
    hair = rl.mask(lambda d: d.polygon([(cx - 100, top + 60), (cx - 60, top - 10), (cx, top - 24), (cx + 60, top - 10),
                                        (cx + 100, top + 60), (cx + 112, top + 330), (cx - 112, top + 330)], fill=255))
    neck = rl.mask(lambda d: d.rectangle([cx - 22, top + 168, cx + 22, top + 236], fill=255))
    torso = rl.mask(lambda d: d.polygon([(cx - 26, top + 215), (cx + 26, top + 215), (cx + 105, top + 262), (cx + 92, top + 380),
                                         (cx + 62, top + 520), (cx + 100, top + 680), (cx + 80, H + 20), (cx - 80, H + 20),
                                         (cx - 100, top + 680), (cx - 62, top + 520), (cx - 92, top + 380), (cx - 105, top + 262)], fill=255))
    arms = rl.mask(lambda d: [d.polygon([(cx + m * 108, top + 262), (cx + m * 142, top + 320), (cx + m * 134, top + 620),
                                         (cx + m * 118, top + 652), (cx + m * 112, top + 330)], fill=255) for m in (-1, 1)])
    ridges = clip(hair - head) * (6 * np.sin(X * 0.45 + 3 * vnoise(4, 12, 2)))
    fig = np.maximum.reduce([rounded(hair) * 62 + ridges, rounded(head) * 112, rounded(neck) * 80, rounded(torso) * 92,
                             rounded(arms) * 84])
    feat = np.zeros((H, W), np.float32)
    for ex in (cx - 24, cx + 24):
        feat -= 10 * rl.mask(lambda d, ex=ex: ell(d, ex, top + 88, 13, 6, 255), r=3)
    feat += 9 * rl.mask(lambda d: d.polygon([(cx - 4, top + 86), (cx + 4, top + 86), (cx + 8, top + 136), (cx - 8, top + 136)], fill=255), r=3)
    feat += 4 * rl.mask(lambda d: ell(d, cx, top + 158, 16, 5, 255), r=2)
    feat -= 4 * rl.mask(lambda d: d.line([(cx - 14, top + 158), (cx + 14, top + 158)], fill=255, width=2), r=1)
    A["v1_fig"] = fig + feat * head
    A["v1_m"] = rl.blur(np.maximum.reduce([hair, head, neck, torso, arms]), 4)
    mc = rl.mask(lambda d: ell(d, 985, 175, 50, 50, 255), r=1.5)
    A["v1_moon_h"] = rl.blur(mc, 6) * 30 + rl.craters(5, 40, 4, 14) * mc * 1.2
    A["v1_moon_m"] = mc


def v1(t):
    ts = step(t)
    a = ease((ts - 0.6) / 4.5)
    breathe = 1 + 0.05 * np.sin(ts * 2.2 + A["v1_phase"])
    peb = A["v1_peb"] * breathe
    m = A["v1_m"]
    hf = np.maximum(peb * (1 - 0.85 * a * m), A["v1_fig"] * a)
    rgb_p = rl.shade(hf, *PINK)
    rgb_f = rl.shade(hf, *PALE)
    w = clip(m * a * 1.4)[..., None]
    f = rgb_p * (1 - w) + rgb_f * w
    moon = rl.shade(A["v1_moon_h"], (0.30, 0.30, 0.48), (0.60, 0.62, 0.78), (0.88, 0.90, 0.98), cav=4)
    mm = A["v1_moon_m"][..., None]
    f = f * (1 - mm) + moon * mm
    return frame(f)


# =====================================================================================
# V2 — the empty dress: teal cliffs, a cloud mass, a dress that fills and lifts away
# =====================================================================================
TEAL = ((0.05, 0.28, 0.30), (0.20, 0.62, 0.58), (0.66, 0.92, 0.84))
CLOUD = ((0.08, 0.18, 0.48), (0.30, 0.48, 0.86), (0.86, 0.93, 1.0))
DRESS = ((0.45, 0.62, 0.60), (0.84, 0.94, 0.90), (1.0, 1.0, 0.98))


def build_v2():
    n = vnoise(10, 3, 3)
    ridge = 330 + 80 * (vnoise(11, 4, 2)[0] - 0.5)
    cliff = (Y > ridge[None, :] + 12 * (n - 0.5)).astype(np.float32)
    b = Y + 70 * (vnoise(12, 5, 3) - 0.5)
    layers = b / 46
    strata = 30 * (np.floor(layers) + clip((layers % 1 - 0.55) / 0.45) ** 2)
    hf = strata + rl.pebbles(13, 900, 6, 22, height=0.5) + 5 * vnoise(14, 30, 2)
    cleft = rl.mask(lambda d: d.polygon([(560, 560), (720, 560), (750, H), (530, H)], fill=255), r=14)
    hf = hf * (1 - 0.8 * cleft)
    cloud_m = (Y < 250 + 45 * (vnoise(15, 6, 2) - 0.5) + 25 * np.sin(X * 0.02)).astype(np.float32)
    cloud_m = rl.blur(cloud_m, 2)
    chf = rl.pebbles(16, 500, 26, 72, height=0.7)
    rc = rl.shade(hf, *TEAL)
    rc *= (1 - 0.45 * cleft)[..., None]
    rk = rl.shade(chf, *CLOUD)
    sky = vgrad((0.05, 0.12, 0.36), (0.12, 0.24, 0.52), 160, 420)
    f = sky * (1 - cliff[..., None]) + rc * cliff[..., None]
    f = f * (1 - cloud_m[..., None]) + rk * cloud_m[..., None]
    A["v2"] = f
    A["v2_cloud_m"] = cloud_m
    A["v2_cloud"] = rk
    w, h = 300, 420
    hem = [(x, 392 + 10 * math.sin(x * 0.08)) for x in np.linspace(262, 38, 30)]
    dm = rl.mask(lambda d: (d.polygon([(112, 64), (188, 64), (184, 150), (170, 186), (130, 186), (116, 150)], fill=255),
                            d.polygon([(130, 182), (170, 182), (200, 260), (262, 392)] + hem + [(38, 392), (100, 260)], fill=255),
                            d.line([(118, 8), (126, 70)], fill=255, width=9), d.line([(182, 8), (174, 70)], fill=255, width=9)), w, h, r=1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pleat = 5 * np.sin(np.arctan2(xx - 150, yy - 182 + 1e-3) * 28) * (yy > 196)
    A["dress_h"] = rounded(dm) * 44 + pleat * dm
    A["dress_m"] = dm


def v2(t):
    ts = step(t)
    f = A["v2"].copy()
    inflate = ease((ts - 0.8) / 1.6)
    lift = ease((ts - 3.6) / 3.0)
    sway = 10 * math.sin(ts * 2.6) * inflate
    dh = A["dress_h"] * (1 + 0.9 * inflate)
    rgb = rl.shade(dh, *DRESS)
    al = A["dress_m"]
    piece = Image.fromarray((np.dstack([rgb, al]) * 255).astype(np.uint8), "RGBA")
    sx = 1 + 0.18 * inflate
    piece = piece.resize((int(piece.width * sx), piece.height), Image.BICUBIC)
    cx, by = 640 + sway, 900 - 600 * lift
    im = to_pil(f).convert("RGBA")
    sh = Image.new("RGBA", piece.size, (0, 0, 0, 0))
    sh.putalpha(piece.getchannel("A").point(lambda v: int(v * 0.35 * (1 - lift))))
    im.alpha_composite(sh, (int(cx - piece.width / 2 + 14), int(by - piece.height + 8)))
    im.alpha_composite(piece, (int(cx - piece.width / 2), int(by - piece.height)))
    f2 = from_pil(im.convert("RGB"))
    swallow = ramp(ts, 5.2, 6.6)
    cm = (A["v2_cloud_m"] * (0.6 + 0.4 * swallow))[..., None] * (Y < 330)[..., None]
    f2 = f2 * (1 - cm * swallow) + A["v2_cloud"] * cm * swallow
    return frame(f2)


# =====================================================================================
# V3 — chalk face: an old woman drawn in chalk is redrawn, stroke by stroke, into a young mask
# =====================================================================================
EYE = (640, 420)
SC = 1.25


def place_src(im, eye_xy, scale):
    """Paste a source image into a frame-sized canvas so its eye midpoint lands on EYE."""
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    c = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    c.alpha_composite(im, (int(EYE[0] - eye_xy[0] * scale), int(EYE[1] - eye_xy[1] * scale))) if False else None
    x, y = int(EYE[0] - eye_xy[0] * scale), int(EYE[1] - eye_xy[1] * scale)
    l, t = max(0, -x), max(0, -y)
    c.alpha_composite(im.crop((l, t, min(im.width, W - x), min(im.height, H - y))), (x + l, y + t))
    return np.asarray(c, np.float32) / 255


def chalk_strokes(lum, msk, n, seed, gamma=2.0, color=(0.88, 0.87, 0.84), flow=7):
    r = RNG(seed)
    sm = rl.blur(lum, flow)
    gy, gx = np.gradient(sm)
    xs, ys = r.uniform(0, W, n * 3), r.uniform(0, H, n * 3)
    xi, yi = xs.astype(int).clip(0, W - 1), ys.astype(int).clip(0, H - 1)
    p = clip(lum[yi, xi]) ** gamma * msk[yi, xi]
    keep = r.random(len(xs)) < p
    xs, ys, xi, yi = xs[keep][:n], ys[keep][:n], xi[keep][:n], yi[keep][:n]
    ang = np.arctan2(gy[yi, xi], gx[yi, xi]) + math.pi / 2 + r.normal(0, 0.10, len(xs))
    ln = r.uniform(8, 18, len(xs))
    b = lum[yi, xi]
    out = []
    for x, y, a, l, bb in zip(xs, ys, ang, ln, b):
        dx, dy = math.cos(a) * l / 2, math.sin(a) * l / 2
        out.append(((x - dx, y - dy, x + dx, y + dy), col(color, 0.65 + 0.4 * bb)))
    return out


def build_v3():
    old = Image.open(os.path.join(RAW, "wc_12212145.jpg")).convert("RGBA").crop((30, 30, 760, 770))
    a = place_src(old, (397 - 30, 338 - 30), SC)
    lum = a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    msk = rl.blur(((lum > 0.58) & (a[..., 3] > 0.5)).astype(np.float32), 3)
    old_b = clip((lum - 0.40) / 0.42)
    young = cu.load("102361")
    b = place_src(young, (410, 470), SC)
    yl = b[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    ym = rl.blur((b[..., 3] > 0.5).astype(np.float32), 2)
    sel = ym > 0.5
    p2, p98 = np.percentile(yl[sel], 3), np.percentile(yl[sel], 99)
    young_b = clip((yl - p2) / (p98 - p2)) ** 1.2
    r = RNG(31)
    so = chalk_strokes(old_b, msk, 17000, 32, flow=14)
    sy = chalk_strokes(young_b, ym, 22000, 33, color=(0.90, 0.89, 0.86))
    A["c_wash_o"] = (rl.blur(old_b * msk, 2) * 0.22)[..., None] * np.array([0.9, 0.88, 0.84])
    A["c_wash_y"] = (rl.blur(young_b * ym, 2) * 0.22)[..., None] * np.array([0.9, 0.9, 0.88])
    A["c_old"] = [(xy, c, r.uniform(0, 1), r.uniform(4.0, 6.6)) for xy, c in so]
    A["c_young"] = [(xy, c, r.uniform(4.2, 7.0)) for xy, c in sy]
    al = np.asarray(Image.fromarray((ym * 255).astype(np.uint8)))
    hair = []
    for k in range(150):
        u = k / 149
        ang = math.radians(lerp(195, 250, u * 2) if u < 0.5 else lerp(290, 345, u * 2 - 1))
        x0 = EYE[0] + 270 * math.cos(ang) * 1.05
        y0 = EYE[1] - 40 + 300 * math.sin(ang)
        side = -1 if math.cos(ang) < 0 else 1
        pts = [(x0, y0)]
        L = r.uniform(380, 620)
        ph, fr, amp = r.uniform(0, 6), r.uniform(0.012, 0.03), r.uniform(8, 22)
        for j in range(1, 40):
            q = j / 39
            pts.append((x0 + side * (60 * q + 40 * q * q) + amp * math.sin(ph + j * fr * 30),
                        y0 + L * q))
        hair.append(dict(pts=pts, t=r.uniform(5.8, 7.6), c=[(0.92, 0.34, 0.20), (0.78, 0.18, 0.12), (0.98, 0.52, 0.30)][k % 3],
                         w=int(r.integers(2, 4))))
    A["c_hair"] = hair
    A["c_bg"] = texture((0.075, 0.068, 0.062), 34, angle=-0.9)


def v3(t):
    ts = step(t)
    drawn = ease(ts / 2.4)
    wo, wy = drawn * (1 - ramp(ts, 4.0, 6.6)), ramp(ts, 4.2, 7.0)
    im = to_pil(A["c_bg"] + A["c_wash_o"] * wo + A["c_wash_y"] * wy)
    d = ImageDraw.Draw(im)
    for xy, c, order, off in A["c_old"]:
        if order < drawn and ts < off:
            d.line(xy, fill=c, width=2)
    for xy, c, on in A["c_young"]:
        if ts >= on:
            d.line(xy, fill=c, width=2)
    for h_ in A["c_hair"]:
        fr = ramp(ts, h_["t"], h_["t"] + 1.2)
        n = int(len(h_["pts"]) * fr)
        if n >= 2:
            d.line(h_["pts"][:n], fill=col(h_["c"]), width=h_["w"], joint="curve")
    return frame(from_pil(im))


# =====================================================================================
# V4 — the doorway: white water-lines flow inside a crumbling stone frame; a mask descends into it
# =====================================================================================
OPEN = (480, 380, 800, H)


def build_v4():
    n = vnoise(40, 10, 3)
    outer = ((X > 370 + 30 * n) & (X < 910 - 30 * n) & (Y > 290 + 25 * n)).astype(np.float32)
    inner = ((X > OPEN[0]) & (X < OPEN[2]) & (Y > OPEN[1])).astype(np.float32)
    broken = rl.mask(lambda d: d.polygon([(900, 520), (990, 600), (960, H), (900, H)], fill=255))
    m = clip(outer - inner + broken)
    hf = rounded(m) * 40 + rl.pebbles(41, 900, 4, 14, height=0.5) * m + rl.craters(42, 160, 3, 9) * m
    rgb = rl.shade(hf, (0.10, 0.08, 0.09), (0.40, 0.33, 0.33), (0.80, 0.74, 0.72), cav=5)
    spk = (stipple(43, 0.015) > 0.5)[..., None] * m[..., None]
    rgb = rgb * (1 - spk) + np.array([0.55, 0.16, 0.22]) * spk
    bg = texture((0.10, 0.09, 0.09), 44, angle=-0.8)
    bg = bg * (1 - inner[..., None]) + np.array([0.02, 0.02, 0.03]) * inner[..., None]
    A["d_frame"] = Image.fromarray((np.dstack([rgb, m]) * 255).astype(np.uint8), "RGBA")
    A["d_bg"] = bg
    w, h = 170, 90
    mm = rl.mask(lambda d: ell(d, 85, 50, 80, 38, 255), w, h, r=1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mh = rounded(mm) * 30
    for ex in (55, 115):
        mh -= 9 * rl.mask(lambda d, ex=ex: d.polygon([(ex - 16, 52), (ex + 16, 46), (ex + 14, 52), (ex - 14, 57)], fill=255), w, h, r=1)
    mh += 6 * rl.mask(lambda d: d.line([(85, 20), (85, 62)], fill=255, width=6), w, h, r=2)
    rgb = rl.shade(mh, (0.18, 0.22, 0.30), (0.52, 0.58, 0.66), (0.90, 0.92, 0.96), cav=4)
    A["d_mask"] = Image.fromarray((np.dstack([rgb, mm]) * 255).astype(np.uint8), "RGBA")


def v4(t):
    ts = step(t)
    im = to_pil(A["d_bg"]).convert("RGBA")
    d = ImageDraw.Draw(im)
    my = lerp(190, 600, ease((ts - 1.0) / 3.5))
    inside = ramp(ts, 3.6, 4.6)
    shrink = lerp(1.0, 0.35, ease((ts - 5.4) / 2.0))
    for k in range(9):
        x0 = lerp(OPEN[0] + 30, OPEN[2] - 30, k / 8)
        pts = []
        for y in np.arange(OPEN[1] + 10, H + 20, 14):
            push = 70 * inside * shrink * math.exp(-((y - my) / 100) ** 2) * (1 if x0 > 640 else -1) * (1 - abs(x0 - 640) / 200)
            pts.append((x0 + 16 * math.sin(y * 0.022 + k * 0.9 - ts * 2.2) + push, y))
        d.line(pts, fill=col((0.60, 0.60, 0.62)), width=14, joint="curve")
        d.line(pts, fill=col((0.93, 0.93, 0.92)), width=9, joint="curve")
    mk = A["d_mask"].resize((int(170 * shrink * 1.4), int(90 * shrink * 1.4)), Image.BICUBIC)
    if inside > 0.5:
        im.alpha_composite(mk, (int(640 - mk.width / 2), int(my - mk.height / 2)))
    im.alpha_composite(A["d_frame"])
    if inside <= 0.5:
        im.alpha_composite(mk, (int(640 - mk.width / 2), int(my - mk.height / 2)))
    return frame(from_pil(im.convert("RGB")))


# =====================================================================================
# V5 — the room: curls on one wall, a stone profile on the other; shells open, a pearl becomes the moon
# =====================================================================================
VP = (640, 400)


def build_v5():
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 0), (W, 0), VP], fill=col((0.07, 0.09, 0.18)))
    d.polygon([(0, 0), VP, (0, H)], fill=col((0.05, 0.07, 0.15)))
    d.polygon([(W, 0), VP, (W, H)], fill=col((0.06, 0.08, 0.17)))
    d.polygon([(0, H), VP, (W, H)], fill=col((0.10, 0.12, 0.22)))
    r = RNG(50)
    for _ in range(700):
        a = r.uniform(0, 2 * math.pi)
        r0, r1 = r.uniform(40, 500), r.uniform(10, 60)
        x0, y0 = VP[0] + math.cos(a) * r0, VP[1] + math.sin(a) * r0
        d.line([(x0, y0), (x0 + math.cos(a) * r1, y0 + math.sin(a) * r1)], fill=col((0.20, 0.24, 0.40), r.uniform(0.4, 1.0)), width=1)
    base = from_pil(im) * texture((1, 1, 1), 51, angle=-0.5)
    cm = rl.mask(lambda d: ell(d, 200, 360, 150, 210, 255), r=2)
    chf = rl.pebbles(52, 700, 8, 20, height=0.9) * cm + rounded(cm) * 30
    curls = rl.shade(chf, (0.08, 0.02, 0.05), (0.32, 0.10, 0.18), (0.70, 0.40, 0.48), cav=4)
    base = base * (1 - cm[..., None]) + curls * cm[..., None]
    A["r_bg"] = clip(base)
    nef = cu.scaled(cu.tone(cu.load("135658"), lo=(0.06, 0.08, 0.16), hi=(0.62, 0.68, 0.82), keep=0.05, seed=53), 330)
    A["r_head"] = nef
    w, h = 140, 90
    sm = rl.mask(lambda d: d.pieslice([4, 4, 136, 170], 180, 360, fill=255), w, h, r=1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rib = 4 * np.cos(np.arctan2(xx - 70, 86 - yy) * 18)
    shf = rounded(sm) * 26 + rib * sm
    rgb = rl.shade(shf, (0.25, 0.02, 0.08), (0.62, 0.08, 0.22), (0.96, 0.50, 0.62), cav=3)
    A["shell"] = Image.fromarray((np.dstack([rgb, sm]) * 255).astype(np.uint8), "RGBA")


def v5(t):
    ts = step(t)
    im = to_pil(A["r_bg"]).convert("RGBA")
    nef = A["r_head"]
    im.alpha_composite(nef, (int(1080 - nef.width / 2), int(340 - nef.height / 2)))
    d = ImageDraw.Draw(im)
    ell(d, 640, 52, 70, 26, col((0.62, 0.68, 0.80)))
    d.chord([570, 26, 710, 78], 0, 180, fill=col((0.40, 0.46, 0.60)))
    op = ease((ts - 1.0) / 1.6)
    for cx in (590, 690):
        bottom = A["shell"].transpose(Image.FLIP_TOP_BOTTOM)
        im.alpha_composite(bottom, (cx - 70, 760))
        top = A["shell"].rotate(-40 * op if cx < 640 else 40 * op, resample=Image.BICUBIC, expand=True,
                                center=(70, 88))
        im.alpha_composite(top, (cx - top.width // 2, 762 - top.height + 6))
    rise = ease((ts - 3.0) / 3.4)
    if op > 0.4:
        px, py = 640, lerp(760, 70, rise)
        pr = lerp(18, 54, rise)
        glow = Image.new("L", (W, H), 0)
        ImageDraw.Draw(glow).ellipse([px - pr * 2, py - pr * 2, px + pr * 2, py + pr * 2], fill=90)
        im.alpha_composite(Image.merge("RGBA", (Image.new("L", (W, H), 230), Image.new("L", (W, H), 232), Image.new("L", (W, H), 245),
                                                 glow.filter(ImageFilter.GaussianBlur(20)))))
        d = ImageDraw.Draw(im)
        ell(d, px, py, pr, pr, col((0.94, 0.93, 0.90)))
        ell(d, px - pr * 0.3, py - pr * 0.3, pr * 0.3, pr * 0.3, col((1.0, 1.0, 1.0)))
    return frame(from_pil(im.convert("RGB")))


# =====================================================================================
# V6 — the sleeping cloud: fine lines drift over a cratered seabed and comb into a sleeping profile
# =====================================================================================
PROFILE = [(470, 205), (560, 180), (640, 220), (668, 290), (676, 335), (672, 352), (722, 420), (684, 440), (690, 458),
           (676, 474), (688, 490), (672, 530), (630, 562)]


def resample(pts, n):
    pts = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(u, cum, pts[:, 0]), np.interp(u, cum, pts[:, 1])], 1)


def build_v6():
    sky = vgrad((0.03, 0.05, 0.18), (0.06, 0.10, 0.30), 0, 800) * texture((1, 1, 1), 60, angle=-0.2, length=26)
    hf = rl.craters(61, 260, 8, 34) + rl.pebbles(62, 600, 4, 12, height=0.4) + 6 * vnoise(63, 30, 2)
    sea = rl.shade(hf, (0.02, 0.03, 0.10), (0.10, 0.14, 0.34), (0.40, 0.48, 0.74), cav=6)
    edge = (Y > 790 + 14 * np.sin(X * 0.01) + 10 * (vnoise(64, 8, 2) - 0.5)).astype(np.float32)
    A["n_bg"] = clip(sky * (1 - edge[..., None]) + sea * edge[..., None])
    A["n_hf"] = hf
    from birds import PROFILE as FULL
    head = [(640 + (x - 590) * 1.15, 120 + (y - 180) * 1.15) for x, y in FULL if y <= 600]
    hm = np.asarray(rl.mask(lambda d: d.polygon(head, fill=255))) > 0.5
    ys = np.nonzero(hm.any(1))[0]
    u = np.linspace(0, 1, 140)
    lines = []
    for k in range(34):
        o = k / 33 - 0.5
        cloud = np.stack([170 + 940 * u, 430 + o * 230 * np.sin(np.pi * u) ** 0.8 + 14 * np.sin(6 * u + k)], 1)
        yk = int(lerp(ys[0] + 4, ys[-1] - 4, k / 33))
        xs = np.nonzero(hm[yk])[0]
        face = np.stack([xs.min() + (xs.max() - xs.min()) * u, yk + 2 * np.sin(u * 20 + k)], 1)
        lines.append((cloud.astype(np.float32), face.astype(np.float32), 0.55 + 0.45 * (1 - abs(o) * 2)))
    A["n_lines"] = lines


def v6(t):
    ts = step(t)
    im = to_pil(A["n_bg"])
    d = ImageDraw.Draw(im)
    q = ease((ts - 1.5) / 2.0) - ease((ts - 5.2) / 2.0)
    dx = 60 * ts / 8
    for k, (cl, fc, b) in enumerate(A["n_lines"]):
        wob = np.stack([np.zeros(140), 4 * np.sin(np.linspace(0, 9, 140) + ts * 3 + k)], 1)
        p = cl * (1 - q) + fc * q + wob + np.array([dx, 0])
        d.line([tuple(v) for v in p.tolist()], fill=col((0.78, 0.84, 0.96), b), width=2, joint="curve")
    return frame(from_pil(im))


# =====================================================================================
# V7 — return: the seabed swells and turns pink; the woman sinks back into the pebbles
# =====================================================================================
def v7(t):
    ts = step(t)
    k = ease(ts / 2.5)
    a = lerp(0.9, 0.0, ease((ts - 2.6) / 3.6))
    peb = A["v1_peb"] * (1 + 0.05 * np.sin(ts * 2.2 + A["v1_phase"]))
    m = A["v1_m"]
    hf = np.maximum(peb * (1 - 0.85 * a * m), A["v1_fig"] * a)
    pal = [tuple(lerp(np.array(n_), np.array(p_), k)) for n_, p_ in zip(((0.02, 0.03, 0.10), (0.10, 0.14, 0.34), (0.40, 0.48, 0.74)), PINK)]
    rgb_p = rl.shade(hf, *pal)
    rgb_f = rl.shade(hf, *PALE)
    w = clip(m * a * 1.4)[..., None]
    f = rgb_p * (1 - w) + rgb_f * w
    return frame(f, w=1 - ramp(t, 6.4, 8.0))


SHOTS = [(v1, 0, 8, 0), (v2, 8, 8, 1.0), (v3, 16, 8, 1.0), (v4, 24, 8, 1.0), (v5, 32, 8, 1.0), (v6, 40, 8, 1.0),
         (v7, 48, 8, 1.0)]
TOTAL = 56 * FPS
BUILDERS = [build_v1, build_v2, build_v3, build_v4, build_v5, build_v6]

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[2:]]
    engine.run(SHOTS, TOTAL, sys.argv[1], frames, BUILDERS, "pastel", boil=2)
