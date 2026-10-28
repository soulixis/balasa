"""THE PLACE BETWEEN DREAMS — 55 s surrealist oil-painting film.

Usage: python3 dreams.py <frames_dir> [frame numbers...]
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

import engine
from engine import A, composite, overlay
from lib import FPS, H, W, X, Y, blur_l, clip, col, cola, ease, from_pil, lerp, ramp, to_pil, vnoise

RNG = np.random.default_rng
HOR = 430
SKY = [(-900, (0.40, 0.58, 0.80)), (0, (0.52, 0.68, 0.84)), (250, (0.74, 0.80, 0.88)), (360, (0.93, 0.82, 0.82)),
       (430, (0.97, 0.91, 0.72))]
GREEN = [(0, (0.66, 0.72, 0.52)), (70, (0.52, 0.64, 0.36)), (530, (0.27, 0.43, 0.17)), (1400, (0.20, 0.34, 0.12))]
GOLD = [(0, (0.82, 0.74, 0.44)), (70, (0.76, 0.63, 0.27)), (530, (0.56, 0.45, 0.15)), (1400, (0.46, 0.37, 0.10))]
GREENS = [(0.30, 0.46, 0.18), (0.42, 0.58, 0.26), (0.52, 0.66, 0.32), (0.24, 0.38, 0.14)]
GOLDS = [(0.85, 0.66, 0.20), (0.70, 0.52, 0.14), (0.92, 0.78, 0.34), (0.60, 0.48, 0.16)]
PORCELAIN = (0.93, 0.92, 0.88)
PBLUE = (0.48, 0.58, 0.76)


# =====================================================================================
# world elements
# =====================================================================================
def rows(ys, stops):
    xs = [s[0] for s in stops]
    return np.stack([np.interp(ys, xs, [s[1][c] for s in stops]) for c in range(3)], -1).astype(np.float32)


def base_img(w=W, h=H, ox=0, oy=0, hor=HOR, ground=GREEN):
    """Sky + ground for a canvas whose world origin sits at (ox, oy)."""
    ys = np.arange(h, dtype=np.float32) - oy
    sk = rows(ys - hor + HOR, SKY)
    gr = rows(ys - hor, ground)
    c = np.where((ys < hor)[:, None], sk, gr)
    haze = (np.exp(-np.abs(ys - hor) / 28) * 0.18)[:, None]
    c = c * (1 - haze) + np.array([0.97, 0.92, 0.80]) * haze
    return np.broadcast_to(c[:, None, :], (h, w, 3)).copy()


def ell(d, cx, cy, rx, ry, fill):
    d.ellipse([cx - abs(rx), cy - abs(ry), cx + abs(rx), cy + abs(ry)], fill=fill)


def grass(d, x0, x1, hor, y1, n, seed, palette, ox=0, oy=0, top=None):
    r = RNG(seed)
    for _ in range(n):
        x = r.uniform(x0, x1)
        y = hor + (y1 - hor) * r.random() ** 1.4
        if top is not None and y < top(x):
            continue
        dep = (y - hor) / max(1, (y1 - hor))
        ln = 3 + 24 * dep
        a = -math.pi / 2 + r.normal(0, 0.28)
        c = palette[r.integers(len(palette))]
        d.line([(x + ox, y + oy), (x + ox + math.cos(a) * ln, y + oy + math.sin(a) * ln)],
               fill=col(c, 0.85 + 0.3 * r.random()), width=max(1, int(1 + 3 * dep)))


def specks(d, hor, y1, n, seed, palette, ox=0, oy=0, x0=0, x1=W):
    r = RNG(seed)
    for _ in range(n):
        y = hor + (y1 - hor) * r.random() ** 1.6
        dep = (y - hor) / max(1, (y1 - hor))
        rr = 1 + 6 * dep
        x = r.uniform(x0, x1)
        c = palette[r.integers(len(palette))]
        ell(d, x + ox, y + oy, rr, rr * 0.7, col(c, 0.75 + 0.35 * dep))


def ocean(d, x0, y0, x1, y1, ox=0, oy=0):
    """The ocean that hangs vertically in the sky."""
    n = 40
    for i in range(n):
        q = i / n
        c = (0.30 + 0.15 * q, 0.52 + 0.14 * q, 0.60 + 0.10 * q)
        ya, yb = y0 + (y1 - y0) * q, y0 + (y1 - y0) * (q + 1 / n)
        d.rectangle([x0 + ox, ya + oy, x1 + ox, yb + oy + 1], fill=col(c))
    r = RNG(5)
    for _ in range(70):
        y, x = r.uniform(y0 + 4, y1 - 4), r.uniform(x0 + 6, x1 - 40)
        d.line([(x + ox, y + oy), (x + ox + r.uniform(8, 40), y + oy)], fill=col((0.72, 0.84, 0.88)), width=2)
    d.line([(x0 + ox, y1 + oy), (x1 + ox, y1 + oy)], fill=col((0.88, 0.94, 0.95)), width=3)
    d.line([(x0 + ox, y0 + oy), (x1 + ox, y0 + oy)], fill=col((0.62, 0.76, 0.80)), width=2)


def island(d, x, y, s, flip=False):
    sy = -1 if flip else 1
    P = lambda dx, dy: (x + dx * s, y + sy * dy * s)
    d.polygon([P(-60, 0), P(60, 0), P(32, 38), P(6, 72), P(-26, 44)], fill=col((0.48, 0.40, 0.32)))
    d.polygon([P(-60, 0), P(60, 0), P(40, 14), P(-40, 14)], fill=col((0.38, 0.32, 0.26)))
    ell(d, *P(0, 0), 62 * s, 11 * s, col((0.46, 0.62, 0.30)))
    d.line([P(-10, 0), P(-10, -40)], fill=col((0.35, 0.25, 0.18)), width=max(1, int(5 * s)))
    ell(d, *P(-10, -55), 22 * s, 22 * s, col((0.32, 0.50, 0.24)))
    if flip:  # hanging trees; roots reach up toward the stars
        for k in range(6):
            dx = -28 + k * 11
            d.line([P(dx, 50), P(dx + (k - 2.5) * 9, 110 + 12 * (k % 2))], fill=col((0.42, 0.32, 0.24)), width=max(1, int(2 * s)))


def pflower(d, x, by, h, R, lean=0.0, face=0.0, gold=False, sy=1, seed=0):
    """Flower taller than a house: thin stem, huge porcelain (or golden) head.
    face: -1..1 turns the head sideways; sy=-1 hangs it upside down."""
    hx, hy = x + lean * h, by - sy * h
    cx, cy = x + lean * h * 0.15, by - sy * h * 0.6
    pts = []
    for i in range(24):
        q = i / 23
        pts.append(((1 - q) ** 2 * x + 2 * (1 - q) * q * cx + q * q * hx, (1 - q) ** 2 * by + 2 * (1 - q) * q * cy + q * q * hy))
    stem = (0.40, 0.52, 0.28) if not gold else (0.42, 0.48, 0.18)
    d.line(pts, fill=col(stem), width=max(2, int(R * 0.06)), joint="curve")
    for q, side in ((0.45, -1), (0.62, 1)):
        px, py = pts[int(q * 23)]
        lw = R * 0.55
        d.polygon([(px, py), (px + side * lw * 0.5, py - sy * lw * 0.35), (px + side * lw, py - sy * lw * 0.1),
                   (px + side * lw * 0.5, py + sy * lw * 0.08)], fill=col(stem, 1.1))
    sq = 1 - 0.55 * abs(face)
    shift = face * R * 0.35
    r = RNG(seed)
    n = 9 if not gold else 14
    for k in range(n):
        a = 2 * math.pi * k / n + seed
        ln = R * r.uniform(0.9, 1.05)
        wd = R * (0.42 if not gold else 0.26)

        def pt(u, v):
            px = math.cos(a) * u - math.sin(a) * v
            py = math.sin(a) * u + math.cos(a) * v
            return (hx + px * sq + shift * (1 - u / R * 0.5), hy + py)

        poly = [pt(0, 0), pt(ln * 0.45, wd / 2), pt(ln, 0), pt(ln * 0.45, -wd / 2)]
        if gold:
            c = (0.97, 0.78, 0.26) if k % 2 else (0.90, 0.64, 0.17)
        else:
            c = PORCELAIN if k % 2 else (0.88, 0.89, 0.90)
        d.polygon(poly, fill=col(c))
        if not gold:
            d.line([pt(ln * 0.22, 0), pt(ln * 0.82, 0)], fill=col(PBLUE), width=max(1, int(R * 0.035)))
            d.line([pt(ln * 0.5, -wd * 0.12), pt(ln * 0.7, -wd * 0.05)], fill=col(PBLUE, 1.1), width=max(1, int(R * 0.02)))
    cr = R * 0.22
    ell(d, hx + shift * 1.4, hy, cr * sq, cr, col((0.80, 0.70, 0.46) if not gold else (0.42, 0.26, 0.10)))


CITY = {}


def city_spec(seed):
    if seed not in CITY:
        r = RNG(100 + seed)
        CITY[seed] = [(2 * math.pi * k / 11 + r.normal(0, 0.12), r.uniform(9, 16), r.uniform(22, 70), int(r.integers(0, 4)),
                       [(0.66, 0.24, 0.18), (0.42, 0.48, 0.60), (0.55, 0.40, 0.30)][r.integers(0, 3)]) for k in range(11)]
    return CITY[seed]


def building(d, bx, base, k, w, h, kind, roof, shade, sy=1):
    P = lambda dx, dy: (bx + dx * k, base - sy * dy * k)
    wall = col((0.86, 0.82, 0.76), shade)
    d.polygon([P(-w / 2, 0), P(w / 2, 0), P(w / 2, h), P(-w / 2, h)], fill=wall)
    if kind == 0:
        d.polygon([P(-w / 2 - 2, h), P(w / 2 + 2, h), P(0, h + w * 1.1)], fill=col(roof, shade))
    elif kind == 1:
        d.polygon([P(-w / 4, h), P(w / 4, h), P(0, h + w * 2.6)], fill=col(roof, shade))
    elif kind == 2:
        d.polygon([P(-w / 2 - 1, h), P(w / 2 + 1, h), P(w / 2 + 1, h + 3), P(-w / 2 - 1, h + 3)], fill=col((0.6, 0.55, 0.5), shade))
    else:
        d.chord([*P(-w / 2, h + w / 2), *P(w / 2, h - w / 2)] if sy > 0 else [*P(-w / 2, h - w / 2), *P(w / 2, h + w / 2)],
                180 if sy > 0 else 0, 360 if sy > 0 else 180, fill=col(roof, shade))
    for j in range(int(h // 14)):
        yy = 8 + j * 14
        d.line([P(-w * 0.15, yy), P(w * 0.15, yy)], fill=col((0.3, 0.28, 0.3), shade), width=max(1, int(2 * k)))


def bldg_pos(x, y, s, rot, idx, seed, sy=1):
    a0, w, h, kind, roof = city_spec(seed)[idx]
    a = a0 + rot
    dep = math.sin(a)
    return x + math.cos(a) * 112 * s, y + sy * (-58 + dep * 14) * s, s * (0.85 + 0.15 * dep), dep


def cloud(d, x, y, s, rot=0.0, city=True, drop=None, sy=1, seed=0):
    """A cloud that carries a miniature city on its back (or underneath it when sy = -1)."""
    r = RNG(seed)
    lobes = []
    for _ in range(11):
        dx = r.uniform(-150, 150)
        dy = -(1 - (dx / 175) ** 2) * 50 * r.uniform(0.5, 1.0)
        lobes.append((dx, dy, r.uniform(46, 88)))
    for dx, dy, rr in lobes:
        ell(d, x + dx * s, y + sy * (dy + 14) * s, rr * s, rr * 0.75 * s, col((0.80, 0.79, 0.86)))
    ell(d, x, y + sy * 22 * s, 175 * s, 26 * s, col((0.82, 0.81, 0.87)))
    for dx, dy, rr in lobes:
        ell(d, x + dx * s - 4 * s, y + sy * dy * s, rr * 0.9 * s, rr * 0.7 * s, col((0.98, 0.97, 0.95)))
    for dx, dy, rr in lobes[:6]:
        ell(d, x + (dx - rr * 0.25) * s, y + sy * (dy - rr * 0.25) * s, rr * 0.4 * s, rr * 0.3 * s, col((1.0, 0.94, 0.91)))
    if not city:
        return
    order = sorted(range(len(city_spec(seed))), key=lambda i: bldg_pos(x, y, s, rot, i, seed, sy)[3])
    for i in order:
        if i == drop:
            continue
        bx, base, k, dep = bldg_pos(x, y, s, rot, i, seed, sy)
        _, w, h, kind, roof = city_spec(seed)[i]
        building(d, bx, base, k, w, h, kind, roof, 0.82 + 0.18 * dep, sy)


def house(d, x, by, s, sy=1, seed=3):
    """Stone house with a red roof. sy = -1 hangs it upside down."""
    P = lambda dx, dy: (x + dx * s, by - sy * dy * s)
    poly = lambda pts, c: d.polygon([P(*p) for p in pts], fill=col(c))
    poly([(-92, 0), (92, 0), (92, 112), (-92, 112)], (0.72, 0.68, 0.60))
    r = RNG(seed)
    for row in range(7):
        for k in range(6):
            x0 = -92 + k * 32 + (row % 2) * 16 + r.uniform(-3, 3)
            y0 = row * 16
            if x0 + 28 > 92 or x0 < -92:
                continue
            d.line([P(x0, y0 + 1), P(x0 + 28, y0 + 1)], fill=col((0.55, 0.52, 0.46)), width=max(1, int(2 * s)))
    poly([(-110, 108), (110, 108), (0, 196)], (0.72, 0.22, 0.16))
    for k in range(1, 5):
        yy = 108 + k * 18
        half = 110 * (1 - k * 18 / 88)
        d.line([P(-half, yy), P(half, yy)], fill=col((0.58, 0.16, 0.12)), width=max(1, int(2 * s)))
    poly([(44, 150), (64, 150), (64, 192), (44, 192)], (0.55, 0.45, 0.38))
    poly([(-16, 0), (16, 0), (16, 56), (-16, 56)], (0.36, 0.24, 0.16))
    for wx in (-62, 44):
        poly([(wx, 52), (wx + 26, 52), (wx + 26, 84), (wx, 84)], (0.95, 0.88, 0.62))
        d.line([P(wx + 13, 52), P(wx + 13, 84)], fill=col((0.4, 0.3, 0.2)), width=max(1, int(2 * s)))


# ---------- the boy ----------
SKIN, HAIRB = (0.93, 0.80, 0.68), (0.42, 0.27, 0.15)
CREAM, CREAM2 = (0.94, 0.90, 0.79), (0.85, 0.80, 0.68)
SOCK, SHOE, COLLAR = (0.97, 0.96, 0.93), (0.36, 0.22, 0.12), (0.55, 0.66, 0.82)


def boy_patch(view, s, walk=None, smile=0.0, reach=0.0):
    w, h = int(340 * s) + 8, int(560 * s) + 8
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, fy = w / 2, h - 4
    P = lambda x, y: (cx + x * s, fy + y * s)
    W_ = lambda v: max(1, int(v * s))
    sw = math.sin(walk) if walk is not None else 0.0
    if view == "side":
        for ph, k in ((sw, 0.9), (-sw, 1.0)):
            hip, foot = (0, -172), (ph * 40, -12)
            d.line([P(*hip), P(*foot)], fill=col(SKIN, k), width=W_(24))
            d.line([P(lerp(hip[0], foot[0], 0.55), lerp(hip[1], foot[1], 0.55)), P(*foot)], fill=col(SOCK, k), width=W_(26))
            ell(d, *P(foot[0] + 12, -8), 24 * s, 11 * s, col(SHOE, k))
        d.polygon([P(-34, -236), P(30, -236), P(38, -160), P(-38, -160)], fill=col(CREAM2))
        d.polygon([P(-32, -366), P(28, -366), P(34, -228), P(-36, -228)], fill=col(CREAM))
        d.polygon([P(-36, -366), P(2, -366), P(-30, -318)], fill=col(COLLAR))
        hand = (lerp(-sw * 22, 96, reach), lerp(-238, -305, reach))
        d.line([P(-2, -350), P(*hand)], fill=col(CREAM, 0.95), width=W_(22))
        ell(d, *P(*hand), 12 * s, 12 * s, col(SKIN))
        d.rectangle([*P(-10, -386), *P(10, -360)], fill=col(SKIN))
        ell(d, *P(4, -428), 50 * s, 58 * s, col(SKIN))
        ell(d, *P(52, -420), 9 * s, 8 * s, col(SKIN))
        d.chord([*P(-50, -490), *P(50, -380)], 150, 360, fill=col(HAIRB))
        d.polygon([P(-50, -440), P(-46, -400), P(-20, -470)], fill=col(HAIRB))
        ell(d, *P(-6, -422), 8 * s, 11 * s, col(SKIN, 0.9))
        ell(d, *P(30, -432), 4 * s, 5 * s, col((0.15, 0.1, 0.08)))
    else:
        lift = (max(0, sw) * 18, max(0, -sw) * 18)
        for m, lf in ((-1, lift[0]), (1, lift[1])):
            d.rectangle([*P(m * 22 - 12, -170 - lf), *P(m * 22 + 12, -12 - lf)], fill=col(SKIN))
            d.rectangle([*P(m * 22 - 13, -95 - lf), *P(m * 22 + 13, -12 - lf)], fill=col(SOCK))
            ell(d, *P(m * 22, -8 - lf), 18 * s, 10 * s, col(SHOE))
        d.polygon([P(-46, -236), P(46, -236), P(50, -160), P(-50, -160)], fill=col(CREAM2))
        d.polygon([P(-44, -366), P(44, -366), P(48, -228), P(-48, -228)], fill=col(CREAM))
        for m in (-1, 1):
            d.line([P(m * 44, -356), P(m * 56, -246)], fill=col(CREAM, 0.95), width=W_(20))
            ell(d, *P(m * 57, -238), 11 * s, 11 * s, col(SKIN))
        d.rectangle([*P(-10, -386), *P(10, -360)], fill=col(SKIN))
        if view == "back":
            d.polygon([P(-42, -366), P(42, -366), P(38, -318), P(-38, -318)], fill=col(COLLAR))
            d.line([P(-34, -326), P(34, -326)], fill=col(SOCK), width=W_(4))
            for m in (-1, 1):
                ell(d, *P(m * 52, -424), 8 * s, 11 * s, col(SKIN, 0.9))
            ell(d, *P(0, -430), 54 * s, 62 * s, col(HAIRB))
        else:
            d.polygon([P(-42, -366), P(0, -366), P(-6, -330)], fill=col(COLLAR))
            d.polygon([P(42, -366), P(0, -366), P(6, -330)], fill=col(COLLAR))
            ell(d, *P(0, -426), 52 * s, 60 * s, col(SKIN))
            d.chord([*P(-54, -492), *P(54, -400)], 180, 360, fill=col(HAIRB))
            d.polygon([P(-54, -446), P(-30, -446), P(-48, -420)], fill=col(HAIRB))
            for m in (-1, 1):
                ell(d, *P(m * 19, -428), 5 * s, 6 * s, col((0.15, 0.1, 0.08)))
                ell(d, *P(m * 30, -404), 9 * s, 5 * s, col((0.96, 0.66, 0.60)))
            d.line([P(-13, -398 - 3 * smile), P(0, -398 + 6 * smile), P(13, -398 - 3 * smile)], fill=col((0.55, 0.25, 0.22)),
                   width=W_(4), joint="curve")
    return im, (cx, fy)


def put_boy(lay, view, x, fy, s, walk=None, smile=0.0, reach=0.0, sx=1.0, rot180=False, alpha=1.0):
    im, (ax, ay) = boy_patch(view, s, walk, smile, reach)
    if sx < 0.999:
        nw = max(2, int(im.width * sx))
        ax = ax * nw / im.width
        im = im.resize((nw, im.height), Image.BICUBIC)
    if rot180:
        im = im.rotate(180)
        ax, ay = im.width - ax, im.height - ay
    if alpha < 1:
        a = im.getchannel("A").point(lambda p: int(p * alpha))
        im.putalpha(a)
    lay.alpha_composite(im, (int(x - ax), int(fy - ay)))


def turn_view(t, seq):
    """seq: [(t_switch, view)...]; returns (view, horizontal squash) with a quick squash at each switch."""
    view, sx = seq[0][1], 1.0
    for ts, v in seq[1:]:
        if t >= ts:
            view = v
        k = (t - ts) / 0.35
        if -1 < k < 1:
            sx = min(sx, max(0.08, abs(k)))
    return view, sx


# ---------- the door ----------
DW, DH = 120, 290
PEEL = [(u, v, u + w, v + h) for u, v, w, h in zip(*[RNG(9).uniform(a, b, 26) for a, b in ((0.02, 0.9), (0.02, 0.95), (0.03, 0.12), (0.01, 0.05))])]


def door(lay, x, fy, s, ang=0.0, beyond=None, pane=None, handle=0.0, gap=None, shadow=True):
    """Free-standing wooden door. Hinge on the right; opens toward the viewer.
    beyond: PIL image shown through the opening. pane(lay, box): draws into the glass."""
    d = ImageDraw.Draw(lay)
    x0, x1, y0 = x - DW / 2 * s, x + DW / 2 * s, fy - DH * s
    if shadow:
        d.polygon([(x0, fy), (x1, fy), (x1 + 250 * s, fy - 22 * s), (x0 + 250 * s, fy - 22 * s)], fill=(30, 36, 14, 70))
    fr = 9 * s
    d.rectangle([x0 - fr, y0 - fr, x1 + fr, fy], fill=col((0.86, 0.84, 0.78)))
    d.line([(x0 - fr, y0 - fr), (x0 - fr, fy)], fill=col((0.55, 0.42, 0.30)), width=max(1, int(2 * s)))
    if ang > 0.001:
        bw, bh = max(2, int(x1 - x0)), max(2, int(fy - y0))
        if beyond is not None:
            lay.paste(beyond.resize((bw, bh), Image.BICUBIC).convert("RGBA"), (int(x0), int(y0)))
        else:
            d.rectangle([x0, y0, x1, fy], fill=col((0.86, 0.93, 0.99)))
    ca = math.cos(ang)
    e = 0.16 * math.sin(ang) * DH * s
    xf = x1 - DW * s * ca

    def q(u, v):
        tx, ty = lerp(xf, x1, u), lerp(y0 - e / 2, y0, u)
        bx, by = lerp(xf, x1, u), lerp(fy + e / 2, fy, u)
        return (tx, lerp(ty, by, v))

    if xf < x1 - 1:
        d.polygon([q(0, 0), q(1, 0), q(1, 1), q(0, 1)], fill=col((0.90, 0.89, 0.84)))
        for u0, v0, u1, v1 in PEEL:
            d.polygon([q(u0, v0), q(u1, v0), q(u1, v1), q(u0, v1)], fill=col((0.52, 0.38, 0.25)))
        d.polygon([q(0.14, 0.46), q(0.86, 0.46), q(0.86, 0.92), q(0.14, 0.92)], outline=col((0.66, 0.64, 0.6)))
        pq = [q(0.2, 0.08), q(0.8, 0.08), q(0.8, 0.36), q(0.2, 0.36)]
        d.polygon(pq, fill=col((0.72, 0.83, 0.90)))
        if pane:
            xs, ys = [p[0] for p in pq], [p[1] for p in pq]
            pane(lay, (min(xs), min(ys), max(xs), max(ys)))
        d.line([q(0.3, 0.1), q(0.45, 0.2)], fill=(255, 255, 255, 150), width=max(1, int(3 * s)))
        d.polygon(pq, outline=col((0.55, 0.45, 0.35)))
        hx, hy = q(0.11, 0.56)
        brass = col((0.80, 0.63, 0.30))
        ell(d, hx, hy, 7 * s * max(ca, 0.3), 7 * s, brass)
        d.line([(hx, hy), (hx + 26 * s * max(ca, 0.3) * math.cos(handle), hy + 26 * s * math.sin(handle))], fill=brass,
               width=max(1, int(5 * s)))
        if gap is not None:
            d.rectangle([x0, fy - 5 * s, x1, fy], fill=col((0.97, 0.95, 0.85)))
            if 0 < gap < 1:
                gx = lerp(x1 + 20 * s, x0 - 20 * s, gap)
                ell(d, gx, fy - 2.5 * s, 22 * s, 3.5 * s, col((0.18, 0.16, 0.14)))


def blue_sky_img():
    im = to_pil(np.broadcast_to(np.linspace([0.92, 0.96, 1.0], [0.62, 0.78, 0.95], 120, dtype=np.float32)[:, None, :], (120, 60, 3)).copy())
    return im


# =====================================================================================
# the original view (seen through doors in shots 5 and 6)
# =====================================================================================
HILL_ORIG = [(0, 960), (0, 760), (250, 722), (450, 704), (650, 708), (850, 742), (1280, 820), (1280, 960)]


def build_orig():
    f = base_img()
    im = to_pil(f)
    d = ImageDraw.Draw(im)
    ocean(d, 960, 150, 1240, 360)
    for x, y, s in ((990, 390, 0.35), (1100, 400, 0.28), (1200, 388, 0.3)):
        island(d, x, y, s)
    specks(d, HOR, 760, 900, 21, [(0.93, 0.92, 0.88), (0.85, 0.86, 0.9)])
    grass(d, 0, W, HOR, H, 3000, 22, GREENS)
    pflower(d, 380, 760, 520, 90, lean=-0.04, seed=1)
    pflower(d, 900, 800, 560, 100, lean=0.05, seed=2)
    d.polygon(HILL_ORIG, fill=col((0.36, 0.52, 0.22)))
    xs, ys = zip(*HILL_ORIG[1:-1])
    grass(d, 0, W, 700, H, 2600, 23, GREENS, top=lambda x: np.interp(x, xs, ys))
    house(d, 680, 712, 0.75)
    cloud(d, 520, 200, 1.0, seed=8)
    cloud(d, 250, 320, 0.6, seed=9)
    A["orig"] = im


# =====================================================================================
# SHOT 1 — the place
# =====================================================================================
def build1():
    pad = 320
    f = base_img(W + pad)
    im = to_pil(f)
    d = ImageDraw.Draw(im)
    ocean(d, 860, 140, 1240, 360)
    for x, y, s in ((890, 388, 0.35), (1010, 398, 0.28), (1160, 384, 0.32)):
        island(d, x, y, s)
    specks(d, HOR, H, 2200, 11, [(0.93, 0.92, 0.88), (0.85, 0.86, 0.9), (0.95, 0.85, 0.6)], x1=W + pad)
    grass(d, 0, W + pad, HOR, H, 9000, 12, GREENS)
    r = RNG(13)
    for _ in range(18):  # far porcelain flowers
        pflower(d, r.uniform(0, W + pad), r.uniform(450, 520), r.uniform(70, 140), r.uniform(14, 24), lean=r.normal(0, 0.05),
                seed=int(r.integers(0, 99)))
    A["s1"] = from_pil(im)
    A["s1_fl"] = [(-60, 1010, 860, 165, 1.0, 0.7), (300, 960, 720, 130, 1.0, -0.6), (1180, 1000, 840, 160, 1.0, -0.8),
                  (1500, 980, 800, 150, 1.0, 0.5), (760, 980, 760, 140, 1.0, 0.6),
                  (120, 700, 420, 80, 0.55, -0.5), (520, 660, 360, 66, 0.55, 0.7), (980, 690, 400, 74, 0.55, -0.6),
                  (1320, 670, 380, 70, 0.55, 0.4), (1650, 700, 420, 80, 0.55, -0.4)]
    A["birds"] = [(r.uniform(120, 520), r.uniform(560, 760), r.uniform(640, 900), r.uniform(395, 425), r.uniform(0, 0.8))
                  for _ in range(16)]
    A["wipe"] = [(r.uniform(-1100, 1100), r.uniform(-650, 650), r.uniform(260, 460)) for _ in range(70)]


def shot1(t):
    g = -300 * ease(t / 9.0)
    off = int(-g * 0.15)
    f = A["s1"][:, off:off + W].copy()

    def draw(d, lay):
        if t > 7.25:  # the small house on a distant hill, revealed by the passing cloud
            hx = 600 + g * 0.15
            d.polygon([(hx - 230, 448), (hx - 120, 420), (hx, 410), (hx + 110, 420), (hx + 220, 448)], fill=col((0.48, 0.60, 0.33)))
            house(d, hx, 412, 0.16)
        tx = 640 + 380 * math.sin(t * 0.45 - 1.0)
        fl = A["s1_fl"]
        for depth in (0.55, 1.0):
            for i, (x, by, h, R, fac, f0) in enumerate(fl):
                if fac != depth:
                    continue
                xx = x + g * fac
                lean = clip((tx - xx) / 1800, -0.16, 0.16) + 0.025 * math.sin(t * 1.1 + i)
                face = f0 * (1 - ramp(t, 0.8, 5.0))
                pflower(d, xx, by, h, R, lean=float(lean), face=face, seed=i)
            if depth == 0.55:
                for i, (sx, sy, ex, ey, dl) in enumerate(A["birds"]):
                    p = ease((t - dl) / 5.6)
                    bx, by = lerp(sx, ex, p), lerp(sy, ey, p)
                    sz = lerp(22, 3, p)
                    m = ramp(p, 0.55, 1.0)
                    if m < 1:
                        fl_ = math.sin(t * 9 + i) * 0.8
                        a = int(255 * (1 - m))
                        for side in (-1, 1):
                            d.line([(bx, by), (bx + side * sz * 0.8, by - sz * (0.3 + 0.5 * fl_)), (bx + side * sz * 1.7, by - sz * 0.2 * fl_)],
                                   fill=(250, 250, 246, a), width=max(1, int(sz * 0.3)))
                    if m > 0:
                        rx = sz * 1.8 + m * 9
                        d.ellipse([bx - rx, by - rx * 0.5, bx + rx, by + rx * 0.5], fill=(250, 248, 244, int(230 * m)))

    f = overlay(f, draw)
    if 6.2 < t < 8.4:
        wl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dw = ImageDraw.Draw(wl)
        cx = 640 + (7.25 - t) * 1500
        for dx, dy, rr in A["wipe"]:
            ell(dw, cx + dx, 480 + dy, rr, rr * 0.8, (250, 248, 246, 255))
        wl = wl.resize((W // 4, H // 4)).resize((W, H), Image.BICUBIC)
        f = composite(f, wl)
    return [(f, (640, 480, 1.04), 1.0, None)]


# =====================================================================================
# SHOT 2 — the boy and the clouds
# =====================================================================================
HILL2 = [(0, 960), (0, 700), (150, 652), (320, 624), (520, 618), (700, 642), (900, 722), (1280, 802), (1280, 960)]
DOOR_FAR = (1010, 600, 0.22)


def build2():
    f = base_img()
    im = to_pil(f).convert("RGBA")
    d = ImageDraw.Draw(im)
    ocean(d, 980, 160, 1250, 370)
    for x, y, s in ((1000, 395, 0.32), (1120, 404, 0.26), (1220, 392, 0.3)):
        island(d, x, y, s)
    specks(d, HOR, 820, 1300, 31, [(0.93, 0.92, 0.88), (0.85, 0.86, 0.9)], x0=600)
    grass(d, 0, W, HOR, 820, 4000, 32, GREENS)
    r = RNG(33)
    for _ in range(9):
        pflower(d, r.uniform(640, 1280), r.uniform(470, 640), r.uniform(120, 220), r.uniform(22, 38), lean=r.normal(0, 0.05),
                seed=int(r.integers(0, 99)))
    door(im, *DOOR_FAR)
    d.polygon(HILL2, fill=col((0.38, 0.54, 0.23)))
    xs, ys = zip(*HILL2[1:-1])
    grass(d, 0, W, 610, H, 3500, 34, GREENS, top=lambda x: np.interp(x, xs, ys))
    house(d, 600, 622, 0.85)
    A["s2"] = from_pil(im.convert("RGB"))


def shot2(t):
    f = A["s2"].copy()
    clouds = [(180 + 22 * t, 190, 1.1, 1, t * 0.05), (990 + 15 * t, 115, 0.9, 2, -t * 0.04),
              (650 + 10 * t, 210 + 175 * ramp(t, 1.5, 5.0), 1.25, 3, t * 0.03)]
    sh = Image.new("L", (W, H), 0)
    dsh = ImageDraw.Draw(sh)
    for x, y, s, _, _ in clouds:
        ell(dsh, x + 140, 470 + (y - 110) * 1.1, 230 * s, 34 * s, 255)
    f = f * (1 - 0.22 * blur_l(sh, 22) * (Y > HOR))[..., None]
    drop = 2
    fall = t - 4.6

    def draw(d, lay):
        for i, (x, y, s, seed, rot) in enumerate(clouds):
            cloud(d, x, y, s, rot=rot, seed=seed, drop=drop if (i == 2 and fall > 0) else None)
        if fall > 0:
            x, y, s, seed, rot = clouds[2]
            bx, base, k, dep = bldg_pos(x, y, s, rot, drop, seed)
            _, w, h, kind, roof = city_spec(seed)[drop]
            a = 1 - ramp(t, 5.6, 6.8)
            if a > 0:
                tl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                building(ImageDraw.Draw(tl), bx + 6 * fall, base - (40 * fall + 130 * fall * fall), k * (1 - 0.35 * ramp(t, 4.6, 6.8)), w, h, kind, roof, 1.0)
                tl.putalpha(tl.getchannel("A").point(lambda p: int(p * a)))
                lay.alpha_composite(tl)
        view, sx = turn_view(t, [(0, "back"), (6.5, "side")])
        put_boy(lay, view, 380, 640, 0.36, sx=sx)

    f = overlay(f, draw)
    q = ramp(t, 6.9, 9.0)
    return [(f, (lerp(640, DOOR_FAR[0], q), lerp(480, DOOR_FAR[1] - 30, q), 1 + 1.2 * q), 1.0, None)]


# =====================================================================================
# SHOT 3 — the door in the golden field
# =====================================================================================
HOR3 = 380
DOOR3 = (700, 722, 1.0)
GOLD_FL = [(90, 1000, 860, 160), (250, 930, 650, 120), (1160, 1010, 880, 175), (970, 950, 640, 125), (420, 600, 330, 70),
           (880, 590, 320, 62), (1060, 560, 260, 55), (300, 540, 240, 48), (560, 520, 200, 40), (790, 505, 180, 36)]


def build3():
    f = base_img(hor=HOR3, ground=GOLD)
    im = to_pil(f)
    d = ImageDraw.Draw(im)
    ocean(d, 1010, 150, 1230, 320)
    specks(d, HOR3, H, 5000, 41, GOLDS + [(0.98, 0.86, 0.4)])
    grass(d, 0, W, HOR3, H, 5000, 42, [(0.55, 0.50, 0.18), (0.66, 0.58, 0.22), (0.45, 0.42, 0.14)])
    A["s3"] = from_pil(im)
    A["bluesky"] = blue_sky_img()


def boy3(t):
    p = 1 - (1 - min(1.0, t / 4.6)) ** 2
    return lerp(230, 612, p), lerp(470, 722, p), lerp(0.11, 0.42, p)


def shot3(t):
    f = A["s3"].copy()
    bx, bfy, bs = boy3(t)
    ang = 0.32 * ramp(t, 2.6, 3.4) + (1.35 - 0.32) * ramp(t, 5.6, 6.4)

    def draw(d, lay):
        turn = ramp(t, 0.8, 3.0)
        for i, (x, by, h, R) in enumerate(GOLD_FL):
            if by > DOOR3[1]:
                continue
            face = float(clip((bx - x) / 350, -1, 1)) * turn
            pflower(d, x, by, h, R, lean=0.02 * math.sin(t + i), face=face, gold=True, seed=i)
        door(lay, *DOOR3, ang=ang, beyond=A["bluesky"])
        reach = ramp(t, 4.8, 5.5)
        put_boy(lay, "side", bx, bfy, bs, walk=(t * 7 if t < 4.6 else None), reach=reach)
        for i, (x, by, h, R) in enumerate(GOLD_FL):
            if by <= DOOR3[1]:
                continue
            face = float(clip((bx - x) / 350, -1, 1)) * turn
            pflower(d, x, by, h, R, lean=0.02 * math.sin(t + i), face=face, gold=True, seed=i)

    f = overlay(f, draw)
    a = ramp(t, 6.0, 7.0)
    f = f * (1 - a) + np.array([0.80, 0.89, 0.98]) * a
    q = ramp(t, 5.9, 7.0)
    return [(f, (lerp(640, 690, q), lerp(480, 575, q), 1 + 1.6 * q), 1.0, None)]


# =====================================================================================
# SHOT 4 — the other side (upside down)
# =====================================================================================
C = 1640


def edge4(x):
    return 180 + 22 * math.sin(x * 0.006) + 10 * math.sin(x * 0.017)


def build4():
    stops = [(0, (0.40, 0.55, 0.78)), (500, (0.70, 0.78, 0.88)), (820, (0.94, 0.85, 0.84)), (1140, (0.74, 0.80, 0.90)),
             (1640, (0.42, 0.56, 0.80))]
    sky = np.broadcast_to(rows(np.arange(C, dtype=np.float32), stops)[:, None, :], (C, C, 3)).copy()
    im = to_pil(sky)
    d = ImageDraw.Draw(im)
    r = RNG(51)
    for _ in range(420):
        x, y = r.uniform(0, C), r.uniform(0, C)
        rr = r.uniform(0.8, 2.2)
        ell(d, x, y, rr, rr, col((0.98, 0.97, 0.9)))
    ocean(d, 260, 610, 1380, 668)
    for x, y, s in ((520, 800, 0.6), (840, 900, 0.5), (1160, 780, 0.55), (690, 1120, 0.45), (1020, 1180, 0.5), (380, 1000, 0.4)):
        island(d, x, y, s, flip=True)
    A["s4_sky"] = im
    top = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(top)
    xs = np.arange(0, W + 20, 20)
    d.polygon([(0, 0)] + [(x, edge4(x)) for x in xs] + [(W, 0)], fill=col((0.40, 0.56, 0.26)))
    for i in range(70):
        y = 10 + i * 2.4
        d.line([(0, y), (W, y)], fill=col((0.36 + 0.002 * i, 0.52, 0.24)), width=2)
    r = RNG(52)
    for _ in range(2600):  # grass hanging overhead
        x = r.uniform(0, W)
        y = edge4(x) - r.uniform(0, 30)
        ln = r.uniform(6, 22)
        d.line([(x, y), (x + r.normal(0, 3), y + ln)], fill=col(GREENS[r.integers(4)]), width=2)
    for i, (x, h, R) in enumerate(((110, 330, 62), (330, 260, 50), (990, 300, 56), (1190, 360, 68))):
        pflower(d, x, edge4(x) - 4, h, R, lean=0.03 * (i - 1.5), sy=-1, seed=i + 20)
    house(d, 770, edge4(770) - 6, 0.62, sy=-1)
    A["s4_top"] = top


def shot4(t):
    ang = 360 * ease((t - 0.8) / 6.6)
    sky = A["s4_sky"].rotate(ang, resample=Image.BICUBIC, center=(C / 2, C / 2))
    sky = sky.crop((C // 2 - W // 2, C // 2 - H // 2, C // 2 + W // 2, C // 2 + H // 2)).convert("RGBA")
    d = ImageDraw.Draw(sky)
    for k, x in enumerate((180, 420, 870, 1110)):  # waterfalls rising into the clouds
        wb, wt = 46, 14
        d.polygon([(x - wb / 2, H), (x + wb / 2, H), (x + wt / 2, 590), (x - wt / 2, 590)], fill=(200, 222, 238, 170))
        for j in range(14):
            y = H - ((j * 30 + t * 140 + k * 13) % 380)
            ww = lerp(wt, wb, (y - 590) / (H - 590)) * 0.35
            d.line([(x - ww, y), (x - ww + 2, y - 14)], fill=(250, 252, 255, 200), width=2)
            d.line([(x + ww * 0.6, y - 7), (x + ww * 0.6 + 1, y - 18)], fill=(250, 252, 255, 160), width=2)
    for x, y, s, seed, sp in ((300, 590, 0.8, 4, 12), (1000, 620, 0.7, 5, -10), (650, 560, 0.55, 6, 6)):
        cloud(d, x + sp * t, y, s, rot=t * 0.06, sy=-1, seed=seed)
    sky.alpha_composite(A["s4_top"])
    a = ramp(t, 5.6, 6.8)
    if a > 0:  # another small figure, far away in the inverted meadow
        put_boy(sky, "front", 640, edge4(640) - 2, 0.12, rot180=True, alpha=a)
    view, sx = turn_view(t, [(0, "front"), (2.6, "side"), (4.8, "back")])
    put_boy(sky, view, 640, 880, 0.55, sx=sx)
    f = from_pil(sky.convert("RGB"))
    b = 1 - ramp(t, 0.0, 1.4)
    f = f * (1 - b) + np.array([0.80, 0.89, 0.98]) * b
    q = ramp(t, 7.0, 10.0)
    return [(f, (640, lerp(480, 300, q), 1 + 0.9 * q), 1.0, None)]


# =====================================================================================
# SHOT 5 — the other boy (his side, which turns upright)
# =====================================================================================
OX, OY = (C - W) // 2, (C - H) // 2
HOR5 = 520
DOOR5 = (760, 830, 1.0)
FL5 = [(120, 980, 700, 130), (330, 900, 520, 95), (1030, 960, 640, 120), (1210, 1000, 760, 150), (520, 640, 300, 55),
       (920, 620, 280, 52)]


def build5():
    f = base_img(C, C, OX, OY, hor=HOR5)
    im = to_pil(f)
    d = ImageDraw.Draw(im)
    ocean(d, 1000, 210, 1240, 430, OX, OY)
    specks(d, HOR5, H + 400, 3000, 61, [(0.93, 0.92, 0.88), (0.85, 0.86, 0.9)], OX, OY, -OX, W + OX)
    grass(d, -OX, W + OX, HOR5, H + 400, 7000, 62, GREENS, OX, OY)
    A["s5"] = im
    A["s5_noise"] = vnoise(63, 10, 3)


def other_boy(t):
    p = ramp(t, 2.4, 5.6)
    return lerp(380, 672, p)


def shot5(t):
    canvas = A["s5"].copy().convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for x, y, s, seed, rot, sp in ((900, 200, 0.9, 6, t * 0.5, -25), (300, 290, 0.75, 7, -t * 0.4, -20)):
        cloud(d, x + sp * t, y, s, rot=rot, seed=seed)
    for i, (x, by, h, R) in enumerate(FL5):
        if by < DOOR5[1]:
            pflower(d, x, by, h, R, lean=0.025 * math.sin(t * 0.9 + i), seed=i + 30)
    m = ramp(t, 1.2, 3.8)
    if m > 0:  # the door emerges as if from an old painting
        dl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ang = 1.2 * ramp(t, 6.4, 7.4)
        door(dl, *DOOR5, ang=ang, beyond=A["orig"].crop((330, 0, 730, 960)))
        mask = clip((m * 1.2 - A["s5_noise"]) / 0.08)
        a = np.asarray(dl.getchannel("A"), np.float32) / 255 * mask
        dl.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
        lay.alpha_composite(dl)
    bx = other_boy(t)
    walking = 2.4 < t < 5.6
    put_boy(lay, "side", bx, 832, 0.42, walk=(t * 7 if walking else None), reach=ramp(t, 5.8, 6.4))
    put_boy(lay, "side", 640 - (bx - 380) * 0.5, 70, 0.12, walk=(t * 7 if walking else None), rot180=True)
    for i, (x, by, h, R) in enumerate(FL5):
        if by >= DOOR5[1]:
            pflower(d, x, by, h, R, lean=0.025 * math.sin(t * 0.9 + i), seed=i + 30)
    canvas.alpha_composite(lay, (OX, OY))
    ang = 180 * (1 - ease(t / 2.6))
    if ang > 0.01:
        canvas = canvas.rotate(ang, resample=Image.BICUBIC, center=(C / 2, C / 2))
    f = from_pil(canvas.crop((OX, OY, OX + W, OY + H)).convert("RGB"))
    q = ramp(t, 7.4, 10.0)
    return [(f, (lerp(640, DOOR5[0], q), lerp(480, DOOR5[1] - 145, q), 1 + 1.4 * q), 1.0, None)]


# =====================================================================================
# SHOT 6 — the circle
# =====================================================================================
DOOR6 = (640, 600, 0.5)
FL6 = [(520, 640, 150, 26), (780, 650, 160, 28), (430, 700, 200, 36), (860, 720, 210, 38), (300, 800, 300, 55),
       (1000, 820, 300, 56), (150, 950, 420, 80), (1150, 960, 430, 82)]


def build6():
    for key, ground, pal, gold in (("s6_gold", GOLD, GOLDS, True), ("s6_green", GREEN, GREENS, False)):
        im = to_pil(base_img(ground=ground))
        d = ImageDraw.Draw(im)
        ocean(d, 990, 160, 1240, 360)
        specks(d, HOR, H, 3500, 71, pal if gold else [(0.93, 0.92, 0.88), (0.85, 0.86, 0.9)])
        grass(d, 0, W, HOR, H, 5000, 72, pal)
        for i, (x, by, h, R) in enumerate(FL6):
            pflower(d, x, by, h, R, gold=gold, seed=i + 40)
        A[key] = from_pil(im)


def reflection(alpha):
    def pane(lay, box):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        dd = ImageDraw.Draw(lay)
        cx, cy, r = (x0 + x1) / 2, y0 + h * 0.58, min(w, h) * 0.34
        a = int(200 * alpha)
        dd.ellipse([cx - r, cy - r * 1.15, cx + r, cy + r * 1.15], fill=cola(SKIN, alpha * 0.8))
        dd.chord([cx - r * 1.05, cy - r * 1.3, cx + r * 1.05, cy + r * 0.2], 180, 360, fill=cola(HAIRB, alpha * 0.8))
        for m in (-1, 1):
            dd.ellipse([cx + m * r * 0.38 - r * 0.09, cy - r * 0.1, cx + m * r * 0.38 + r * 0.09, cy + r * 0.08], fill=(40, 25, 20, a))
        dd.line([(cx - r * 0.3, cy + r * 0.42), (cx, cy + r * 0.6), (cx + r * 0.3, cy + r * 0.42)], fill=(140, 60, 55, a),
                width=max(1, int(r * 0.08)), joint="curve")
    return pane


def shot6(t):
    k = ramp(t, 6.0, 9.4)
    f = A["s6_gold"] * (1 - k) + A["s6_green"] * k
    smile = ramp(t, 1.6, 3.0)
    z = lerp(5.6, 1.0, ease((t - 4.0) / 5.0)) if t > 4.0 else 5.6

    def post(d, m, z, lay):
        view = A["orig"].crop((330, 0, 730, 960)).convert("RGBA")
        put_boy(view, "front", 190, 712, 0.5, smile=smile)
        sx, sfy = m(DOOR6[0], DOOR6[1])
        door(lay, sx, sfy, DOOR6[2] * z, ang=1.0, beyond=view, pane=reflection(smile))

    return [(f, (DOOR6[0], DOOR6[1] - 72, z), 1.0, post)]


# =====================================================================================
# SHOT 7 — the place between dreams
# =====================================================================================
DOOR7 = (640, 650, 0.55)
FL7 = [(80, 1000, 820, 150), (1190, 990, 800, 150), (430, 620, 330, 60), (900, 600, 300, 55), (1020, 720, 430, 80)]
HILL7 = [(0, 960), (0, 560), (120, 520), (260, 505), (400, 520), (560, 580), (720, 700), (720, 960)]


def build7():
    im = to_pil(base_img())
    d = ImageDraw.Draw(im)
    ocean(d, 900, 140, 1240, 380)
    for x, y, s in ((930, 405, 0.34), (1060, 412, 0.27), (1180, 400, 0.3)):
        island(d, x, y, s)
    specks(d, HOR, H, 3000, 81, [(0.93, 0.92, 0.88), (0.85, 0.86, 0.9)])
    grass(d, 0, W, HOR, H, 6000, 82, GREENS)
    d.polygon(HILL7, fill=col((0.40, 0.55, 0.25)))
    xs, ys = zip(*HILL7[1:-1])
    grass(d, 0, 720, 500, H, 2500, 83, GREENS, top=lambda x: np.interp(x, xs, ys))
    house(d, 250, 508, 0.55)
    A["s7"] = from_pil(im)


def shot7(t):
    def draw(d, lay):
        for i, (x, by, h, R) in enumerate(FL7):
            pflower(d, x, by, h, R, lean=0.03 * math.sin(t * 0.8 + i * 1.3), seed=i + 50)

    f = overlay(A["s7"].copy(), draw)
    gap = (t - 1.2) / 1.5 if 1.2 < t < 2.7 else -1
    handle = math.radians(28) * ramp(t, 2.9, 3.4)

    def post(d, m, z, lay):
        sx, sfy = m(DOOR7[0], DOOR7[1])
        door(lay, sx, sfy, DOOR7[2] * z, gap=gap, handle=handle)

    w = 1 - ramp(t, 3.3, 4.95)
    return [(f, (640, lerp(480, 560, ease(t / 5)), 1 + 1.3 * ease(t / 5)), w, post)]


# =====================================================================================
SHOTS = [(shot1, 0, 8, 0), (shot2, 8, 8, 1.0), (shot3, 16, 7, 1.0), (shot4, 23, 9, 0), (shot5, 32, 9, 1.0),
         (shot6, 41, 9, 1.0), (shot7, 50, 5, 1.0)]
TOTAL = 55 * FPS

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[2:]]
    engine.run(SHOTS, TOTAL, sys.argv[1], frames, [build_orig, build1, build2, build3, build4, build5, build6, build7], "oil")
