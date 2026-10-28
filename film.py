"""THE HOTEL THAT DREAMS OF ME — 60 s procedural film in the style of Sabin Balasa.

Usage: python3 film.py <frames_dir> [frame numbers...]
"""
import math
import multiprocessing as mp
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import characters as ch
from lib import (BOIL, FPS, H, OH, OW, W, X, Y, add_glow, almond_pts, blur_l, clip, col, cola,
                 ease, font, from_pil, lerp, over, ramp, stipple, stroke_mod, to_pil, vgrad, vnoise)

A = {}
AMBER = (0.98, 0.70, 0.28)
RNG = np.random.default_rng


# =====================================================================================
# helpers
# =====================================================================================
def overlay(f, draw_fn):
    """Draw with PIL onto a transparent layer and composite it over float frame f."""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(lay), lay)
    return composite(f, lay)


def composite(f, lay):
    a = np.asarray(lay, np.float32) / 255
    return over(f, a[..., :3], a[..., 3])


def draw_np(f, draw_fn):
    im = to_pil(f)
    draw_fn(ImageDraw.Draw(im))
    return from_pil(im)


class Corridor:
    """One-point perspective corridor. World: x lateral, y down (floor = FLOOR), z depth."""

    def __init__(self, vx=640, vy=400, wc=700, floor=560, ceil=520):
        self.vx, self.vy, self.wc, self.floor, self.ceil = vx, vy, wc, floor, ceil

    def p(self, x, y, z):
        s = 1 / (1 + z)
        return (self.vx + x * s, self.vy + y * s)

    def wall_quad(self, side, z0, z1, y0, y1):
        x = side * self.wc
        return [self.p(x, y0, z0), self.p(x, y0, z1), self.p(x, y1, z1), self.p(x, y1, z0)]

    def shell(self, d, wall, floor, ceil, zmax, n=110):
        zs = np.expm1(np.linspace(0, math.log1p(zmax), n))
        for z0, z1 in zip(zs[:-1], zs[1:]):
            fog = (0.18 + 0.82 * (1 / (1 + (z0 + z1) / 2)) ** 0.55)
            for side in (-1, 1):
                d.polygon(self.wall_quad(side, z0, z1, -self.ceil, self.floor), fill=col(wall, fog))
            w = self.wc
            d.polygon([self.p(-w, self.floor, z0), self.p(w, self.floor, z0), self.p(w, self.floor, z1),
                       self.p(-w, self.floor, z1)], fill=col(floor, fog))
            d.polygon([self.p(-w, -self.ceil, z0), self.p(w, -self.ceil, z0), self.p(w, -self.ceil, z1),
                       self.p(-w, -self.ceil, z1)], fill=col(ceil, fog))
        s = 1 / (1 + zmax)
        d.rectangle([self.vx - self.wc * s, self.vy - self.ceil * s, self.vx + self.wc * s, self.vy + self.floor * s],
                    fill=col(wall, 0.15))


def fogk(z):
    return 0.18 + 0.82 * (1 / (1 + z)) ** 0.55


# =====================================================================================
# SHOT 1 — the hotel on the ocean floor
# =====================================================================================
HX = np.linspace(440, 840, 9)
HY = np.linspace(285, 765, 14)
DARK_WIN = (HX[5], HY[7])          # 7th floor (rows counted from the bottom)


def window_shape(d, x, y, fill):
    d.rectangle([x - 11, y - 6, x + 11, y + 13], fill=fill)
    d.ellipse([x - 11, y - 16, x + 11, y + 4], fill=fill)


def build1():
    rng = RNG(11)
    bg = vgrad((0.12, 0.25, 0.42), (0.02, 0.05, 0.12), 0, H, 0.9)
    rays = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(rays)
    for k in range(6):
        x0 = 120 + k * 210 + rng.uniform(-40, 40)
        dr.polygon([(x0 - 30, 0), (x0 + 35, 0), (x0 + 230, H), (x0 + 70, H)], fill=70)
    A["s1_rays"] = blur_l(rays, 30)

    hotel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(hotel)
    ivory = (0.80, 0.76, 0.64)
    d.rectangle([400, 230, 880, 860], fill=col(ivory))
    d.rectangle([450, 175, 830, 232], fill=col(ivory, 0.95))
    d.rectangle([520, 128, 760, 177], fill=col(ivory, 0.9))
    d.ellipse([588, 78, 692, 160], fill=col(ivory, 0.92))
    d.line([(640, 78), (640, 30)], fill=col(ivory, 0.8), width=4)
    for y in (232, 520, 800):                       # cornices
        d.rectangle([392, y - 6, 888, y + 6], fill=col(ivory, 1.08))
    for x in np.linspace(414, 866, 10):             # pilasters
        d.line([(x, 240), (x, 850)], fill=col(ivory, 0.88), width=5)
    win = Image.new("L", (W, H), 0)
    dw = ImageDraw.Draw(win)
    one = Image.new("L", (W, H), 0)
    for x in HX:
        for y in HY:
            window_shape(d, x, y, col(AMBER))
            window_shape(dw, x, y, 255)
            if (x, y) == DARK_WIN:
                window_shape(ImageDraw.Draw(one), x, y, 255)
    for x in (520, 600, 680, 760):                  # crown windows
        window_shape(d, x, 205, col(AMBER, 0.9))
        window_shape(dw, x, 205, 255)
    d.rectangle([598, 790, 682, 860], fill=col(AMBER))
    d.ellipse([598, 755, 682, 825], fill=col(AMBER))
    dw.rectangle([598, 790, 682, 860], fill=255)
    for _ in range(1100):                           # barnacles, thicker toward the sea floor
        x = rng.uniform(395, 885)
        y = 860 - abs(rng.normal(0, 200))
        if y < 120:
            continue
        r = rng.uniform(1.5, 4.5)
        c = (0.86, 0.83, 0.74) if rng.random() < 0.5 else (0.55, 0.55, 0.52)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col(c))
        if r > 3:
            d.ellipse([x - r * 0.35, y - r * 0.35, x + r * 0.35, y + r * 0.35], fill=col((0.3, 0.3, 0.3)))
    ha = np.asarray(hotel, np.float32) / 255
    shade = vgrad((1.05, 1.05, 1.05), (0.50, 0.60, 0.72), 80, 860)
    is_win = from_pil(win)[..., None]
    rgb = ha[..., :3] * (shade * (1 - is_win) + is_win)
    f = over(bg, rgb, ha[..., 3])
    # sea floor
    sea = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sea)
    xs = np.linspace(0, W, 60)
    top = 840 + 20 * np.sin(xs * 0.011) + rng.normal(0, 4, xs.size)
    ds.polygon([(0, H)] + list(zip(xs, top)) + [(W, H)], fill=col((0.06, 0.08, 0.12)))
    for _ in range(260):
        x, y = rng.uniform(0, W), rng.uniform(850, H)
        r = rng.uniform(4, 22)
        ds.ellipse([x - r, y - r * 0.5, x + r, y + r * 0.5], fill=col((0.10, 0.13, 0.18) if rng.random() < 0.5 else (0.03, 0.04, 0.07)))
    f = composite(f, sea)
    A["s1"] = f
    A["s1_glow_all"] = blur_l(win, 9)
    A["s1_glow_one"] = blur_l(one, 9)
    A["snow"] = (rng.uniform(0, W, 380), rng.uniform(0, H, 380), rng.uniform(4, 14, 380), rng.uniform(1, 3, 380))


def whale(d, x, y, s, flip, c=(0.03, 0.07, 0.15)):
    up, lo = [], []
    for u in np.linspace(0, 1, 32):
        th = 60 * math.sin(math.pi * min(u * 1.05, 1)) ** 0.6 * (0.3 + 0.7 * u)
        xx = -200 + 430 * u
        up.append((x + flip * xx * s, y - th * 0.9 * s))
        lo.append((x + flip * xx * s, y + th * 0.7 * s))
    d.polygon(up + lo[::-1], fill=col(c))
    d.polygon([(x + flip * -195 * s, y), (x + flip * -268 * s, y - 44 * s), (x + flip * -238 * s, y),
               (x + flip * -268 * s, y + 44 * s)], fill=col(c))
    d.polygon([(x + flip * 70 * s, y + 25 * s), (x + flip * 10 * s, y + 85 * s), (x + flip * 100 * s, y + 35 * s)], fill=col(c))
    for k in range(5):  # pale throat grooves
        yy = y + (10 + k * 7) * s
        d.line([(x + flip * 90 * s, yy), (x + flip * 210 * s, yy - 8 * s)], fill=col((0.10, 0.16, 0.27)), width=max(1, int(3 * s)))


def shot1(t, v):
    f = A["s1"].copy()
    f += np.array([0.10, 0.16, 0.22]) * (np.roll(A["s1_rays"], int(25 * math.sin(t * 0.5)), 1) * (0.6 + 0.4 * math.sin(t * 0.8)))[..., None]
    dark = ramp(t, 2.8, 3.4)
    sil = ramp(t, 3.8, 4.6)
    f += np.array(AMBER) * (0.3 * (A["s1_glow_all"] - dark * A["s1_glow_one"]))[..., None]

    def draw(d, lay):
        wx, wy = DARK_WIN
        wc = tuple(lerp(np.array(AMBER), np.array((0.03, 0.04, 0.08)), dark))
        wc = tuple(lerp(np.array(wc), np.array((0.20, 0.25, 0.38)), sil * 0.8))
        window_shape(d, wx, wy, col(wc))
        if sil > 0:
            sc = cola((0.0, 0.0, 0.01), sil)
            d.ellipse([wx - 5, wy - 10, wx + 5, wy + 1], fill=sc)
            d.polygon([(wx - 10, wy + 13), (wx + 10, wy + 13), (wx + 7, wy + 3), (wx - 7, wy + 3)], fill=sc)
        whale(d, 1450 - 260 * t, 110, 0.75, -1)
        whale(d, -150 + 130 * t, 55, 0.4, 1, (0.05, 0.10, 0.19))
        xs, ys, sp, sz = A["snow"]
        for x, y, s_, z in zip(xs, ys, sp, sz):
            yy = (y + s_ * t * 4) % H
            xx = x + 6 * math.sin(t * 0.7 + y)
            d.ellipse([xx - z, yy - z, xx + z, yy + z], fill=(200, 215, 230, 110))

    f = overlay(f, draw)
    a = ramp(t, 0, 4.2)
    cx, cy, z = lerp(640, 650, a), lerp(480, 520, a), 1 + 0.25 * a
    p = ramp(t, 4.2, 7.8)
    cam = (lerp(cx, DARK_WIN[0], p), lerp(cy, DARK_WIN[1], p), z * (1 + 2.6 * p))
    return [(f, cam, 1.0, None)]


# =====================================================================================
# SHOT 2 — the horse-headed bellhop
# =====================================================================================
C2 = Corridor()
DOOR217 = (1.3, 1.9)
KEYHOLE = C2.p(700, 160, 1.38)


def portrait(d, quad, sepia=(0.55, 0.42, 0.30)):
    """Gold-framed picture on a wall quad, with a blank-faced figure."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad

    def q(a, b):
        top = (lerp(x0, x1, a), lerp(y0, y1, a))
        bot = (lerp(x3, x2, a), lerp(y3, y2, a))
        return (lerp(top[0], bot[0], b), lerp(top[1], bot[1], b))

    d.polygon(quad, fill=col((0.78, 0.6, 0.26)))
    d.polygon([q(0.1, 0.07), q(0.9, 0.07), q(0.9, 0.93), q(0.1, 0.93)], fill=col(sepia))
    head = [q(0.5 + 0.13 * math.cos(a), 0.38 + 0.11 * math.sin(a)) for a in np.linspace(0, 6.28, 20)]
    d.polygon([q(0.22, 0.93), q(0.78, 0.93), q(0.7, 0.6), q(0.3, 0.6)], fill=col(sepia, 0.65))
    d.polygon(head, fill=col((0.86, 0.80, 0.68)))


def build2():
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    C2.shell(d, (0.36, 0.07, 0.10), (0.22, 0.04, 0.06), (0.16, 0.04, 0.05), 30)
    for side in (-1, 1):  # carpet runner borders + wallpaper stripes
        a, b = C2.p(side * 380, 560, 0), C2.p(side * 380, 560, 30)
        d.line([a, b], fill=col((0.62, 0.46, 0.18)), width=5)
    for z in np.expm1(np.linspace(0.05, math.log1p(30), 40)):
        d.line([C2.p(-380, 560, z), C2.p(380, 560, z)], fill=col((0.5, 0.36, 0.12), fogk(z)), width=max(1, int(6 / (1 + z))))
    frames_l = [(0.3, 0.8), (1.4, 2.0), (2.8, 3.6), (4.8, 5.8), (7.5, 9), (11, 13), (16, 19)]
    frames_r = [(0.2, 0.7), (2.3, 2.9), (4.6, 5.4), (8, 9.2)]
    doors_r = [DOOR217, (3.3, 4.1), (6.0, 7.3), (10, 12)]
    for z0, z1 in frames_l:
        portrait(d, C2.wall_quad(-1, z0, z1, -260, 60), tuple(np.array((0.55, 0.42, 0.30)) * fogk(z0)))
    for z0, z1 in frames_r:
        portrait(d, C2.wall_quad(1, z0, z1, -260, 60), tuple(np.array((0.55, 0.42, 0.30)) * fogk(z0)))
    for z0, z1 in doors_r:
        k = fogk(z0)
        d.polygon(C2.wall_quad(1, z0, z1, -200, 560), fill=col((0.62, 0.46, 0.18), k))
        d.polygon(C2.wall_quad(1, z0 + 0.03, z1 - 0.03, -180, 560), fill=col((0.20, 0.07, 0.05), k))
    zc = sum(DOOR217) / 2
    x, y = C2.p(700, -240, zc)
    d.rectangle([x - 34, y - 18, x + 34, y + 18], fill=col((0.82, 0.64, 0.28)))
    d.text((x, y), "217", font=font(24, True), fill=col((0.15, 0.06, 0.04)), anchor="mm")
    kx, ky = KEYHOLE
    d.rectangle([kx - 7, ky - 16, kx + 7, ky + 16], fill=col((0.75, 0.58, 0.25)))
    d.ellipse([kx - 3, ky - 8, kx + 3, ky - 2], fill=(5, 2, 2))
    d.polygon([(kx - 2, ky - 4), (kx + 2, ky - 4), (kx + 3, ky + 7), (kx - 3, ky + 7)], fill=(5, 2, 2))
    lamps = Image.new("L", (W, H), 0)
    dl = ImageDraw.Draw(lamps)
    for z in (0.5, 2, 4, 7, 11, 17):
        x, y = C2.p(0, -500, z)
        s = 1 / (1 + z)
        d.ellipse([x - 60 * s, y - 14 * s, x + 60 * s, y + 14 * s], fill=col((1.0, 0.85, 0.55)))
        dl.ellipse([x - 60 * s, y - 14 * s, x + 60 * s, y + 14 * s], fill=255)
    f = from_pil(im)
    A["s2"] = add_glow(f, lamps, (0.9, 0.6, 0.3), 25, 0.9)


def shot2(t, v):
    f = A["s2"].copy()
    p = 1 - (1 - min(1.0, t / 4.6)) ** 2
    z = lerp(16, 2.0, p)
    s = 1 / (1 + z)
    L = lerp(0, 500, ramp(t, 2.6, 4.6))
    fs = s * 0.78
    fx, fy = C2.p(L, 560, z)
    walking = t < 4.6
    # shadow walks the other way, along the left wall
    zs = lerp(1.2, 18, p)
    ss = 1 / (1 + zs)
    sh = Image.new("L", (W, H), 0)
    sx, sy = C2.p(-640, 560, zs)
    ch.bellhop(ImageDraw.Draw(sh), sx, sy, ss * 0.78, walk=(-t * 7 if walking else None), sil=255)
    f = f * (1 - 0.6 * blur_l(sh, 3))[..., None]
    if t < 4.9:
        tray, glass = 0.0, True
    elif t < 5.7:
        tray, glass = ramp(t, 4.9, 5.6), True
    else:
        tray, glass = 1 - ramp(t, 5.8, 6.4), False

    def draw(d, lay):
        ch.bellhop(d, fx, fy, fs, walk=(t * 7 if walking else None), tray=tray, glass=glass)
        if glass:
            hx, hy = 175 + 40 * tray, -640 + 560 * tray - 16
            ch.steam(d, fx + hx * fs, fy + (hy - 86) * fs, fs, t)
        else:
            gx, gy = C2.p(L + 168, 560, z)
            ch.milk_glass(d, gx, gy, fs)
            ch.steam(d, gx, gy - 86 * fs, fs, t)

    f = overlay(f, draw)
    m = ramp(t, 6.25, 6.6) * (1 - ramp(t, 6.7, 7.0))
    if m > 0:  # a wet exhale through the keyhole
        mist = Image.new("L", (W, H), 0)
        r = 6 + 30 * ramp(t, 6.25, 7.0)
        ImageDraw.Draw(mist).ellipse([KEYHOLE[0] - r, KEYHOLE[1] - r * 0.7, KEYHOLE[0] + r, KEYHOLE[1] + r * 0.7], fill=255)
        f = f + np.array([0.6, 0.6, 0.62]) * (blur_l(mist, 8) * 0.6 * m)[..., None]
    q = ramp(t, 5.8, 7.0)
    cam = (lerp(640, KEYHOLE[0], q), lerp(420, KEYHOLE[1], q), (1 + 0.06 * min(t, 5.8) / 5.8) * (1 + 5 * q))
    return [(f, cam, 1.0, None)]


# =====================================================================================
# SHOT 3 — the elevator to floor -12
# =====================================================================================
BW = (330, 150, 950, 830)
DOOR = (440, 300, 840, 830)
DIAL = (640, 238, 60)


def wallpt(side, u, v):
    nx = 0 if side < 0 else W
    fx = BW[0] if side < 0 else BW[2]
    ny, fy = v * H, BW[1] + v * (BW[3] - BW[1])
    return (lerp(nx, fx, u), lerp(ny, fy, u))


def build3():
    rng = RNG(33)
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    brass, side_c = (0.50, 0.37, 0.15), (0.38, 0.28, 0.11)
    d.rectangle(BW, fill=col(brass))
    d.polygon([(0, 0), (W, 0), (BW[2], BW[1]), (BW[0], BW[1])], fill=col((0.22, 0.16, 0.07)))
    d.polygon([(0, 0), (BW[0], BW[1]), (BW[0], BW[3]), (0, H)], fill=col(side_c))
    d.polygon([(W, 0), (BW[2], BW[1]), (BW[2], BW[3]), (W, H)], fill=col(side_c))
    d.polygon([(0, H), (W, H), (BW[2], BW[3]), (BW[0], BW[3])], fill=col((0.80, 0.78, 0.72)))
    for side in (-1, 1):
        for u in (0.36, 0.62, 0.86):
            d.line([wallpt(side, u, 0), wallpt(side, u, 1)], fill=col(side_c, 0.75), width=4)
        for (u0, u1) in ((0.06, 0.30), (0.40, 0.58), (0.66, 0.82)):
            for (v0, v1) in ((0.12, 0.34), (0.44, 0.66)):
                portrait(d, [wallpt(side, u0, v0), wallpt(side, u1, v0), wallpt(side, u1, v1), wallpt(side, u0, v1)])
    for x0 in (348, 862):
        for y0 in (330, 470, 610):
            portrait(d, [(x0, y0), (x0 + 70, y0), (x0 + 70, y0 + 100), (x0, y0 + 100)])
    # marble veins
    f = from_pil(im)
    fl = Image.new("L", (W, H), 0)
    ImageDraw.Draw(fl).polygon([(0, H), (W, H), (BW[2], BW[3]), (BW[0], BW[3])], fill=255)
    veins = (np.abs(vnoise(34, 6, 3) - 0.5) < 0.012).astype(np.float32) * from_pil(fl)
    f *= (1 - 0.35 * veins)[..., None]
    # dial
    cx, cy, r = DIAL

    def dial(d):
        d.pieslice([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], 180, 360, fill=col((0.70, 0.54, 0.22)))
        d.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=col((0.88, 0.82, 0.64)))
        for k in range(13):
            a = math.radians(180 + k * 15)
            d.line([(cx + math.cos(a) * r * 0.8, cy + math.sin(a) * r * 0.8), (cx + math.cos(a) * r * 0.95, cy + math.sin(a) * r * 0.95)],
                   fill=col((0.2, 0.15, 0.1)), width=2)
            if k % 2 == 0:
                d.text((cx + math.cos(a) * r * 0.62, cy + math.sin(a) * r * 0.62), str(-k) if k else "0",
                       font=font(11), fill=col((0.2, 0.15, 0.1)), anchor="mm")
    f = draw_np(f, dial)
    lamp = Image.new("L", (W, H), 0)
    ImageDraw.Draw(lamp).ellipse([580, 40, 700, 90], fill=255)
    f = draw_np(f, lambda d: d.ellipse([580, 40, 700, 90], fill=col((1.0, 0.9, 0.65))))
    A["s3"] = add_glow(f, lamp, (0.8, 0.6, 0.3), 40, 1.2)
    A["s3_lamp"] = blur_l(lamp, 120)


def sunflower(d, x0, y0, h, t, grow):
    pts = []
    for i in range(30):
        q = i / 29
        pts.append((x0 + 16 * math.sin(q * 3 + t * 0.8) * q, y0 - h * q))
    d.line(pts, fill=col((0.24, 0.40, 0.12)), width=12, joint="curve")
    for q, side in ((0.35, -1), (0.6, 1)):
        if q * h < 40:
            continue
        x, y = pts[int(q * 29)]
        lw = 70 * grow
        d.polygon([(x, y), (x + side * lw * 0.5, y - 30 * grow), (x + side * lw, y - 8 * grow), (x + side * lw * 0.5, y + 12 * grow)],
                  fill=col((0.28, 0.46, 0.14)))
    hx, hy = pts[-1]
    pr = 70 * grow
    for k in range(18):
        a = 2 * math.pi * k / 18 + 0.1 * math.sin(t)
        tip = (hx + math.cos(a) * pr, hy + math.sin(a) * pr * 0.85)
        lft = (hx + math.cos(a - 0.17) * pr * 0.45, hy + math.sin(a - 0.17) * pr * 0.4)
        rgt = (hx + math.cos(a + 0.17) * pr * 0.45, hy + math.sin(a + 0.17) * pr * 0.4)
        d.polygon([lft, tip, rgt, (hx, hy)], fill=col((0.98, 0.78, 0.12) if k % 2 else (0.92, 0.66, 0.08)))
    cr = 34 * grow
    d.ellipse([hx - cr, hy - cr * 0.85, hx + cr, hy + cr * 0.85], fill=col((0.30, 0.18, 0.08)))


def shot3(t, v):
    f = A["s3"].copy()
    f *= (0.94 + 0.06 * math.sin(t * 13) * math.sin(t * 3.1))
    open_ = ramp(t, 4.2, 5.4)
    floor_f = lerp(-5, -12, ramp(t, 0.5, 4.0))
    off = 200 * open_
    if open_ > 0:  # field beyond the doors
        x0, y0, x1, y1 = DOOR
        fld = to_pil(A["s4"]).resize((int((x1 - x0) * 1.6), int((y1 - y0) * 1.6)), Image.BICUBIC)
        crop = from_pil(fld)[int((y1 - y0) * 0.3):int((y1 - y0) * 0.3) + (y1 - y0), int((x1 - x0) * 0.3):int((x1 - x0) * 0.3) + (x1 - x0)]
        f[y0:y1, x0:x1] = crop

    def draw(d, lay):
        x0, y0, x1, y1 = DOOR
        pc = (0.62, 0.47, 0.20)
        for side in (-1, 1):
            if side < 0:
                a, b = x0 - off, 640 - off
            else:
                a, b = 640 + off, x1 + off
            a, b = max(a, x0), min(b, x1)
            if b - a < 2:
                continue
            d.rectangle([a, y0, b, y1], fill=col(pc))
            if b - a < 60:
                continue
            d.rectangle([a + 25, y0 + 30, b - 25, y0 + 240], outline=col(pc, 0.7), width=5)
            d.rectangle([a + 25, y0 + 280, b - 25, y1 - 30], outline=col(pc, 0.7), width=5)
        d.line([(640 - off, y0), (640 - off, y1)], fill=col(pc, 0.5), width=3)
        d.line([(640 + off, y0), (640 + off, y1)], fill=col(pc, 0.5), width=3)
        cx, cy, r = DIAL
        a = math.radians(180 + (-floor_f) * 15)
        d.line([(cx, cy), (cx + math.cos(a) * r * 0.9, cy + math.sin(a) * r * 0.9)], fill=col((0.6, 0.08, 0.06)), width=4)
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col((0.3, 0.2, 0.1)))
        d.rectangle([608, 250, 672, 286], fill=col((0.08, 0.05, 0.03)))
        d.text((640, 268), f"−{int(round(-floor_f))}", font=font(24, True), fill=col(AMBER), anchor="mm")
        grow = ramp(t, 0.0, 2.5)
        for k in range(6):  # cracked marble
            a = k * 1.05 + 0.3
            d.line([(520, 905), (520 + math.cos(a) * 70, 905 + math.sin(a) * 22)], fill=col((0.3, 0.3, 0.28)), width=3)
        if grow > 0:
            sunflower(d, 520, 905, 520 * grow, t, max(grow, 0.05))

    f = overlay(f, draw)
    p = ramp(t, 5.2, 7.8)
    return [(f, (640, lerp(480, 560, p), 1 + 1.6 * p), 1.0, None)]


# =====================================================================================
# SHOT 4 — the field of yellow flowers
# =====================================================================================
BOY = (800, 770, 0.42)
CLOTHES = [(0.15, 0.18, 0.32), (0.38, 0.26, 0.16), (0.42, 0.42, 0.44), (0.88, 0.86, 0.80), (0.55, 0.62, 0.55),
           (0.08, 0.08, 0.09), (0.50, 0.18, 0.15), (0.32, 0.36, 0.48), (0.60, 0.52, 0.36)]
HAIR = [(0.12, 0.08, 0.06), (0.30, 0.20, 0.12), (0.55, 0.50, 0.45), (0.45, 0.30, 0.15)]


def build4():
    rng = RNG(44)
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    sky = vgrad((0.02, 0.02, 0.03), (0.07, 0.07, 0.08), 0, 400)
    ground = vgrad((0.12, 0.10, 0.03), (0.22, 0.18, 0.05), 400, H)
    base = np.where((Y < 400)[..., None], sky, ground)
    im = to_pil(base)
    d = ImageDraw.Draw(im)
    for _ in range(140):  # cloud band
        x, y = rng.uniform(-50, W + 50), rng.uniform(-30, 120)
        r = rng.uniform(40, 110)
        d.ellipse([x - r, y - r * 0.5, x + r, y + r * 0.5], fill=col((0.16, 0.17, 0.22) if rng.random() < 0.6 else (0.24, 0.25, 0.31)))
    for x in (420, 470, 540):  # ropes
        d.line([(x, 100), (x + (x - 480) * 0.1, 165)], fill=col((0.35, 0.33, 0.3)), width=2)
    d.rectangle([400, 160, 560, 290], fill=col((0.62, 0.58, 0.52)))
    d.polygon([(385, 290), (575, 290), (480, 372)], fill=col((0.35, 0.22, 0.18)))
    d.rectangle([522, 300, 540, 352], fill=col((0.35, 0.22, 0.18)))
    d.rectangle([465, 160, 495, 212], fill=col((0.30, 0.22, 0.16)))
    for x, y in ((420, 230), (515, 230)):
        d.rectangle([x, y, x + 28, y + 34], fill=col((0.75, 0.62, 0.30)))
    d.line([(0, 400), (W, 400)], fill=col((0.40, 0.34, 0.08)), width=3)
    # flowers and people, back to front
    n = 12000
    u = rng.random(n) ** 1.6
    fys = 400 + 560 * u
    fxs = rng.uniform(-10, W + 10, n)
    people = []
    while len(people) < 75:
        fy = rng.uniform(440, 940)
        fx = rng.uniform(40, 1240)
        if abs(fx - BOY[0]) < 120 and 640 < fy < 880:
            continue
        people.append((fy, fx, int(rng.integers(0, len(CLOTHES))), int(rng.integers(0, len(HAIR))), rng.random()))
    people.sort()
    order = np.argsort(fys)
    pi = 0
    for i in order:
        y = fys[i]
        while pi < len(people) and people[pi][0] <= y:
            fy, fx, c, h, rr = people[pi]
            s = 0.10 + 0.62 * (fy - 400) / 560
            ch.back_figure(d, fx, fy, s, CLOTHES[c], HAIR[h], hat=(0.1, 0.1, 0.1) if rr < 0.12 else None, kind=1 if rr > 0.6 else 0)
            pi += 1
        dep = (y - 400) / 560
        r = 1.5 + 13 * dep ** 1.2
        k = 0.5 + 0.5 * dep
        c = (0.97, 0.80, 0.16) if rng.random() < 0.6 else (0.90, 0.64, 0.10)
        x = fxs[i]
        d.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=col(c, k))
        if r > 5:
            d.ellipse([x - r * 0.3, y - r * 0.22, x + r * 0.3, y + r * 0.22], fill=col((0.3, 0.18, 0.05), k))
    A["s4"] = from_pil(im)


def boy(d, cx, fy, s, front):
    P = lambda x, y: (cx + x * s, fy + y * s)
    stripe, cloth = col((0.30, 0.40, 0.62)), col((0.86, 0.87, 0.90))

    def striped(x0, y0, x1, y1):
        d.rectangle([*P(x0, y0), *P(x1, y1)], fill=cloth)
        x = x0 + 6
        while x < x1:
            d.rectangle([*P(x, y0), *P(min(x + 9, x1), y1)], fill=stripe)
            x += 24

    striped(-55, -150, -6, 0)
    striped(6, -150, 55, 0)
    striped(-62, -330, 62, -150)
    d.line([P(-58, -320), P(-80, -200)], fill=cloth, width=max(2, int(26 * s)))
    d.line([P(58, -320), P(100, -250)], fill=cloth, width=max(2, int(26 * s)))
    d.ellipse([*P(88, -266), *P(112, -242)], fill=col((0.88, 0.78, 0.68)))
    d.rectangle([*P(-14, -350), *P(14, -325)], fill=col((0.88, 0.78, 0.68)))
    if front:
        ch.face(d, *P(0, -410), s, (0.88, 0.80, 0.70), (0.30, 0.20, 0.12), eyes="black", hair_len=0.0, child=True)
    else:
        d.ellipse([*P(-66, -485), *P(66, -340)], fill=col((0.30, 0.20, 0.12)))
        for m in (-1, 1):
            d.ellipse([*P(m * 66 - 8, -425), *P(m * 66 + 8, -395)], fill=col((0.82, 0.70, 0.60)))


def shot4(t, v):
    f = A["s4"].copy()
    bx, bfy, bs = BOY
    k = ramp(t, 3.4, 4.1)
    sx = max(0.06, abs(math.cos(math.pi * k)))
    patch = Image.new("RGBA", (320, 460), (0, 0, 0, 0))
    boy(ImageDraw.Draw(patch), 160, 440, bs, front=k > 0.5)
    pw = max(2, int(320 * sx))
    patch = patch.resize((pw, 460), Image.BICUBIC)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lay.paste(patch, (int(bx - pw / 2), int(bfy - 440)), patch)
    f = composite(f, lay)
    a = ramp(t, 0, 6.0)
    cam_c = (lerp(640, bx - 40, a), lerp(480, bfy - 190, a))
    cam = (cam_c[0], cam_c[1], 1 + 0.9 * a)
    p = ramp(t, 6.0, 8.0)
    hand = (bx + 100 * bs * sx, bfy - 254 * bs)
    base = (bx + 60 * bs + 8 * math.sin(t * 1.3), bfy - 720 * bs + 6 * math.sin(t * 0.9))
    bc = (lerp(base[0], cam_c[0], ease(p)), lerp(base[1], cam_c[1] - 20, ease(p)))
    r = 46 * bs / 0.42 * (1 + 30 * p ** 3)

    def draw(d, lay):
        end = hand if p == 0 else (bc[0], bc[1] + r * 1.6)
        mid = ((hand[0] + bc[0]) / 2 + 10 * math.sin(t * 2), (hand[1] + bc[1]) / 2 + r * 0.3)
        d.line([hand, mid, (bc[0], bc[1] + r * 1.15)] if p == 0 else [end, (bc[0], bc[1] + r * 1.15)],
               fill=col((0.85, 0.85, 0.8)), width=2)
        d.ellipse([bc[0] - r, bc[1] - r * 1.15, bc[0] + r, bc[1] + r * 1.15], fill=col((0.80, 0.08, 0.07)))
        d.ellipse([bc[0] - r * 0.55, bc[1] - r * 0.8, bc[0] - r * 0.2, bc[1] - r * 0.35], fill=col((0.95, 0.45, 0.4)))
        d.polygon([(bc[0] - r * 0.12, bc[1] + r * 1.15), (bc[0] + r * 0.12, bc[1] + r * 1.15), (bc[0], bc[1] + r * 1.05)],
                  fill=col((0.6, 0.05, 0.05)))

    f = overlay(f, draw)
    return [(f, cam, 1.0, None)]


# =====================================================================================
# SHOT 5 — the mother in the upside-down house
# =====================================================================================
KWIN = (550, 425)
HALL = (1040, 250, 1230, 760)


def build5():
    rng = RNG(55)
    sky = vgrad((0.86, 0.86, 0.82), (0.70, 0.75, 0.78))
    sun = np.exp(-np.sqrt((X - 950) ** 2 + (Y - 180) ** 2) / 160)
    im = to_pil(sky + np.array([0.12, 0.10, 0.04]) * sun[..., None])
    d = ImageDraw.Draw(im)
    for _ in range(90):  # roots rising from the hanging soil
        x = rng.uniform(450, 830)
        pts = [(x, 160)]
        for i in range(6):
            pts.append((pts[-1][0] + rng.normal(0, 8), pts[-1][1] - rng.uniform(5, 18)))
        d.line(pts, fill=col((0.35, 0.26, 0.18)), width=2)
    d.polygon([(440, 140), (840, 140), (835, 178), (445, 178)], fill=col((0.30, 0.22, 0.15)))
    d.rectangle([450, 170, 830, 520], fill=col((0.80, 0.70, 0.52)))
    d.polygon([(420, 520), (860, 520), (640, 760)], fill=col((0.55, 0.28, 0.20)))
    for k in range(1, 7):
        y = 520 + k * 34
        half = 220 * (1 - k * 34 / 240)
        d.line([(640 - half, y), (640 + half, y)], fill=col((0.42, 0.20, 0.15)), width=3)
    d.rectangle([700, 600, 736, 700], fill=col((0.48, 0.24, 0.18)))
    d.rectangle([610, 178, 670, 300], fill=col((0.38, 0.25, 0.16)))
    d.ellipse([624, 272, 636, 284], fill=col(ch.GOLD))
    for (x0, y0) in ((680, 230), (500, 230), (680, 380)):
        d.rectangle([x0, y0, x0 + 100, y0 + 90], fill=col((0.45, 0.50, 0.55)))
        d.line([(x0 + 50, y0), (x0 + 50, y0 + 90)], fill=col((0.8, 0.72, 0.55)), width=4)
    d.rectangle([500, 380, 600, 470], fill=col((1.0, 0.86, 0.55)))
    d.line([(550, 380), (550, 470)], fill=col((0.8, 0.72, 0.55)), width=4)
    d.line([(500, 425), (600, 425)], fill=col((0.8, 0.72, 0.55)), width=4)
    A["s5_ext"] = from_pil(im)

    # kitchen
    wall = vgrad((0.88, 0.76, 0.50), (0.68, 0.54, 0.32), 0, 720)
    im = to_pil(wall)
    d = ImageDraw.Draw(im)
    d.rectangle([90, 170, 330, 470], fill=col((1.0, 0.96, 0.80)))
    d.line([(210, 170), (210, 470)], fill=col((0.6, 0.48, 0.3)), width=8)
    d.line([(90, 320), (330, 320)], fill=col((0.6, 0.48, 0.3)), width=8)
    d.rectangle([80, 160, 340, 480], outline=col((0.55, 0.42, 0.26)), width=10)
    x0, y0, x1, y1 = HALL
    vx, vy = (x0 + x1) / 2, (y0 + y1) / 2 - 20
    d.rectangle(HALL, fill=col((0.05, 0.07, 0.14)))
    for k in range(18):  # a hallway that goes on too long
        q = 1 - (k / 18) ** 0.6
        ax, ay = lerp(vx, x0, q), lerp(vy, y0, q)
        bx, by = lerp(vx, x1, q), lerp(vy, y1, q)
        c = 0.12 + 0.5 * q
        d.rectangle([ax, ay, bx, by], outline=col((0.35, 0.40, 0.55), c), width=2)
        if k % 2 == 0 and k < 16:
            dw = (bx - ax) * 0.12
            for side_x in (ax + 2, bx - dw - 2):
                d.rectangle([side_x, ay + (by - ay) * 0.25, side_x + dw, by - 2], fill=col((0.16, 0.30, 0.58), c * 1.4))
    d.rectangle(HALL, outline=col((0.55, 0.42, 0.26)), width=10)
    d.rectangle([0, 720, W, H], fill=col((0.86, 0.86, 0.80)))
    for x in range(0, W, 48):
        d.line([(x, 720), (x - 60, H)], fill=col((0.55, 0.62, 0.78)), width=3)
    for y in range(740, H, 48):
        d.line([(0, y), (W, y)], fill=col((0.55, 0.62, 0.78)), width=3)
    f = from_pil(im)
    beam = Image.new("L", (W, H), 0)
    ImageDraw.Draw(beam).polygon([(330, 170), (330, 470), (760, 960), (1060, 960)], fill=90)
    f = f + np.array([0.25, 0.22, 0.12]) * blur_l(beam, 30)[..., None]
    A["s5_int"] = f


def clock(d, cx, cy, r, t):
    d.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=col((0.55, 0.42, 0.22)))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col((0.94, 0.90, 0.80)))
    for k in range(12):
        a = math.radians(k * 30)
        d.line([(cx + math.sin(a) * r * 0.82, cy - math.cos(a) * r * 0.82), (cx + math.sin(a) * r * 0.95, cy - math.cos(a) * r * 0.95)],
               fill=col((0.2, 0.15, 0.1)), width=3)
    for ang, ln, w in ((300, 0.5, 6), (120 - t * 3, 0.75, 4), ((30 - math.floor(t)) * 6, 0.88, 2)):
        a = math.radians(ang)
        d.line([(cx, cy), (cx + math.sin(a) * r * ln, cy - math.cos(a) * r * ln)], fill=col((0.12, 0.08, 0.06)), width=w)


def shot5(t, v):
    out = []
    kin = ramp(t, 2.2, 3.0)
    if kin < 1:
        p = ramp(t, 0.3, 2.8)
        out.append((A["s5_ext"], (lerp(640, KWIN[0], p), lerp(480, KWIN[1], p), 1 + 4 * p), 1 - kin, None))
    if kin > 0:
        f = A["s5_int"].copy()
        look = t > 4.2
        speak = 4.6 < t < 8.4
        point = ramp(t, 7.8, 8.5)
        skin, hair, dress = (0.86, 0.80, 0.72), (0.35, 0.22, 0.14), (0.40, 0.45, 0.58)
        fs = 1.45
        fcx, fcy = 640, 430

        def body(d, lay):
            rx, ry = 68 * fs, 104 * fs
            d.polygon([(fcx - rx * 1.25, fcy - ry * 0.4), (fcx - rx * 0.9, fcy - ry * 1.12), (fcx, fcy - ry * 1.22),
                       (fcx + rx * 0.9, fcy - ry * 1.12), (fcx + rx * 1.25, fcy - ry * 0.4), (fcx + rx * 1.4, fcy + ry * 2.0),
                       (fcx - rx * 1.4, fcy + ry * 2.0)], fill=col(hair))
            d.rectangle([610, 560, 670, 660], fill=col(skin, 0.9))
            d.polygon([(470, 640), (810, 640), (850, 760), (430, 760)], fill=col(dress))
            d.rectangle([0, 720, W, 760], fill=col((0.86, 0.86, 0.80)))
            peel = 30 + 220 * ramp(t, 0, 9)
            pts = [(662 + 14 * math.sin(i * 0.9 + t), 712 + peel * i / 20) for i in range(21)]
            d.line(pts, fill=col((0.75, 0.12, 0.10)), width=8, joint="curve")
            d.line([(520, 650), (560, 700), (615, 705)], fill=col(dress), width=34, joint="curve")
            d.ellipse([598, 688, 632, 720], fill=col(skin))
            d.ellipse([616, 662, 676, 718], fill=col((0.78, 0.10, 0.08)))
            d.ellipse([626, 670, 642, 684], fill=col((0.95, 0.55, 0.5)))
            hx, hy = lerp(690, 960, point), lerp(705, 520, point)
            d.line([(760, 650), (lerp(730, 860, point), lerp(700, 590, point)), (hx, hy)], fill=col(dress), width=34, joint="curve")
            d.ellipse([hx - 17, hy - 15, hx + 17, hy + 15], fill=col(skin))
            if point > 0.5:
                d.line([(hx + 10, hy - 4), (hx + 52, hy - 22)], fill=col(skin), width=9)
            else:
                d.line([(hx - 4, hy - 10), (hx - 24, hy - 34)], fill=col((0.75, 0.76, 0.78)), width=4)
            clock(d, 960, 190, 70, t)

        f = overlay(f, body)
        # the face that will not come into focus
        fl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dfl = ImageDraw.Draw(fl)
        ch.face(dfl, fcx, fcy, fs, skin, hair, eyes="open" if look else "down", hair_len=0.0)
        if speak:
            m = 0.5 + 0.5 * math.sin(t * 17) * math.sin(t * 5.3)
            dfl.ellipse([fcx - 14, fcy + 87 - 2 - 6 * m, fcx + 14, fcy + 87 + 2 + 6 * m], fill=col((0.25, 0.08, 0.08)))
        rb = 2.5 + 3.5 * (0.5 + 0.5 * math.sin(t * 2.1))
        blurred = fl.filter(ImageFilter.GaussianBlur(rb))
        f = composite(f, blurred)
        ghost = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ghost.paste(blurred, (int(14 * math.sin(t * 1.3)), int(6 * math.sin(t * 1.9))))
        ga = np.asarray(ghost, np.float32) / 255
        f = over(f, ga[..., :3], ga[..., 3] * 0.4)
        q = ramp(t, 8.2, 10.0)
        a = ramp(t, 2.2, 8.2)
        cam = (lerp(640, (HALL[0] + HALL[2]) / 2, q), lerp(lerp(480, 460, a), 485, q), lerp(1.15 + 0.1 * a, 3.4, q))
        out.append((f, cam, kin, None))
    return out


# =====================================================================================
# SHOT 6 — the hallway of blue doors
# =====================================================================================
C6 = Corridor(vy=400)
Z6 = 6.0
DOORS6 = [(0.25, 0.7), (1.0, 1.45), (1.8, 2.3), (2.7, 3.15), (3.6, 4.05), (4.45, 4.9), (5.3, 5.75)]


def build6():
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    C6.shell(d, (0.40, 0.45, 0.53), (0.20, 0.16, 0.14), (0.28, 0.31, 0.38), Z6)
    s = 1 / (1 + Z6)
    d.rectangle([640 - 700 * s, 400 - 520 * s, 640 + 700 * s, 400 + 560 * s], fill=col((0.30, 0.35, 0.42), 0.55))
    lights = []
    for side in (-1, 1):
        for z0, z1 in DOORS6:
            k = fogk(z0)
            d.polygon(C6.wall_quad(side, z0 - 0.04, z1 + 0.04, -230, 560), fill=col((0.75, 0.70, 0.60), k))
            d.polygon(C6.wall_quad(side, z0, z1, -210, 560), fill=col((0.16, 0.30, 0.58), k))
            zc = (z0 + z1) / 2
            x, y = C6.p(side * 700, -40, zc)
            sc = 1 / (1 + zc)
            d.rectangle([x - 14 * sc, y - 9 * sc, x + 14 * sc, y + 9 * sc], fill=col((0.78, 0.62, 0.30), k))
            d.line([(x - 9 * sc + i * 3 * sc, y + 2 * sc * math.sin(i * 1.7)) for i in range(7)], fill=col((0.2, 0.12, 0.06)), width=1)
            lights.append(C6.wall_quad(side, z0, z1, 548, 560))
            kx, ky = C6.p(side * 700, 180, z0 + 0.08 if side > 0 else z1 - 0.08)
            d.ellipse([kx - 7 * sc, ky - 7 * sc, kx + 7 * sc, ky + 7 * sc], fill=col(ch.GOLD, k))
    A["s6"] = from_pil(im)
    A["s6_lights"] = lights


def end_door(d, m, z, t):
    """End-of-hall door, drawn after the camera move so it stays sharp."""
    s = 1 / (1 + Z6)
    x0, x1 = 640 - 150 * s, 640 + 150 * s
    y0, y1 = 400 - 250 * s, 400 + 560 * s
    P = lambda x, y: m(x, y)
    d.rectangle([*P(x0 - 3, y0 - 3), *P(x1 + 3, y1)], fill=col((0.75, 0.70, 0.60), 0.4))
    d.rectangle([*P(x0, y0), *P(x1, y1)], fill=(3, 3, 5))
    ang = math.radians(78) * ramp(t, 3.0, 3.9) * (1 - ramp(t, 7.9, 8.05))
    hand = ramp(t, 4.0, 5.4) * (1 - ramp(t, 7.85, 7.95))
    if hand > 0:
        hx, hy = lerp(x1 - 6, x1 + 4, hand), 400 + 140 * s
        k = 1.0 + 0.6 * hand
        a = int(255 * hand)
        pale = cola((0.86, 0.85, 0.80), hand)
        d.polygon([P(x0 + 18, hy - 3 * k), P(hx, hy - 4 * k), P(hx, hy + 4 * k), P(x0 + 18, hy + 3 * k)], fill=pale)
        d.ellipse([*P(hx - 2, hy - 6 * k), *P(hx + 9 * k, hy + 6 * k)], fill=pale)
        for i in range(4):
            fy = hy - 5 * k + i * 3.3 * k
            d.line([P(hx + 8 * k, fy), P(hx + 14 * k, fy + 1.5 * k)], fill=pale, width=max(1, int(2.6 * k * z)))
        bx, by = hx + 13 * k, hy - 8 * k
        brass = (180, 140, 60, a)
        d.ellipse([*P(bx - 3.2 * k, by - 3.2 * k), *P(bx + 3.2 * k, by + 3.2 * k)], outline=brass, width=max(1, int(1.6 * k * z)))
        d.line([P(bx, by + 3 * k), P(bx + 3 * k, by + 14 * k)], fill=brass, width=max(1, int(1.6 * k * z)))
        d.line([P(bx + 2.4 * k, by + 11 * k), P(bx + 5 * k, by + 10.5 * k)], fill=brass, width=max(1, int(1.4 * k * z)))
    w = (x1 - x0) * math.cos(ang)
    dy = 6 * s * math.sin(ang)
    blue = col((0.16, 0.30, 0.58), 0.75)
    d.polygon([P(x0, y0), P(x0 + w, y0 - dy), P(x0 + w, y1 + dy), P(x0, y1)], fill=blue)
    kx = x0 + w * 0.82
    d.ellipse([*P(kx - 1.6, 400 + 175 * s - 1.6), *P(kx + 1.6, 400 + 175 * s + 1.6)], fill=col(ch.GOLD, 0.7))


def shot6(t, v):
    if t >= 8.15:
        return []
    f = A["s6"].copy()
    rng = RNG(int(t * FPS) + 600)

    def draw(d, lay):
        for i, q in enumerate(A["s6_lights"]):
            fl = 0.5 + 0.5 * math.sin(t * (2 + i % 5) + i * 1.7) * (0.7 + 0.3 * rng.random())
            d.polygon(q, fill=cola((1.0, 0.82, 0.45), 0.25 + 0.6 * fl))

    f = overlay(f, draw)
    zc = 1 + 3.3 * ramp(t, 0, 8.0)
    return [(f, (640, 425, zc), 1.0, lambda d, m, z: end_door(d, m, z, t))]


# =====================================================================================
# SHOT 7 — room 217, the key
# =====================================================================================
BH = (640, 880, 0.62)
NECK = (BH[0], BH[1] - 770 * BH[2])


def build7():
    f = vgrad((0.09, 0.13, 0.26), (0.05, 0.07, 0.15), 0, 640)
    im = to_pil(f)
    d = ImageDraw.Draw(im)
    d.polygon([(0, 640), (W, 640), (W, H), (0, H)], fill=col((0.10, 0.12, 0.20)))
    for x in range(-400, W + 400, 70):
        d.line([(640 + (x - 640) * 0.45, 640), (x, H)], fill=col((0.13, 0.15, 0.25)), width=3)
    d.line([(250, 0), (250, 640)], fill=col((0.06, 0.09, 0.18)), width=4)
    d.line([(1030, 0), (1030, 640)], fill=col((0.06, 0.09, 0.18)), width=4)
    d.rectangle([500, 120, 780, 520], fill=col((0.55, 0.68, 0.88)))
    d.line([(640, 120), (640, 520)], fill=col((0.2, 0.28, 0.45)), width=8)
    d.line([(500, 320), (780, 320)], fill=col((0.2, 0.28, 0.45)), width=8)
    for x0, x1 in ((450, 600), (680, 830)):
        d.rectangle([x0, 95, x1, 580], fill=col((0.24, 0.34, 0.55)))
        for k in range(8):
            x = x0 + (x1 - x0) * (k + 0.5) / 8
            d.line([(x, 95), (x + 6 * math.sin(k), 580)], fill=col((0.36, 0.48, 0.72)), width=6)
    d.rectangle([440, 85, 840, 100], fill=col((0.3, 0.22, 0.14)))
    portrait(d, [(120, 220), (210, 230), (210, 360), (120, 370)], (0.28, 0.30, 0.42))
    f = from_pil(im)
    shaft = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shaft).polygon([(560, 330), (720, 330), (930, 960), (360, 960)], fill=60)
    f = f + np.array([0.25, 0.32, 0.45]) * blur_l(shaft, 25)[..., None]
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).rectangle([600, 120, 680, 520], fill=255)
    A["s7"] = add_glow(f, glow, (0.3, 0.4, 0.6), 40, 0.5)

    front = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(front)
    d.rectangle([40, 600, 330, 640], fill=col((0.30, 0.20, 0.14)))
    d.rectangle([60, 640, 310, H], fill=col((0.24, 0.16, 0.11)))
    d.rectangle([90, 700, 280, 790], outline=col((0.18, 0.12, 0.08)), width=4)
    ch.milk_glass(d, 190, 604, 1.3)
    d.rectangle([230, 690, 1050, 760], fill=col((0.22, 0.13, 0.09)))
    for x in (230, 1010):
        d.rectangle([x, 640, x + 40, 790], fill=col((0.18, 0.10, 0.07)))
        d.ellipse([x - 6, 618, x + 46, 660], fill=col((0.26, 0.16, 0.10)))
    xs = np.linspace(100, 1180, 40)
    top = 770 - 25 * np.exp(-((xs - 520) / 70) ** 2) - 25 * np.exp(-((xs - 760) / 70) ** 2)
    d.polygon([(100, H)] + list(zip(xs, top)) + [(1180, H)], fill=col((0.42, 0.48, 0.62)))
    for k in range(9):
        y0 = 800 + k * 18
        d.arc([150 + k * 20, y0 - 30, 1130 - k * 20, y0 + 60], 200, 340, fill=col((0.32, 0.38, 0.52)), width=4)
    A["s7_front"] = front
    rng = RNG(77)
    pts = []
    while len(pts) < 60:
        x, y = rng.uniform(-1, 1, 2)
        if x * x + y * y < 0.9 and y > 0.05:
            pts.append((x, y))
    A["refl"] = pts


def key_post(d, m, z, t, palm):
    """Brass key and its paper tag, drawn sharp after the camera move."""
    px, py = palm
    bx, by = px - 30, py - 6
    R, r = 17, 11.5
    gold, dark = (196, 152, 66, 255), (110, 80, 30, 255)
    sz = lambda v: max(1, int(v * z))
    # tag on a string
    tx, ty = px - 52, py + 26
    d.line([m(bx - 4, by + R - 2), m(tx + 10, ty)], fill=(220, 215, 200, 255), width=sz(0.9))
    tag = [m(tx - 8, ty - 2), m(tx + 40, ty + 4), m(tx + 37, ty + 26), m(tx - 11, ty + 20)]
    d.polygon(tag, fill=(222, 210, 178, 255))
    d.ellipse([*m(tx - 4, ty + 1), *m(tx + 0, ty + 5)], outline=(120, 100, 70, 255), width=sz(0.6))
    d.text(m(tx + 15, ty + 13), "ROOM 1", font=font(max(6, 8.5 * z), True), fill=(60, 40, 25, 255), anchor="mm")
    # shaft and teeth
    d.line([m(bx + R - 1, by), m(bx + 62, by + 3)], fill=gold, width=sz(5))
    for i, (dx, h) in enumerate(((50, 9), (56, 6), (61, 10))):
        d.rectangle([*m(bx + dx, by + 2), *m(bx + dx + 4, by + 2 + h)], fill=gold)
    # bow, with the field reflected inside it
    d.ellipse([*m(bx - R, by - R), *m(bx + R, by + R)], fill=gold)
    d.ellipse([*m(bx - R + 2, by - R + 2), *m(bx + R - 2, by + R - 2)], outline=dark, width=sz(1))
    d.ellipse([*m(bx - r, by - r), *m(bx + r, by + r)], fill=(4, 4, 6, 255))
    d.line([m(bx - r * 0.95, by + r * 0.05), m(bx + r * 0.95, by + r * 0.05)], fill=(90, 75, 20, 255), width=sz(0.4))
    for x, y in A["refl"]:
        rr = 0.25 + 0.6 * y
        d.ellipse([*m(bx + x * r - rr, by + y * r - rr * 0.7), *m(bx + x * r + rr, by + y * r + rr * 0.7)], fill=(240, 196, 40, 255))
    d.arc([*m(bx - R + 3, by - R + 3), *m(bx + R - 3, by + R - 3)], 200, 260, fill=(250, 235, 180, 255), width=sz(1.2))


def shot7(t, v):
    if t >= 11.2:
        return []
    f = A["s7"].copy()
    cx, fy, s = BH
    ka = ramp(t, 2.0, 3.8)
    tilt = ramp(t, 9.0, 10.0)
    swap = int(round(t * FPS)) == int(round(10.8 * FPS))
    lh = (cx - 138 * s, fy - 440 * s)
    rh = (cx + (150 - 120 * ka) * s, fy + (-440 - 140 * ka) * s)

    def draw(d, lay):
        ch.bellhop(d, cx, fy, s, tray=None, wet=True, key_arm=max(ka, 1e-3), head=False)
        for i in range(40):  # drips that never reach the carpet
            ts = i * 0.31
            if ts > t:
                break
            hx, hy = (lh if i % 2 == 0 else rh)
            age = t - ts
            y = hy + 18 + 0.5 * 900 * age ** 2
            a = 1 - clip((y - 610) / 50)
            if a <= 0:
                continue
            d.ellipse([hx - 3, y - 5, hx + 3, y + 3], fill=cola((0.62, 0.72, 0.85), float(a) * 0.9))

    f = overlay(f, draw)
    head = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(head)
    hcx, hcy = cx, fy - 890 * s
    if swap:
        ch.face(hd, hcx, hcy + 10, 0.85, (0.82, 0.78, 0.72), (0.30, 0.20, 0.12), eyes="open", hair_len=0.15)
    else:
        ch.horse_head(hd, hcx, hcy, s)
    if tilt > 0:
        head = head.rotate(-16 * tilt, resample=Image.BICUBIC, center=NECK)
    f = composite(f, head)
    f = composite(f, A["s7_front"])
    f = overlay(f, lambda d, lay: ch.steam(d, 190, 604 - 112, 1.3, t, 0.45))
    a = ramp(t, 0, 3.6)
    cam = (640, lerp(480, 470, a), 1 + 0.1 * a)
    kp = (rh[0], rh[1])
    bow = (kp[0] - 10, kp[1] + 4)
    q = ramp(t, 3.6, 6.4)
    cam = (lerp(cam[0], bow[0], q), lerp(cam[1], bow[1], q), lerp(cam[2], 4.4, q))
    q2 = ramp(t, 7.4, 8.8)
    cam = (lerp(cam[0], 640, q2), lerp(cam[1], hcy + 10, q2), lerp(cam[2], 2.1, q2))
    post = (lambda d, m, z: key_post(d, m, z, t, kp)) if ka > 0.7 else None
    return [(f, cam, 1.0, post)]


# =====================================================================================
# timeline, camera, finishing
# =====================================================================================
# (function, start s, duration s, dissolve-in s)
SHOTS = [(shot1, 0, 7, 0), (shot2, 7, 7, 0.8), (shot3, 14, 7, 0), (shot4, 21, 8, 1.0),
         (shot5, 29, 9, 1.0), (shot6, 38, 9, 1.0), (shot7, 47, 13, 0)]
TOTAL = 60 * FPS


def camera(frame, cam, v, post):
    """Warp (line boil), paint texture, crop/zoom, then draw sharp overlays."""
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
        post(ImageDraw.Draw(lay), lambda x, y: ((x - x0) * z, (y - y0) * z), z)
        im = Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")
    return from_pil(im)


def layers_at(T):
    out = []
    for k, (fn, st, dur, _) in enumerate(SHOTS):
        nxt = SHOTS[k + 1][3] if k + 1 < len(SHOTS) else 0
        if not (st <= T < st + dur + nxt):
            continue
        w = 1.0
        _, _, _, xin = SHOTS[k]
        if xin and T < st + xin and k > 0:
            w = ease((T - st) / xin)
        if nxt and T >= st + dur:
            w = 1 - ease((T - st - dur) / nxt)
        for f, cam, lw, post in fn(T - st, None):
            out.append((f, cam, w * lw, post))
    return out


def balasa_grade(f):
    """Push toward the reference: dark toned paper, muted palette, saturated accents kept."""
    f = clip(f)
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    chroma = f.max(-1) - f.min(-1)
    sat = 0.42 + 0.58 * clip((chroma - 0.22) / 0.3)
    g = lum[..., None] + (f - lum[..., None]) * sat[..., None]
    g = clip(g) ** 1.25 * 0.94
    sh = (1 - clip(lum / 0.55))[..., None]
    return g * (1 - 0.3 * sh) + np.array([0.025, 0.04, 0.085], np.float32) * sh


def render(i):
    T = i / FPS
    v = (i // 3) % BOIL
    acc = np.zeros((H, W, 3), np.float32)
    lays = layers_at(T)
    for f, cam, w, post in lays:
        if w <= 0.001:
            continue
        acc += camera(f, cam, v, post) * w
    acc = balasa_grade(acc)
    f = acc * A["tooth"][v][..., None] + 0.06 * A["speck"][v][..., None] * (1 - acc.mean(-1, keepdims=True))
    f *= min(1.0, T / 1.0)
    rng = RNG(5000 + i)
    f *= 1 + 0.015 * math.sin(i * 0.7) + rng.normal(0, 0.006)
    im = to_pil(f).resize((OW, OH), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    if lays:
        a += rng.normal(0, 5, (OH, OW, 1))
        a *= A["vig"][..., None]
        if rng.random() < 0.25:
            a = np.roll(a, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), (0, 1))
        for _ in range(rng.poisson(0.5)):
            yy, xx = rng.integers(0, OH), rng.integers(0, OW)
            a[max(0, yy - 1):yy + 1, max(0, xx - 1):xx + 1] = 190
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(OUTDIR, f"f_{i:04d}.png"), compress_level=1)
    return i


def build_all():
    A["warp"], A["paint"], A["tooth"], A["speck"] = [], [], [], []
    for i in range(BOIL):
        dx = (vnoise(300 + i, 12, 2) - 0.5) * 8
        dy = (vnoise(310 + i, 12, 2) - 0.5) * 8
        A["warp"].append((np.clip(Y + dy, 0, H - 1).astype(np.int32), np.clip(X + dx, 0, W - 1).astype(np.int32)))
        A["paint"].append(1 + 0.19 * stroke_mod(340 + i, 20000, 20, -1.0, 0.35) + 0.24 * (vnoise(350 + i, 5) - 0.5)
                          + 0.38 * (stipple(360 + i, 0.06) - 0.06))
        g = RNG(320 + i).random((H, W)).astype(np.float32)
        A["tooth"].append(0.8 + 0.4 * (from_pil(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9))) * 2 - 0.5))
        A["speck"].append(stipple(330 + i, 0.03))
    oy, ox = np.mgrid[0:OH, 0:OW].astype(np.float32)
    A["vig"] = 1 - 0.35 * clip(((ox - OW / 2) / (OW / 2)) ** 2 + ((oy - OH / 2) / (OH / 2)) ** 2) ** 1.5
    for b in (build1, build2, build4, build3, build5, build6, build7):
        b()
        print("built", b.__name__, flush=True)


if __name__ == "__main__":
    OUTDIR = sys.argv[1]
    os.makedirs(OUTDIR, exist_ok=True)
    build_all()
    frames = [int(x) for x in sys.argv[2:]] or list(range(TOTAL))
    with mp.get_context("fork").Pool(4) as p:
        for n, _ in enumerate(p.imap_unordered(render, frames, chunksize=2)):
            if n % 120 == 0:
                print("frame", n, flush=True)
    print("done")
