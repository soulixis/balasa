"""THE DROWNED TEMPLE — silent stop-motion film from public-domain / CC0 museum pieces, in the style of Sabin Balasa.

Usage: python3 temple.py <frames_dir> [frame numbers...]
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import cutouts as cu
import engine
from birds import bird, jit, make_tiles, sea, step, texture
from engine import A
from lib import FPS, H, W, Y, clip, col, ease, from_pil, lerp, ramp, to_pil, vgrad
from metamorph import comp, comp_c, fade, xform

RNG = np.random.default_rng
RAW = os.path.join(os.path.dirname(cu.ASSETS), "raw")
HOR = 700
CORAL = (0.95, 0.45, 0.30)


# =====================================================================================
# helpers
# =====================================================================================
def ghost(path, alpha=0.38, width=1150):
    """A historic temple view, toned into the night sky like a half-remembered image."""
    im = Image.open(path).convert("RGBA")
    im = cu.tone(im, lo=(0.05, 0.06, 0.13), hi=(0.36, 0.40, 0.54), keep=0.0, torn=False, seed=3)
    s = width / im.width
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    yy, xx = np.mgrid[0:im.height, 0:im.width].astype(np.float32)
    v = clip(1 - (((xx - im.width / 2) / (im.width / 2)) ** 2 + ((yy - im.height / 2) / (im.height / 2)) ** 2) ** 1.5)
    a = np.asarray(im.getchannel("A"), np.float32) / 255 * v * alpha
    im.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    return im


def frame_layer(piece, cx, bottom):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lay.alpha_composite(piece, (int(cx - piece.width / 2), int(bottom - piece.height)))
    return lay


def glow_piece(piece, color, radius, strength):
    g, pad = cu.halo(piece, color, radius, strength)
    return g, pad


def base(ts, bgkey="bg_a"):
    im = to_pil(A[bgkey]).convert("RGBA")
    return im


def water(im, ts):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle([0, HOR, W, H], fill=255)
    im.paste(A["sea_fill"], (0, 0), m)
    sea(ImageDraw.Draw(im), HOR, ts)


def piece(key, h, keep=0.25, seed=0, crop=None, **kw):
    im = cu.load(key)
    if crop:
        im = im.crop((int(im.width * crop[0]), int(im.height * crop[1]), int(im.width * crop[2]), int(im.height * crop[3])))
        im = im.crop(im.getbbox())
    return cu.scaled(cu.tone(im, keep=keep, seed=seed, **kw), h)


# =====================================================================================
# build
# =====================================================================================
def build():
    sky = vgrad((0.03, 0.04, 0.10), (0.10, 0.12, 0.21), 0, HOR) * texture((1, 1, 1), 5, angle=-0.3, length=22)
    seab = texture((0.04, 0.06, 0.13), 6, angle=0.0, jit_=0.05, length=30, ss=0.2)
    night = clip(np.where((Y < HOR)[..., None], sky, seab))
    A["sea_fill"] = to_pil(seab)
    for key, path, width, x, bottom in (("bg_a", "wc_32572562.jpg", 1180, 640, HOR + 10),
                                        ("bg_b", "artvee_temple_ruins_2.jpg", 1300, 640, HOR + 40)):
        im = to_pil(night).convert("RGBA")
        g = ghost(os.path.join(RAW, path), width=width)
        comp(im, g, x - g.width / 2, bottom - g.height)
        A[key] = from_pil(im.convert("RGB"))
    # gateway built from rising stones
    gate = piece("148071", 600, keep=0.2, seed=1)
    A["gate"] = gate
    A["gate_pos"] = (640, 722)
    tiles = make_tiles(frame_layer(gate, 640, 722), 30, 8, 11)
    r = RNG(12)
    top = 722 - gate.height
    for tl in tiles:
        hx, hy = tl["home"]
        tl["arr"] = 0.9 + 5.0 * (1 - (hy - top) / gate.height) + r.uniform(-0.25, 0.25)
        tl["x0"] = hx + r.normal(0, 90)
        tl["spin"] = r.uniform(-300, 300)
    A["gate_tiles"] = tiles
    face = piece("152487", 430, keep=0.25, seed=2)
    A["face"] = face
    A["face_glow"] = glow_piece(face, (0.55, 0.65, 0.95), 70, 0.6)
    A["conch"] = piece("679656", 120, seed=3)
    A["flyers"] = [piece("155586", 210, keep=0.45, seed=4), piece("153617", 190, keep=0.45, seed=5),
                   piece("130636", 220, keep=0.45, seed=6), piece("155586", 190, keep=0.45, seed=7).transpose(Image.FLIP_LEFT_RIGHT)]
    A["nataraja"] = piece("111811", 860, keep=0.1, seed=8, lo=(0.12, 0.03, 0.03), hi=(0.99, 0.58, 0.38))
    A["nat_glow"] = glow_piece(A["nataraja"], (0.95, 0.45, 0.30), 50, 0.35)
    A["sparks"] = [(r.uniform(0, 2 * math.pi), r.uniform(0.6, 6.5), r.uniform(10, 18)) for _ in range(60)]
    A["horus"] = piece("119791", 170, seed=9)
    A["mandala"] = piece("160954", 220, keep=0.5, seed=10, hi=(0.98, 0.90, 0.66))
    A["mandala_glow"] = glow_piece(A["mandala"], (0.98, 0.85, 0.55), 60, 0.55)
    # tower of faces
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    stack = [("170075", 300, (0, 0, 1, 0.62)), ("105881", 250, None), ("159834", 215, None), ("152487", 185, None),
             ("145411", 175, (0.2, 0, 0.8, 0.30))]
    y = 790
    for i, (key, h, crop) in enumerate(stack):
        p = piece(key, h, seed=20 + i, crop=crop)
        lay.alpha_composite(p, (int(640 - p.width / 2), int(y - p.height)))
        y -= h * 0.66
    A["tower"] = lay
    tt = make_tiles(lay, 28, 8, 13)
    top = y
    for tl in tt:
        hx, hy = tl["home"]
        tl["arr"] = 0.6 + 4.6 * clip((790 - hy) / (790 - top)) + r.uniform(-0.2, 0.2)
        tl["x0"] = hx + r.normal(0, 70)
        tl["spin"] = r.uniform(-300, 300)
        tl["drop"] = 0.6 + 4.8 * clip((hy - top) / (790 - top)) + r.uniform(-0.2, 0.2)
        tl["vx"] = r.normal(0, 40)
    A["tower_tiles"] = tt
    A["lotus"] = piece("144932", 210, keep=0.4, seed=30, lo=(0.05, 0.14, 0.14), hi=(0.70, 0.88, 0.76))
    A["gold"] = to_pil(clip(vgrad((0.99, 0.90, 0.64), (0.86, 0.62, 0.30), 120, 320) * texture((1, 1, 1), 41, angle=-0.4, ss=0.25))).convert("RGBA")


# =====================================================================================
# shots
# =====================================================================================
def draw_gate_tiles(im, ts):
    for tl in A["gate_tiles"]:
        q = ease((ts - (tl["arr"] - 1.4)) / 1.4)
        if q <= 0:
            continue
        hx, hy = tl["home"]
        x, y = lerp(tl["x0"], hx, q), lerp(1010, hy, q)
        img = tl["img"]
        rot = tl["spin"] * (1 - q)
        if abs(rot) > 0.5:
            img = img.rotate(rot, resample=Image.BICUBIC, expand=True)
        dx, dy = jit(tl["k"], ts, 0.8)
        comp(im, img, x - img.width / 2 + dx, y - img.height / 2 + dy)


def draw_face(im, ts, cy, a=1.0):
    g, pad = A["face_glow"]
    f = A["face"]
    comp(im, fade(g, a * (0.8 + 0.2 * math.sin(ts * 1.3))), 640 - f.width / 2 - pad, cy - f.height / 2 - pad)
    comp(im, fade(f, a), 640 - f.width / 2, cy - f.height / 2)


def draw_gate(im):
    g = A["gate"]
    comp(im, g, 640 - g.width / 2, 722 - g.height)


FACE_Y = 400


def s1(t):
    """Stones rise out of the sea and lock into a gateway."""
    ts = step(t)
    im = base(ts)
    water(im, ts)
    draw_gate_tiles(im, ts)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle([0, HOR + 18, W, H], fill=255)
    im.paste(A["sea_fill"], (0, 0), m)
    sea(ImageDraw.Draw(im), HOR + 18, ts)
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def scene(im, ts, face_y=FACE_Y, face_a=1.0, extra_behind=None):
    if extra_behind:
        extra_behind(im)
    if face_a > 0:
        draw_face(im, ts, face_y, face_a)
    draw_gate(im)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle([0, HOR + 18, W, H], fill=255)
    im.paste(A["sea_fill"], (0, 0), m)
    sea(ImageDraw.Draw(im), HOR + 18, ts)


def s2(t):
    """Through the gate, a Bodhisattva face rises like a moon."""
    ts = step(t)
    im = base(ts)
    cy = lerp(980, FACE_Y, ease((ts - 0.3) / 5.2))
    scene(im, ts, cy, ramp(ts, 0.0, 1.5))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def orbit(i, ts, start):
    phi = i * math.pi / 2 + 0.55 * (ts - start)
    return 640 + 400 * math.cos(phi), FACE_Y - 30 + 170 * math.sin(phi), math.sin(phi)


def flyers(im, ts, start, front, leave=None):
    for i, p in enumerate(A["flyers"]):
        e = ts - (start + i * 0.9)
        if e <= 0:
            continue
        q = ease(e / 1.8)
        ox, oy, depth = orbit(i, ts, start)
        cx, cy = lerp(990, ox, q), lerp(HOR - 10, oy, q)
        s = lerp(0.15, 1.0, q) * (0.85 + 0.15 * depth)
        a = 1.0
        if leave is not None:
            lq = ease((ts - leave - i * 0.2) / 2.0)
            cx, cy, a = cx + (cx - 640) * 1.5 * lq, cy - 500 * lq, 1 - lq
        if (depth > 0) != front or a <= 0:
            continue
        pp = xform(p, s, s, 8 * math.sin(ts * 2 + i))
        comp_c(im, fade(pp, a), cx, cy)


def s3(t):
    """A conch drifts on the water; apsaras and musicians float out of it and circle the face."""
    ts = step(t)
    im = base(ts)
    scene(im, ts, extra_behind=lambda im_: flyers(im_, ts, 0.8, False))
    flyers(im, ts, 0.8, True)
    c = A["conch"]
    comp_c(im, xform(c, rot=6 * math.sin(ts * 1.7)), 990 + 10 * math.sin(ts * 0.8), HOR + 4 + 4 * math.sin(ts * 2.1))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def s4(t):
    """Shiva's ring of fire descends behind the face; its flames fly off as coral birds."""
    ts = step(t + 8)
    tl = t
    im = base(ts)
    n = A["nataraja"]
    ny = lerp(-520, FACE_Y + 20, ease((tl - 0.2) / 3.0))
    rot = -18 * ease((tl - 3.0) / 5.0)

    def behind(im_):
        flyers(im_, ts, 0.8, False, leave=8.4)
        g, pad = A["nat_glow"]
        comp_c(im_, xform(g, rot=rot), 640, ny)
        comp_c(im_, xform(n, rot=rot), 640, ny)

    scene(im, ts, extra_behind=behind)
    flyers(im, ts, 0.8, True, leave=8.4)
    d = ImageDraw.Draw(im)
    for i, (a0, t0, sz) in enumerate(A["sparks"]):
        if tl < t0 or tl < 3.2:
            continue
        k = tl - max(t0, 3.2)
        a = a0 + rot * math.pi / 180
        x = 640 + math.cos(a) * (390 + 260 * k)
        y = ny + math.sin(a) * (390 + 260 * k) - 90 * k * k
        if -40 < x < W + 40 and -40 < y < H:
            bird(d, x, y, sz, ts * 22 + i, c=CORAL)
    c = A["conch"]
    comp_c(im, xform(c, rot=6 * math.sin(ts * 1.7)), 990 + 10 * math.sin(ts * 0.8), HOR + 4 + 4 * math.sin(ts * 2.1))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def tear_mask(tl):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    r = RNG(40)
    prog = ease((tl - 0.6) / 3.6)
    close = ease((tl - 4.6) / 2.4)
    for k in range(int(70 * prog)):
        u = k / 70
        x = lerp(1150, 260, u) + r.normal(0, 14)
        y = 200 + 60 * math.sin(u * 5) + r.normal(0, 10)
        rr = (34 + 26 * math.sin(u * 9 + 1)) * (1 - close)
        if rr > 1:
            d.ellipse([x - rr, y - rr * 0.7, x + rr, y + rr * 0.7], fill=255)
    return m.filter(ImageFilter.GaussianBlur(2))


MOON = (300, 210)


def s5(t):
    """The Horus falcon crosses the sky; the sky tears behind it into a lotus mandala that becomes the moon."""
    ts = step(t + 16)
    tl = t
    im = base(ts)
    m = tear_mask(tl)
    im.paste(A["gold"], (0, 0), m)
    rim = Image.new("RGBA", (W, H), col((0.05, 0.04, 0.06)) + (255,))
    edge = np.clip(np.asarray(m.filter(ImageFilter.GaussianBlur(5)), np.float32) - np.asarray(m, np.float32) * 0.9, 0, 255)
    im.paste(rim, (0, 0), Image.fromarray((edge * 1.6).clip(0, 255).astype(np.uint8)))
    mg, pad = A["mandala_glow"]
    mand = A["mandala"]
    ma = ramp(tl, 2.6, 4.0)
    if ma > 0:
        comp_c(im, fade(mg, ma), *MOON)
        comp_c(im, fade(mand, ma), *MOON)
    scene(im, ts)
    hx = lerp(1350, -150, ease((tl - 0.3) / 4.8))
    comp_c(im, xform(A["horus"], rot=-6 + 4 * (round(ts * 12) % 2)), hx, 200 + 30 * math.sin(tl * 1.4))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def moon(im, ts):
    mg, pad = A["mandala_glow"]
    comp_c(im, fade(mg, 0.85 + 0.15 * math.sin(ts * 1.1)), *MOON)
    comp_c(im, A["mandala"], *MOON)


def tower_tiles(im, ts, sinking=None):
    for tl in A["tower_tiles"]:
        hx, hy = tl["home"]
        if sinking is None:
            q = ease((ts - (tl["arr"] - 1.3)) / 1.3)
            if q <= 0:
                continue
            x, y, rot = lerp(tl["x0"], hx, q), lerp(1000, hy, q), tl["spin"] * (1 - q)
        else:
            k = sinking - tl["drop"]
            if k <= 0:
                x, y, rot = hx, hy, 0
            else:
                x, y, rot = hx + tl["vx"] * k, hy + 0.5 * 700 * k * k, tl["spin"] * 0.3 * k
                if y > H + 40:
                    continue
        img = tl["img"].rotate(rot, resample=Image.BICUBIC, expand=True) if abs(rot) > 0.5 else tl["img"]
        dx, dy = jit(tl["k"] + 900, ts, 0.8)
        comp(im, img, x - img.width / 2 + dx, y - img.height / 2 + dy)


def s6(t):
    """The drowned gods' faces rise from the sea and stack into a tower, under the mandala moon."""
    ts = step(t + 32)
    im = base(ts, "bg_b")
    moon(im, ts)
    tower_tiles(im, step(t))
    water(im, ts)
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def s7(t):
    """The tower sinks piece by piece; only lotus leaves stay on the water."""
    ts = step(t + 40)
    im = base(ts, "bg_b")
    moon(im, ts)
    tower_tiles(im, ts, sinking=step(t))
    water(im, ts)
    for i, (x0, sp) in enumerate(((300, 14), (520, -10), (820, 12), (1040, -8))):
        a = ramp(t, 2.0 + i * 0.6, 3.5 + i * 0.6)
        if a > 0:
            lp = xform(A["lotus"], 1.0, 0.45, 4 * math.sin(ts + i))
            comp_c(im, fade(lp, a), x0 + sp * t, HOR + 40 + i * 28 + 3 * math.sin(ts * 2 + i))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1 - ramp(t, 6.2, 8.0), None)]


SHOTS = [(s1, 0, 8, 0), (s2, 8, 8, 0), (s3, 16, 8, 0), (s4, 24, 8, 0), (s5, 32, 8, 0), (s6, 40, 8, 1.0), (s7, 48, 8, 0)]
TOTAL = 56 * FPS

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[2:]]
    engine.run(SHOTS, TOTAL, sys.argv[1], frames, [build], "pastel", boil=2)
