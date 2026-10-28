"""BIRDS AND THREAD — silent stop-motion short in the style of Sabin Balasa.

A flock lands and becomes a stone man. His thought becomes a house. The house's light
becomes thread, the thread becomes a woman. They touch, unravel into birds and fish,
and rebuild themselves as a boat that the sea folds away.

Everything animates on twos (12 drawings per second) like hand-placed cut-outs.
Usage: python3 birds.py <frames_dir> [frame numbers...]
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import engine
from engine import A
from lib import FPS, H, W, X, Y, clip, col, ease, from_pil, lerp, ramp, stipple, stroke_mod, to_pil, vgrad, vnoise

RNG = np.random.default_rng
PROFILE = [(470, 205), (560, 180), (640, 220), (668, 290), (676, 335), (672, 352), (722, 420), (684, 440), (690, 458),
           (676, 474), (688, 490), (672, 530), (630, 562), (618, 600), (624, 960), (512, 960), (528, 600), (474, 470),
           (452, 330)]
SH = (-40, 60)
MAN = [(x + SH[0], y + SH[1] + (60 if y == 960 else 0)) for x, y in PROFILE]
EYE = (590, 407)
WX = 40
WOMAN = [(1280 - x + WX, y) for x, y in PROFILE]
CORAL = [(0.95, 0.42, 0.28), (0.82, 0.20, 0.14), (0.98, 0.58, 0.40), (0.88, 0.36, 0.30)]
ROSE = [(0.86, 0.48, 0.44), (0.95, 0.66, 0.56), (0.78, 0.32, 0.28)]
INK = (0.04, 0.04, 0.06)


# =====================================================================================
# helpers
# =====================================================================================
def step(t):
    """Stop-motion time: the world only changes 12 times a second."""
    return math.floor(t * 12 + 1e-6) / 12


def jit(i, ts, amp=1.2):
    s = round(ts * 12)
    return amp * math.sin(i * 12.9898 + s * 78.233), amp * math.sin(i * 39.346 + s * 11.135)


def texture(base, seed, n=12000, length=16, angle=-1.0, jit_=0.3, sn=0.22, ss=0.16, st=0.0, grain=0.05):
    r = RNG(seed + 99)
    f = (1 + (vnoise(seed, 5) - 0.5) * 2 * sn + stroke_mod(seed + 1, n, length, angle, jit_) * ss
         + r.normal(0, grain, (H, W)).astype(np.float32))
    if st:
        f += stipple(seed + 2, 0.06) * st
    return clip(np.array(base, np.float32)[None, None, :] * f[..., None])


def poly_mask(pts, blur=1.2):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    return from_pil(im.filter(ImageFilter.GaussianBlur(blur))) if blur else from_pil(im)


def rgba_layer(rgb, alpha):
    a = np.dstack([clip(rgb), clip(alpha)[..., None]])
    return Image.fromarray((a * 255).astype(np.uint8), "RGBA")


def ell(d, cx, cy, rx, ry, fill):
    d.ellipse([cx - abs(rx), cy - abs(ry), cx + abs(rx), cy + abs(ry)], fill=fill)


def make_tiles(layer, cell, pad, seed):
    """Cut a painted layer into torn paper pieces that cover it exactly when in place."""
    r = RNG(seed)
    al = np.asarray(layer.getchannel("A"), np.float32) / 255
    tiles = []
    for gy in range(0, H, cell):
        for gx in range(0, W, cell):
            if al[gy:gy + cell, gx:gx + cell].mean() < 0.05:
                continue
            sz = cell + 2 * pad
            box = (gx - pad, gy - pad, gx - pad + sz, gy - pad + sz)
            crop = layer.crop(box)
            m = Image.new("L", (sz, sz), 0)
            c = sz / 2
            pts = []
            for k in range(11):
                a = 2 * math.pi * k / 11 + r.uniform(-0.15, 0.15)
                rr = r.uniform(cell * 0.73, cell * 0.73 + pad)
                pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
            ImageDraw.Draw(m).polygon(pts, fill=255)
            a_ = np.minimum(np.asarray(crop.getchannel("A")), np.asarray(m))
            crop.putalpha(Image.fromarray(a_))
            tiles.append(dict(img=crop, home=(gx + cell / 2, gy + cell / 2), rot=r.uniform(-25, 25), k=len(tiles)))
    return tiles


def put(im, img, x, y, angle=0.0):
    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    im.paste(img, (int(round(x - img.width / 2)), int(round(y - img.height / 2))), img)


def bird(d, x, y, sz, ph, c=INK):
    up = math.sin(ph)
    ell(d, x, y, sz * 0.38, sz * 0.17, col(c))
    for m in (-1, 1):
        d.polygon([(x, y - sz * 0.08), (x + m * sz * 0.5, y - sz * 0.15 - sz * 0.3 * up), (x + m * sz * 1.15, y - sz * 0.75 * up),
                   (x + m * sz * 0.55, y + sz * 0.1 - sz * 0.1 * up), (x, y + sz * 0.08)], fill=col(c))
    ell(d, x + sz * 0.38, y - sz * 0.05, sz * 0.13, sz * 0.12, col(c))


def fish(d, x, y, sz, ang, ph, c):
    ca, sa = math.cos(ang), math.sin(ang)
    R = lambda u, v: (x + u * ca - v * sa, y + u * sa + v * ca)
    w = 0.25 * math.sin(ph)
    d.polygon([R(sz, 0), R(sz * 0.3, sz * 0.32), R(-sz * 0.6, sz * 0.12), R(-sz * 0.6, -sz * 0.12), R(sz * 0.3, -sz * 0.32)], fill=col(c))
    d.polygon([R(-sz * 0.55, 0), R(-sz * 1.15, sz * (0.42 + w)), R(-sz * 0.95, 0), R(-sz * 1.15, -sz * (0.42 - w))], fill=col(c, 0.85))
    ell(d, *R(sz * 0.55, -sz * 0.06), sz * 0.07, sz * 0.07, col(INK))


def sea(d, horizon, ts, color=(0.42, 0.50, 0.66), seed=5, top=None):
    r = RNG(seed)
    y = horizon + 4.0
    xs = np.arange(-24, W + 48, 24, dtype=np.float32)
    sv = math.floor(ts * 12) % 3
    while y < H + 10:
        dep = clip((y - horizon) / max(1, H - horizon))
        amp = 2 + 12 * dep
        ph = r.uniform(0, 6.28) + sv * 0.9
        fr = r.uniform(0.006, 0.012)
        ys = y + amp * np.sin(xs * fr + ph)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=col(color, r.uniform(0.45, 1.05)), width=2 + int(dep * 2))
        y += 4 + 12 * dep


def strand(d, s, ts, off=(0, 0), frac_override=None, ease_override=None):
    """Draw one thread: it snakes in from a loose shape and settles onto its target path."""
    e = ease((ts - s["t0"]) / 1.5) if ease_override is None else ease_override
    frac = ramp(ts, s["t0"], s["t0"] + 1.0) if frac_override is None else frac_override
    if frac <= 0:
        return
    pts = s["pts"]
    n = max(2, int(len(pts) * frac))
    p = pts[:n] + (1 - e) * s["loose"][:n]
    p = p + np.array(off)
    d.line([tuple(q) for q in p.tolist()], fill=col(s["c"]), width=s["w"], joint="curve")


# =====================================================================================
# builds
# =====================================================================================
def build_backgrounds():
    sky = vgrad((0.08, 0.10, 0.19), (0.17, 0.18, 0.27), 0, 700) * texture((1, 1, 1), 10, angle=-0.3, jit_=0.2, length=22)
    seab = texture((0.06, 0.08, 0.15), 11, angle=0.0, jit_=0.05, length=30, ss=0.2)
    base = np.where((Y < 700)[..., None], sky, seab)
    sun = np.sqrt((X - 1010) ** 2 + (Y - 200) ** 2)
    base = base + np.array([0.06, 0.06, 0.07]) * np.exp(-sun / 70)[..., None]
    base = np.where((sun < 22)[..., None], np.array([0.76, 0.74, 0.66]) * (0.9 + 0.2 * stipple(12, 0.2))[..., None], base)
    A["bg_sea"] = clip(base)
    A["bg_char"] = texture((0.13, 0.12, 0.115), 13, angle=-0.9)
    A["bg_navy"] = clip(vgrad((0.05, 0.07, 0.15), (0.09, 0.11, 0.20)) * texture((1, 1, 1), 14, angle=-0.4))
    A["bg_deep"] = texture((0.08, 0.09, 0.15), 15, angle=-1.2)
    sky7 = vgrad((0.11, 0.12, 0.20), (0.23, 0.21, 0.27), 0, 600) * texture((1, 1, 1), 16, angle=-0.2, length=24)
    sun7 = np.sqrt((X - 320) ** 2 + (Y - 250) ** 2)
    sky7 = sky7 + np.array([0.18, 0.15, 0.10]) * np.exp(-sun7 / 90)[..., None]
    sky7 = np.where((sun7 < 34)[..., None], np.array([0.86, 0.78, 0.60]) * (0.9 + 0.2 * stipple(17, 0.2))[..., None], sky7)
    A["bg_dusk"] = clip(sky7)
    A["sea_fill"] = texture((0.06, 0.08, 0.15), 18, angle=0.0, jit_=0.05, length=30, ss=0.2)


def build_man():
    mask = poly_mask(MAN)
    shade = 0.55 + 0.5 * clip((X - 412) / 240)
    shade = shade * (1 - 0.3 * clip((Y - 620) / 340))
    ink = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(ink)
    d.arc([558, 388, 622, 426], 15, 165, fill=200, width=4)
    d.arc([554, 356, 630, 406], 200, 300, fill=90, width=3)
    d.line([(622, 535), (644, 536)], fill=170, width=3)
    d.arc([632, 484, 656, 504], 180, 300, fill=120, width=3)
    d.arc([476, 390, 516, 456], 280, 80, fill=130, width=4)
    d.line(MAN[2:14], fill=110, width=3)
    inkf = from_pil(ink.filter(ImageFilter.GaussianBlur(1.0)))
    rgb = texture((0.73, 0.73, 0.71), 60, n=12000, ss=0.18, st=0.6, grain=0.1) * shade[..., None] * (1 - 0.6 * inkf[..., None])
    A["man"] = make_tiles(rgba_layer(rgb, mask), 32, 8, 1)
    r = RNG(2)
    for tl in A["man"]:
        hx, hy = tl["home"]
        tl["land"] = 2.4 + 3.4 * clip((1020 - hy) / 840) + r.uniform(-0.25, 0.25)
        tl["r0"] = r.uniform(170, 330)
        tl["a0"] = r.uniform(0, 6.28)
        tl["sz"] = r.uniform(14, 22)
    A["extra_birds"] = [(r.uniform(150, 330), r.uniform(0, 6.28), r.uniform(12, 20), r.uniform(4.5, 6.0)) for _ in range(34)]
    A["eye_tile"] = min(A["man"], key=lambda tl: (tl["home"][0] - EYE[0]) ** 2 + (tl["home"][1] - EYE[1]) ** 2)["k"]


def house_layer(cx, cy, s, seed):
    P = lambda x, y: (cx + x * s, cy + y * s)
    walls = Image.new("L", (W, H), 0)
    ImageDraw.Draw(walls).rectangle([*P(-90, 0), *P(90, 130)], fill=255)
    roof = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(roof)
    dr.polygon([P(-108, 2), P(108, 2), P(0, -112)], fill=255)
    dr.rectangle([*P(42, -92), *P(64, -30)], fill=255)
    win = Image.new("L", (W, H), 0)
    dw = ImageDraw.Draw(win)
    dw.rectangle([*P(-40, 24), *P(40, 84)], fill=255)
    dw.rectangle([*P(-15, 92), *P(15, 130)], fill=255)
    wa, ra, wi = from_pil(walls), from_pil(roof), from_pil(win)
    rgb = texture((0.72, 0.70, 0.66), seed, st=0.55, grain=0.09) * (1 - wa * 0 + 0)[..., None]
    rgb = rgb * (1 - ra[..., None]) + texture((0.30, 0.30, 0.37), seed + 5, st=0.3) * ra[..., None]
    rgb = rgb * (1 - wi[..., None]) + np.array([0.04, 0.04, 0.07]) * wi[..., None]
    ink = Image.new("L", (W, H), 0)
    di = ImageDraw.Draw(ink)
    for k in range(1, 8):
        y = k * 16
        di.line([P(-90, y), P(90, y)], fill=70, width=max(1, int(2 * s)))
    di.line([P(0, 24), P(0, 84)], fill=160, width=max(2, int(4 * s)))
    di.line([P(-40, 54), P(40, 54)], fill=160, width=max(2, int(4 * s)))
    rgb = rgb * (1 - 0.4 * from_pil(ink)[..., None])
    alpha = np.maximum(wa, ra)
    return rgba_layer(rgb, alpha)


def build_house():
    A["house_s"] = make_tiles(house_layer(930, 250, 1.0, 20), 24, 6, 3)
    man_top = [tl for tl in A["man"] if tl["home"][1] < 330 and tl["home"][0] < 640]
    r = RNG(4)
    hs = sorted(A["house_s"], key=lambda tl: -tl["home"][1])
    for j, tl in enumerate(hs):
        src = man_top[j % len(man_top)]
        tl["src"] = src
        tl["dep"] = 2.0 + 3.4 * j / len(hs) + r.uniform(-0.1, 0.1)
        tl["spin"] = r.choice([-1, 1]) * r.uniform(180, 540)
        tl["lift"] = r.uniform(80, 220)
        src["gone"] = min(src.get("gone", 99), tl["dep"])
    big = house_layer(640, 380, 2.6, 21)
    A["house_b"] = big
    A["house_b_win"] = (640 - 40 * 2.6, 380 + 24 * 2.6, 640 + 40 * 2.6, 380 + 84 * 2.6)
    x0, y0, x1, y1 = A["house_b_win"]
    threads = []
    for k in range(40):
        p = np.array([r.uniform(x0 + 10, x1 - 10), r.uniform(y0 + 10, y1 - 10)])
        a = r.uniform(-0.3, math.pi + 0.3) if r.random() < 0.8 else r.uniform(0, 6.28)
        pts = [p.copy()]
        for _ in range(90):
            a += r.normal(0, 0.22)
            p = p + 13 * np.array([math.cos(a), math.sin(a)])
            pts.append(p.copy())
        threads.append(dict(pts=np.array(pts), t0=r.uniform(2.0, 4.8), c=CORAL[k % 4], w=int(r.integers(3, 6))))
    A["threads"] = threads


def resample(pts, step_=9.0):
    pts = np.asarray(pts, np.float32)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step_))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def build_woman():
    r = RNG(7)
    strands = []

    def add(pts, t0, c, w):
        pts = resample(pts)
        n = len(pts)
        ang = r.uniform(0, 6.28)
        u = np.linspace(0, 1, n)[:, None]
        loose = (r.uniform(90, 180) * np.array([math.cos(ang), math.sin(ang)])[None, :] * (0.4 + u)
                 + 40 * np.sin(u * r.uniform(6, 14) + r.uniform(0, 6)) * np.array([[1, -1]]))
        strands.append(dict(pts=pts, loose=loose.astype(np.float32), t0=t0, c=c, w=w))

    head = np.array([p for p in WOMAN if p[1] < 600], np.float32)
    cen = head.mean(0)
    poly = np.array(WOMAN + [WOMAN[0]], np.float32)
    for k, f in enumerate((1.0, 0.95, 0.90, 0.85, 0.80, 0.74, 0.68)):
        add(cen + (poly - cen) * f, 0.2 + 0.28 * k, CORAL[k % 2], 4 if k == 0 else 3)
    mask = poly_mask(WOMAN, 0)
    for j, y in enumerate(range(196, 950, 16)):
        xs = np.nonzero(mask[y] > 0.5)[0]
        if len(xs) < 4:
            continue
        xx = np.linspace(xs.max(), xs.min(), max(3, int((xs.max() - xs.min()) / 14)))
        add(np.stack([xx, y + 2 * np.sin(xx * 0.05 + j)], 1), 1.0 + 3.4 * (y - 196) / 754, ROSE[j % 3], 2)
    back = resample([(800, 190), (850, 205), (868, 330), (846, 470), (800, 560)], 4)
    for i in range(44):
        sx, sy = back[int(i / 43 * (len(back) - 1))]
        L, amp, fr, ph = r.uniform(250, 420), r.uniform(15, 40), r.uniform(1.0, 2.2), r.uniform(0, 6)
        q = np.linspace(0, 1, 40)
        pts = np.stack([sx + L * q, sy + (sy - 360) * 0.5 * q + amp * q ** 1.2 * np.sin(2 * math.pi * q * fr + ph)], 1)
        add(pts, 2.4 + 3.0 * r.random(), CORAL[i % 4], 3)
    ex, ey = 692, 347
    eye = [(ex + 32 * math.cos(math.radians(a)), ey + 19 * math.sin(math.radians(a))) for a in range(15, 166, 10)]
    add(eye, 4.6, (0.55, 0.10, 0.08), 4)
    add([(ex + 32 * math.cos(math.radians(a)), ey - 18 + 25 * math.sin(math.radians(a))) for a in range(200, 301, 10)], 4.8, (0.60, 0.14, 0.10), 3)
    add([(658, 476), (632, 477)], 5.0, (0.60, 0.10, 0.10), 4)
    A["woman"] = strands
    A["woman_mask"] = Image.fromarray((mask * 255).astype(np.uint8))
    A["woman_front"] = min(p[0] for p in WOMAN if 280 < p[1] < 300)


def boat_layer():
    hull = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(hull)
    pts = [(430, 560)] + [(640 + 210 * math.cos(a), 560 + 85 * math.sin(a)) for a in np.linspace(0, math.pi, 30)][::-1] + [(850, 560)]
    d.polygon([(430, 560), (850, 560)] + [(640 + 210 * math.cos(a), 560 + 85 * math.sin(a)) for a in np.linspace(0, math.pi, 30)], fill=255)
    d.rectangle([636, 250, 645, 562], fill=255)
    ha = from_pil(hull)
    rgb = texture((0.38, 0.34, 0.31), 30, st=0.5, grain=0.09)
    ink = Image.new("L", (W, H), 0)
    di = ImageDraw.Draw(ink)
    for k in range(1, 5):
        di.arc([430 + k * 8, 560 - 85 + k * 18, 850 - k * 8, 560 + 85 - k * 4], 0, 180, fill=110, width=3)
    rgb = rgb * (1 - 0.4 * from_pil(ink)[..., None])
    return rgba_layer(rgb, ha)


def build_boat():
    A["hull"] = make_tiles(boat_layer(), 26, 7, 5)
    r = RNG(8)
    for tl in A["hull"]:
        tl["s"] = 0.2 + 3.6 * r.random()
        tl["R0"] = r.uniform(650, 900)
        tl["a0"] = r.uniform(0, 6.28)
        tl["sz"] = r.uniform(14, 20)
    sail = []
    for k in range(22):
        y = lerp(272, 532, k / 21)
        xr = 648 + (y - 262) / 258 * 212 if y <= 520 else 860 - (y - 520) / 20 * 212
        xx = np.linspace(652, xr - 4, 18)
        yy = y + 10 * np.sin((xx - 652) / max(1, xr - 652) * math.pi) * 0.6
        sail.append(dict(pts=np.stack([xx, yy], 1), c=CORAL[k % 4], w=4, s=0.4 + 3.8 * r.random(), R0=r.uniform(650, 900),
                         a0=r.uniform(0, 6.28)))
    A["sail"] = sail
    # the finished boat with its two passengers, for the crossing
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for cx, c in ((585, (0.70, 0.70, 0.68)), (700, (0.86, 0.50, 0.46))):
        d.polygon([(cx - 24, 566), (cx + 24, 566), (cx + 18, 520), (cx - 18, 520)], fill=col(c, 0.75))
        ell(d, cx, 492, 15, 22, col(c))
        d.arc([cx - 9, 486, cx - 1, 494], 10, 170, fill=col(INK), width=2)
        d.arc([cx + 1, 486, cx + 9, 494], 10, 170, fill=col(INK), width=2)
    for tl in A["hull"]:
        put(im, tl["img"], *tl["home"])
    for s in A["sail"]:
        d.line([tuple(p) for p in s["pts"]], fill=col(s["c"]), width=s["w"])
    A["boat"] = im


# =====================================================================================
# shots
# =====================================================================================
def flock_pos(tl, ts):
    a = tl["a0"] + 0.9 * ts
    cx, cy = 640 + 220 * math.cos(0.5 * ts), 300 + 90 * math.sin(0.8 * ts)
    return cx + tl["r0"] * math.cos(a), cy + tl["r0"] * 0.45 * math.sin(a)


def shot1(t):
    ts = step(t)
    im = to_pil(A["bg_sea"])
    d = ImageDraw.Draw(im)
    sea(d, 700, ts)
    for tl in A["man"]:
        L = tl["land"]
        if ts >= L:
            settle = 1 - ramp(ts, L, L + 0.5)
            put(im, tl["img"], *np.add(tl["home"], jit(tl["k"], ts)), tl["rot"] * settle)
        else:
            q = ramp(ts, L - 1.0, L)
            fx, fy = flock_pos(tl, min(ts, L - 1.0))
            hx, hy = tl["home"]
            x = lerp(fx, hx, q)
            y = lerp(fy, hy, q) - 120 * math.sin(math.pi * q)
            bird(d, x, y, tl["sz"] * (1 - 0.4 * q), ts * 22 + tl["k"])
    for i, (r0, a0, sz, leave) in enumerate(A["extra_birds"]):
        a = a0 + 0.9 * ts
        cx, cy = 640 + 220 * math.cos(0.5 * ts), 300 + 90 * math.sin(0.8 * ts)
        x, y = cx + r0 * math.cos(a), cy + r0 * 0.45 * math.sin(a)
        if ts > leave:
            k = ts - leave
            x, y = x - 300 * k * k * (1 if i % 2 else -1), y - 260 * k * k
        bird(d, x, y, sz, ts * 22 + i)
    return [(from_pil(im), (640, 480, 1.0), 1.0, None)]


def draw_man(im, ts, off=(0, 0), skip=None):
    for tl in A["man"]:
        if ts >= tl.get("gone", 99) or (skip and skip(tl)):
            continue
        put(im, tl["img"], *np.add(np.add(tl["home"], off), jit(tl["k"], ts, 0.8)))


def open_eye(d, ts, off=(0, 0), a=1.0):
    x, y = EYE[0] + off[0], EYE[1] + off[1]
    lid = a
    d.polygon([(x - 28, y), (x - 10, y - 11 * lid), (x + 14, y - 11 * lid), (x + 28, y), (x + 12, y + 9 * lid), (x - 12, y + 9 * lid)],
              fill=col((0.80, 0.82, 0.84)))
    ell(d, x + 6, y, 9 * lid, 9 * lid, col((0.15, 0.30, 0.78)))
    ell(d, x + 6, y, 3.5 * lid, 3.5 * lid, col(INK))
    d.line([(x - 28, y), (x - 10, y - 11 * lid), (x + 14, y - 11 * lid), (x + 28, y)], fill=col((0.2, 0.18, 0.18)), width=3)


def shot2(t):
    ts = step(t)
    im = to_pil(A["bg_sea"])
    d = ImageDraw.Draw(im)
    sea(d, 700, ts)
    draw_man(im, ts)
    lid = ramp(ts, 1.0, 1.4)
    if lid > 0:
        open_eye(d, ts, a=lid)
    for tl in A["house_s"]:
        dep = tl["dep"]
        if ts < dep:
            continue
        q = ease((ts - dep) / 1.4)
        sx, sy = tl["src"]["home"]
        hx, hy = tl["home"]
        x = lerp(sx, hx, q)
        y = lerp(sy, hy, q) - tl["lift"] * math.sin(math.pi * q)
        img = tl["src"]["img"] if q < 0.5 else tl["img"]
        put(im, img, *np.add((x, y), jit(tl["k"] + 500, ts, 0.8)), tl["spin"] * (1 - q))
    glow = ramp(ts, 6.0, 7.0)
    if glow > 0:
        d.rectangle([892, 276, 968, 332], fill=col((0.95, 0.45, 0.30), 0.3 + 0.7 * glow))
    return [(from_pil(im), (640, 480, 1.0), 1.0, None)]


def shot3(t):
    ts = step(t)
    im = to_pil(A["bg_navy"])
    d = ImageDraw.Draw(im)
    im.paste(A["house_b"], (int(jit(1, ts)[0]), int(jit(2, ts)[1])), A["house_b"])
    x0, y0, x1, y1 = A["house_b_win"]
    g = ramp(ts, 0.0, 1.6)
    d.rectangle([x0, y0, x1, y1], fill=col(lerp(np.array((0.30, 0.10, 0.08)), np.array((0.98, 0.50, 0.32)), g)))
    d.line([(640, y0), (640, y1)], fill=col((0.22, 0.12, 0.10)), width=10)
    d.line([(x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)], fill=col((0.22, 0.12, 0.10)), width=10)
    for th in A["threads"]:
        n = int(clip((ts - th["t0"]) * 26, 0, len(th["pts"])))
        if n >= 2:
            d.line([tuple(p) for p in th["pts"][:n]], fill=col(th["c"]), width=th["w"], joint="curve")
    f = from_pil(im)
    win = Image.new("L", (W, H), 0)
    ImageDraw.Draw(win).rectangle([x0, y0, x1, y1], fill=255)
    f = f + np.array([0.45, 0.18, 0.10]) * (from_pil(win.filter(ImageFilter.GaussianBlur(40))) * g * 0.6)[..., None]
    return [(f, (640, 480, 1.0), 1.0, None)]


def woman_wash(im, a, off=(0, 0)):
    if a <= 0:
        return
    m = A["woman_mask"].point(lambda p: int(p * 0.28 * a))
    wash = Image.new("RGB", (W, H), col((0.80, 0.42, 0.38)))
    im.paste(wash, off, m)


def shot4(t):
    ts = step(t)
    im = to_pil(A["bg_char"])
    woman_wash(im, ramp(ts, 5.4, 7.0))
    d = ImageDraw.Draw(im)
    for s in A["woman"]:
        strand(d, s, ts)
    return [(from_pil(im), (640, 480, 1.0), 1.0, None)]


MAN_FRONT = 628
CONTACT = (640, 310)


def shot5(t):
    ts = step(t)
    im = to_pil(A["bg_deep"])
    a = ease(ts / 4.2)
    om = (lerp(-22 - 330, -22, a), -40)
    ow = (int(lerp(62 + 330, 62, a)), 20)
    fray = 4.4
    if "rank_m" not in A:
        live = [tl for tl in A["man"] if "gone" not in tl]
        live.sort(key=lambda tl: (tl["home"][0] + 8 - CONTACT[0]) ** 2 + (tl["home"][1] - 40 - CONTACT[1]) ** 2)
        for k, tl in enumerate(live):
            tl["leave"] = fray + 4.0 * k / len(live)
        ws = sorted(A["woman"], key=lambda s: ((s["pts"][:, 0].mean() + 32 - CONTACT[0]) ** 2 + (s["pts"][:, 1].mean() + 20 - CONTACT[1]) ** 2))
        for k, s in enumerate(ws):
            s["leave"] = fray + 4.0 * k / len(ws)
        A["rank_m"] = True
    remain = sum(1 for s in A["woman"] if ts < s["leave"]) / len(A["woman"])
    woman_wash(im, remain, ow)
    d = ImageDraw.Draw(im)
    for s in A["woman"]:
        if ts < s["leave"]:
            strand(d, s, ts, ow, 1.0, 1.0)
        else:
            k = ts - s["leave"]
            mid = s["pts"][len(s["pts"]) // 2] + np.array(ow)
            ang = -0.6 + 0.9 * math.sin(s["t0"] * 7)
            x = mid[0] + (200 * k + 90 * k * k) * math.cos(ang)
            y = mid[1] + (200 * k + 90 * k * k) * math.sin(ang)
            if -60 < x < W + 60 and -60 < y < H + 60:
                fish(d, x, y, 18, ang, ts * 20 + s["t0"] * 9, s["c"])
    draw_man(im, ts, om, skip=lambda tl: "gone" in tl or ts >= tl["leave"])
    eye_tl = A["man"][A["eye_tile"]]
    if ts < eye_tl.get("leave", 99):
        open_eye(d, ts, om)
    for tl in A["man"]:
        if "gone" in tl or ts < tl["leave"]:
            continue
        k = ts - tl["leave"]
        ang = math.radians(-100 - 60 * ((tl["k"] * 0.618) % 1))
        x = tl["home"][0] + om[0] + (220 * k + 120 * k * k) * math.cos(ang)
        y = tl["home"][1] + om[1] + (220 * k + 120 * k * k) * math.sin(ang)
        if -60 < x < W + 60 and -60 < y < H + 60:
            bird(d, x, y, tl.get("sz", 16), ts * 22 + tl["k"])
    return [(from_pil(im), (640, 480, 1.0), 1.0, None)]


def spiral(home, R0, a0, q):
    return (home[0] + (1 - q) * R0 * math.cos(a0 + 3.5 * math.pi * q),
            home[1] + (1 - q) * R0 * 0.7 * math.sin(a0 + 3.5 * math.pi * q))


def shot6(t):
    ts = step(t)
    im = to_pil(A["bg_navy"])
    d = ImageDraw.Draw(im)
    for tl in A["hull"]:
        q = ease((ts - tl["s"]) / 2.0)
        if q >= 1:
            put(im, tl["img"], *np.add(tl["home"], jit(tl["k"], ts, 0.8)))
        else:
            x, y = spiral(tl["home"], tl["R0"], tl["a0"], max(q, 0))
            bird(d, x, y, tl["sz"] * (1 - 0.5 * q) + 4, ts * 22 + tl["k"])
    for k, s in enumerate(A["sail"]):
        q = ease((ts - s["s"]) / 2.0)
        if q >= 1:
            fr = ramp(ts, s["s"] + 2.0, s["s"] + 2.4)
            n = max(2, int(len(s["pts"]) * fr))
            d.line([tuple(p) for p in s["pts"][:n]], fill=col(s["c"]), width=s["w"])
        else:
            mid = s["pts"][len(s["pts"]) // 2]
            x, y = spiral(mid, s["R0"], s["a0"], max(q, 0))
            x2, y2 = spiral(mid, s["R0"], s["a0"], min(1, max(q, 0) + 0.02))
            fish(d, x, y, 18, math.atan2(y2 - y, x2 - x), ts * 20 + k, s["c"])
    sea(d, 610, ts)
    return [(from_pil(im), (640, 480, 1.0), 1.0, None)]


def shot7(t):
    ts = step(t)
    hor = lerp(610, -40, ease((ts - 3.6) / 3.8))
    im = to_pil(A["bg_dusk"])
    bx = lerp(0, 520, ts / 8.0)
    by = 6 * math.sin(ts * 2.0)
    im.paste(A["boat"], (int(bx), int(by)), A["boat"])
    sea_m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sea_m).rectangle([0, hor, W, H], fill=255)
    im.paste(to_pil(A["sea_fill"]), (0, 0), sea_m.filter(ImageFilter.GaussianBlur(1)))
    d = ImageDraw.Draw(im)
    sea(d, max(hor, -40), ts)
    w = 1 - ramp(t, 7.0, 8.0)
    return [(from_pil(im), (640, 480, 1.0), w, None)]


SHOTS = [(shot1, 0, 7, 0), (shot2, 7, 7, 0), (shot3, 14, 6, 1.0), (shot4, 20, 8, 1.0), (shot5, 28, 9, 1.0),
         (shot6, 37, 7, 1.0), (shot7, 44, 8, 1.0)]
TOTAL = 52 * FPS

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[2:]]
    engine.run(SHOTS, TOTAL, sys.argv[1], frames,
               [build_backgrounds, build_man, build_house, build_woman, build_boat], "pastel", boil=2)
