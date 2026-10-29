"""STONE, MOTH, HORSE, RAIN — silent stop-motion vignettes in the style of Sabin Balasa.

No plot: small transformations, each flowing into the next.
Museum pieces (CC0 / public domain) are repainted into the Balasa palette by cutouts.py.
Usage: python3 metamorph.py <frames_dir> [frame numbers...]
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import cutouts as cu
import engine
from birds import jit, sea, step, texture
from engine import A
from lib import FPS, H, W, Y, clip, col, ease, from_pil, lerp, ramp, stipple, to_pil, vgrad, vnoise

RNG = np.random.default_rng
RAW = os.path.join(os.path.dirname(cu.ASSETS), "raw")
HOR = 760
AMBER = (0.98, 0.72, 0.30)


# =====================================================================================
# helpers
# =====================================================================================
def comp(im, piece, x, y):
    """Alpha-composite piece with its top-left at (x, y); clips at the frame edges."""
    x, y = int(round(x)), int(round(y))
    l, t = max(0, -x), max(0, -y)
    r, b = min(piece.width, im.width - x), min(piece.height, im.height - y)
    if r > l and b > t:
        im.alpha_composite(piece.crop((l, t, r, b)), (x + l, y + t))


def comp_c(im, piece, cx, cy):
    comp(im, piece, cx - piece.width / 2, cy - piece.height / 2)


def xform(piece, sx=1.0, sy=1.0, rot=0.0):
    w, h = max(2, int(piece.width * sx)), max(2, int(piece.height * sy))
    p = piece.resize((w, h), Image.BICUBIC) if (w, h) != piece.size else piece
    return p.rotate(rot, resample=Image.BICUBIC, expand=True) if abs(rot) > 0.3 else p


def fade(piece, a):
    if a >= 0.999:
        return piece
    p = piece.copy()
    p.putalpha(p.getchannel("A").point(lambda v: int(v * max(0.0, a))))
    return p


def arr(piece):
    return np.asarray(piece, np.float32) / 255


def img(a):
    return Image.fromarray((clip(a) * 255).astype(np.uint8), "RGBA")


def ell(d, cx, cy, rx, ry, fill):
    d.ellipse([cx - abs(rx), cy - abs(ry), cx + abs(rx), cy + abs(ry)], fill=fill)


def walk(r, p, heading, n, stepl, turn, bias=None, pull=0.0):
    pts = [np.array(p, np.float32)]
    h = heading
    for _ in range(n):
        h += r.normal(0, turn)
        if bias is not None:
            want = math.atan2(bias[1] - pts[-1][1], bias[0] - pts[-1][0])
            h += pull * math.atan2(math.sin(want - h), math.cos(want - h))
        pts.append(pts[-1] + stepl * np.array([math.cos(h), math.sin(h)], np.float32))
    return np.array(pts)


def polyline(d, pts, frac, c, w):
    n = int(len(pts) * clip(frac))
    if n >= 2:
        d.line([tuple(p) for p in pts[:n].tolist()], fill=c, width=w, joint="curve")


def ease_inv(y):
    lo, hi = 0.0, 1.0
    for _ in range(30):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if ease(m) < y else (lo, m)
    return lo


def night():
    return to_pil(A["night"]).convert("RGBA")


def void(stars=True):
    return to_pil(A["void_stars" if stars else "void"]).convert("RGBA")


def sea_front(im, ts, hor=HOR):
    """Water in front of everything below the horizon."""
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle([0, hor, W, H], fill=255)
    im.paste(A["sea_fill"], (0, 0), m)
    sea(ImageDraw.Draw(im), hor, ts)


# =====================================================================================
# builds
# =====================================================================================
def build_world():
    base = clip(vgrad((0.035, 0.045, 0.10), (0.09, 0.10, 0.17)) * texture((1, 1, 1), 3, angle=-0.4))
    A["void"] = base
    s = to_pil(base)
    d = ImageDraw.Draw(s)
    r = RNG(4)
    for _ in range(260):
        x, y, rr = r.uniform(0, W), r.uniform(0, H), r.uniform(0.7, 2.0)
        ell(d, x, y, rr, rr, col((0.85, 0.85, 0.80), r.uniform(0.4, 1.0)))
    A["void_stars"] = from_pil(s)
    sky = vgrad((0.05, 0.06, 0.13), (0.14, 0.15, 0.24), 0, HOR) * texture((1, 1, 1), 5, angle=-0.3, length=22)
    seab = texture((0.05, 0.07, 0.14), 6, angle=0.0, jit_=0.05, length=30, ss=0.2)
    A["night"] = clip(np.where((Y < HOR)[..., None], sky, seab))
    A["sea_fill"] = to_pil(seab)


def build_v1():
    head = cu.tone(cu.load("170075"), seed=11)
    head = cu.scaled(head.crop((0, 0, head.width, int(head.height * 0.62))), 600)
    A["khmer"] = head
    h, w = head.height, head.width
    a = arr(head)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    A["k_d"] = clip(np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * 0.55) / (h * 0.55)) ** 2))
    A["k_n"] = [vnoise(20 + i, 22 if i in (0, 3) else 7, 3, w, h) for i in range(4)]
    score = 0.45 * A["k_n"][1] + 0.55 * (1 - A["k_d"])
    A["k_score"] = score
    r = RNG(12)
    ys, xs = np.nonzero(a[..., 3] > 0.6)
    pick = r.choice(len(xs), 420, replace=False)
    grains = []
    for i in pick:
        s = score[ys[i], xs[i]]
        tf = 3.0 + 3.6 * ease_inv(clip((s + 0.1) / 1.15))
        grains.append((xs[i], ys[i], tf, tuple(a[ys[i], xs[i], :3]), r.uniform(1.5, 3.5)))
    A["grains"] = grains
    # the river stone that remains
    sw, sh_ = 330, 236
    yy, xx = np.mgrid[0:sh_, 0:sw].astype(np.float32)
    u, v = (xx - sw / 2) / 150, (yy - sh_ / 2) / 104
    dd = np.sqrt(u * u + v * v)
    al = clip((1 - dd) * 60 + 0.5)
    shade = 0.62 + 0.45 * clip(1 - ((u + 0.35) ** 2 + (v + 0.45) ** 2) / 1.4) - 0.2 * dd ** 3
    rgb = np.array([0.50, 0.54, 0.60]) * shade[..., None] * (0.88 + 0.24 * vnoise(13, 5, 3, sw, sh_))[..., None]
    rgb += 0.12 * stipple(14, 0.05, sw, sh_)[..., None]
    A["stone"] = img(np.dstack([clip(rgb), al]))


STONE_C = (640, 520)


def build_v2():
    r = RNG(21)
    st = A["stone"]
    w, h = st.size
    main = [(w / 2 + r.normal(0, 3), -6)]
    while main[-1][1] < h + 6:
        main.append((main[-1][0] + r.uniform(-11, 11), main[-1][1] + 12))
    main = np.array(main, np.float32)
    A["crack_main"] = main
    branches = []
    for k in range(9):
        i = int(r.integers(3, len(main) - 3))
        side = r.choice([-1, 1])
        pts = walk(r, main[i], math.pi / 2 + side * r.uniform(0.6, 1.3), int(r.integers(5, 11)), 10, 0.35)
        branches.append((pts, r.uniform(0.9, 2.4)))
    A["crack_br"] = branches
    for name, poly in (("half_l", [(0, -10)] + [tuple(p) for p in main] + [(0, h + 10)]),
                       ("half_r", [(w, -10)] + [tuple(p) for p in main] + [(w, h + 10)])):
        m = Image.new("L", st.size, 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        half = st.copy()
        half.putalpha(Image.fromarray(np.minimum(np.asarray(st.getchannel("A")), np.asarray(m))))
        dd = ImageDraw.Draw(half)
        dd.line([tuple(p) for p in main.tolist()], fill=col((0.03, 0.03, 0.05)), width=4)
        A[name] = half
    moth = cu.load("8785324")
    moth = moth.crop((int(moth.width * 0.06), 0, moth.width, int(moth.height * 0.82)))
    moth = moth.crop(moth.getbbox())
    A["moth"] = cu.scaled(cu.tone(moth, lo=(0.05, 0.05, 0.10), hi=(0.92, 0.88, 0.80), keep=0.6, seed=22), 400)


def build_v4():
    mp = Image.open(os.path.join(RAW, "wc_88717005.jpg")).convert("RGBA")
    s = max(W / mp.width, H / mp.height)
    mp = mp.resize((int(mp.width * s) + 1, int(mp.height * s) + 1), Image.LANCZOS)
    mp = mp.crop(((mp.width - W) // 2, (mp.height - H) // 2, (mp.width - W) // 2 + W, (mp.height - H) // 2 + H))
    A["map"] = cu.tone(mp, lo=(0.04, 0.05, 0.11), hi=(0.44, 0.42, 0.36), keep=0.15, torn=False, seed=40)
    # a sleeping mask-face with a cap of dark hair
    w, h = 280, 380
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, rx, ry = w / 2, h / 2 - 8, 100, 150
    v = (yy - cy) / ry
    rxx = rx * (1 - 0.25 * clip(v) ** 2)
    dd = np.sqrt(((xx - cx) / rxx) ** 2 + v ** 2)
    al = clip((1 - dd) * 70 + 0.5)
    shade = 0.80 + 0.25 * (1 - dd ** 2) - 0.12 * clip(v)
    rgb = np.array([0.84, 0.81, 0.75]) * shade[..., None] * (0.9 + 0.2 * vnoise(41, 4, 3, w, h))[..., None]
    rgb *= (1 - 0.25 * stipple(42, 0.07, w, h))[..., None]
    ink = Image.new("L", (w, h), 0)
    di = ImageDraw.Draw(ink)
    for m in (-1, 1):
        ex, ey = cx + m * 38, cy - 18
        di.arc([ex - 28, ey - 10, ex + 28, ey + 12], 15, 165, fill=210, width=3)
        di.arc([ex - 32, ey - 38, ex + 32, ey - 2], 205, 335, fill=90, width=3)
    di.line([(cx + 2, cy - 16), (cx - 4, cy + 52)], fill=110, width=3)
    di.arc([cx - 16, cy + 44, cx + 12, cy + 60], 20, 160, fill=110, width=3)
    rgb *= (1 - 0.65 * from_pil(ink.filter(ImageFilter.GaussianBlur(0.8))))[..., None]
    lips = Image.new("L", (w, h), 0)
    ImageDraw.Draw(lips).ellipse([cx - 20, cy + 84, cx + 20, cy + 94], fill=255)
    lm = from_pil(lips.filter(ImageFilter.GaussianBlur(1.2)))[..., None]
    rgb = rgb * (1 - lm) + np.array([0.62, 0.30, 0.28]) * lm
    hair = clip((cy - ry * 0.45 - yy + 18 * np.abs((xx - cx) / rx)) / 6) * (al > 0)
    rgb = rgb * (1 - hair[..., None]) + np.array([0.10, 0.08, 0.08]) * hair[..., None]
    A["face"] = xform(img(np.dstack([clip(rgb), al])), 1.3, 1.3)
    r = RNG(43)
    fcx, fcy = 640, 320
    strands, towns = [], []
    for k in range(84):
        a = math.radians(lerp(198, 342, k / 83) + r.normal(0, 2))
        p = (fcx + 135 * math.cos(a), fcy - 10 + 200 * math.sin(a))
        side = -1 if math.cos(a) < 0 else 1
        target = (fcx + side * r.uniform(300, 900), r.uniform(700, 1150))
        pts = walk(r, p, a, 95, 11, 0.12, bias=target, pull=0.06)
        strands.append(dict(pts=pts, w=int(r.integers(2, 4)), g=r.uniform(0.8, 1.2)))
    for k in range(40):
        s_ = strands[int(r.integers(0, len(strands)))]
        i = int(r.uniform(0.35, 0.95) * len(s_["pts"]))
        x, y = s_["pts"][i]
        if not (20 < x < W - 20 and 360 < y < H - 20):
            continue
        dots = [(x + r.normal(0, 7), y + r.normal(0, 5)) for _ in range(int(r.integers(3, 6)))]
        towns.append(dict(p=(float(x), float(y)), dots=dots, t=r.uniform(4.3, 6.4)))
    A["strands"], A["towns"] = strands, towns


HORSE_C = (640, 500)


def build_v5():
    horse = cu.scaled(cu.tone(cu.load("84457"), lo=(0.05, 0.06, 0.13), hi=(0.88, 0.88, 0.84), keep=0.25, seed=50), 520)
    A["horse"] = horse
    a = np.asarray(horse.getchannel("A"))[::4, ::4] > 128
    er = a.copy()
    er[1:-1, 1:-1] = a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:]
    ys, xs = np.nonzero(a & ~er)
    pts = np.stack([xs * 4, ys * 4], 1).astype(np.float32)
    n = len(A["towns"])
    chosen = [int(np.argmin(pts[:, 1]))]
    dmin = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[i], axis=1))
    stars = pts[chosen] + np.array([HORSE_C[0] - horse.width / 2, HORSE_C[1] - horse.height / 2])
    order, left = [0], set(range(1, n))
    while left:
        last = stars[order[-1]]
        j = min(left, key=lambda k: np.linalg.norm(stars[k] - last))
        order.append(j)
        left.remove(j)
    A["stars"], A["star_order"] = stars, order
    r = RNG(51)
    for k, tw in enumerate(A["towns"]):
        tw["star"] = tuple(stars[k])
        tw["lift"] = r.uniform(0.2, 1.4)
    hp = arr(horse)
    h, w = horse.height, horse.width
    xx = np.mgrid[0:h, 0:w][1].astype(np.float32)
    A["h_score"] = 0.75 * (xx / w) + 0.25 * vnoise(52, 8, 3, w, h)
    ys, xs = np.nonzero(hp[..., 3] > 0.6)
    pick = r.choice(len(xs), 500, replace=False)
    foam = []
    for i in pick:
        s = A["h_score"][ys[i], xs[i]]
        tf = 1.5 + 4.5 * ease_inv(clip((s + 0.05) / 1.15))
        foam.append((xs[i], ys[i], tf, r.uniform(-170, -50), r.uniform(-220, -70)))
    A["foam"] = foam
    waves = []
    for k in range(26):
        top = 170 + 9 * k
        pts = []
        for x in np.linspace(1320, 480 + 3 * k, 60):
            q = (1320 - x) / (840 - 3 * k)
            pts.append((x, HOR + 4 - top * q ** 1.6 + 5 * math.sin(x * 0.03 + k) * (1 - q)))
        r0 = 34 + 2.4 * k
        cx, cy = pts[-1][0], pts[-1][1] + r0
        for aa in np.linspace(-math.pi / 2, -math.pi / 2 - 3.6, 26)[1:]:
            rr = r0 * (1 - 0.5 * (-math.pi / 2 - aa) / 3.6)
            pts.append((cx + rr * math.cos(aa), cy + rr * math.sin(aa)))
        waves.append(np.array(pts, np.float32))
    A["waves"] = waves


POSE_DY = [0, -9, -3, -12]
POSE_ROT = [0, 1.6, 0, -1.6]


def horse_pose(ts, run):
    k = int(round(ts * 12)) % 4 if run else 0
    return POSE_DY[k], POSE_ROT[k]


def build_v7():
    hand = cu.scaled(cu.tone(cu.load("76324410"), keep=0.35, seed=70), 420)
    A["hand_l"] = hand
    A["hand_r"] = hand.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(hand.getchannel("A")) > 128
    tips = []
    for fx in (0.22, 0.42, 0.62, 0.80):
        c = int(fx * hand.width)
        col_ = a[:, max(0, c - 6):c + 6].any(1)
        ys = np.nonzero(col_)[0]
        if len(ys):
            tips.append((c - hand.width / 2, ys[0] - hand.height))
    A["tips"] = tips
    r = RNG(71)
    branches, flowers = [], []
    for side, hx in ((-1, 430), (1, 850)):
        for (tx, ty) in tips:
            x0 = hx + (tx if side < 0 else -tx)
            p0 = (x0, 820 + ty)
            target = (640 - side * r.uniform(40, 220), r.uniform(160, 330))
            trunk = walk(r, p0, -math.pi / 2, 34, 11, 0.12, bias=target, pull=0.10)
            branches.append(dict(pts=trunk, d=0, t0=2.2 + r.uniform(0, 0.4)))
            for j in range(3):
                i = int(r.integers(8, len(trunk) - 4))
                sub = walk(r, trunk[i], -math.pi / 2 + r.normal(0, 0.9), int(r.integers(8, 16)), 9, 0.25)
                branches.append(dict(pts=sub, d=1, t0=3.0 + r.uniform(0, 0.8)))
                flowers.append(dict(p=tuple(sub[-1]), t=r.uniform(4.8, 6.4), r=r.uniform(14, 24), a=r.uniform(0, 6)))
            flowers.append(dict(p=tuple(trunk[-1]), t=r.uniform(4.8, 6.4), r=r.uniform(18, 26), a=r.uniform(0, 6)))
    A["branches"], A["flowers"] = branches, flowers
    for fl in flowers:
        fl["petals"] = [dict(a=fl["a"] + k * 2 * math.pi / 5, vx=r.normal(0, 25), t=r.uniform(0.3, 3.0), sw=r.uniform(0, 6))
                        for k in range(5)]
    A["gandhara"] = cu.scaled(cu.tone(cu.load("152487"), seed=80), 470)
    A["rain"] = [(r.uniform(0, W), r.uniform(0, 1), r.uniform(2.0, 7.0), r.uniform(26, 50)) for _ in range(260)]


# =====================================================================================
# vignettes
# =====================================================================================
def v1(t):
    """Erosion: moss blooms over a Khmer stone head, then it wears away to a river stone."""
    ts = step(t)
    im = void(False)
    sa = ramp(ts, 2.8, 3.4)
    if sa > 0:
        comp_c(im, fade(A["stone"], sa), *STONE_C)
    a = arr(A["khmer"]).copy()
    n = A["k_n"]
    g = 0.42 * ramp(ts, 0.4, 3.0)
    on = (a[..., 3] > 0.3).astype(np.float32)
    moss = (n[0] < g).astype(np.float32) * on
    mc = np.array([0.22, 0.30, 0.14]) + np.array([0.28, 0.22, 0.14]) * n[2][..., None]
    a[..., :3] = a[..., :3] * (1 - 0.6 * moss[..., None]) + mc * 0.6 * moss[..., None]
    lich = (n[3] < g * 0.22).astype(np.float32) * on
    a[..., :3] = a[..., :3] * (1 - lich[..., None]) + np.array([0.78, 0.74, 0.48]) * lich[..., None]
    e = -0.1 + 1.15 * ease((ts - 3.0) / 3.6)
    a[..., 3] *= (A["k_score"] > e)
    head = img(a)
    hx, hy = 640 - head.width / 2, 470 - head.height / 2
    comp(im, head, hx, hy)
    d = ImageDraw.Draw(im)
    for gx, gy, tf, c, sz in A["grains"]:
        if ts > tf:
            k = ts - tf
            y = hy + gy + 0.5 * 1400 * k * k
            if y < H:
                ell(d, hx + gx + 20 * k, y, sz, sz, col(c))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def v2(t):
    """The crack: the stone splits and a moth unfolds out of it."""
    ts = step(t)
    im = void(False)
    q = ease((ts - 3.0) / 1.2)
    st = A["stone"]
    x0, y0 = STONE_C[0] - st.width / 2, STONE_C[1] - st.height / 2
    if q <= 0:
        comp(im, st, x0, y0)
        d = ImageDraw.Draw(im)
        main = A["crack_main"] + np.array([x0, y0])
        fr = ramp(ts, 0.3, 1.8)
        polyline(d, main, fr, col((0.03, 0.03, 0.05)), 4)
        polyline(d, main + np.array([1.5, 0]), fr, col((0.78, 0.80, 0.84)), 1)
        for pts, t0 in A["crack_br"]:
            polyline(d, pts + np.array([x0, y0]), ramp(ts, t0, t0 + 0.8), col((0.03, 0.03, 0.05)), 3)
    else:
        for name, sgn in (("half_l", -1), ("half_r", 1)):
            p = xform(A[name], rot=-sgn * 14 * q)
            comp_c(im, p, STONE_C[0] + sgn * 210 * q, STONE_C[1] + 30 * q)
    if ts > 3.3:
        s = lerp(0.12, 0.62, ease((ts - 3.3) / 3.0))
        wx = lerp(0.08, 1.0, ease((ts - 3.8) / 2.2)) * [1, 0.9, 0.8, 0.9][int(round(ts * 12)) % 4]
        m = xform(A["moth"], s * wx, s)
        comp_c(im, m, 640, lerp(520, 430, ease((ts - 3.3) / 3.0)))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def v3(t):
    """The moth circles a moon, growing as the moon shrinks into a dot on its wing."""
    ts = step(t)
    im = void(True)
    th = -math.pi / 2 + 2 * math.pi * 1.75 * ease(ts / 6.2)
    R = lerp(360, 0, ramp(ts, 4.6, 6.6))
    mx, my = 640 + R * math.cos(th), 430 + 0.5 * R * math.sin(th)
    s = lerp(0.3, 2.7, ease(ts / 6.8))
    wx = [1, 0.75, 0.5, 0.75][int(round(ts * 12)) % 4]
    m = xform(A["moth"], s * wx, s, 10 * math.cos(th) * (1 - ramp(ts, 4.6, 6.0)))
    mr = lerp(80, 7, ease(ts / 6.0))
    k = ramp(ts, 4.6, 6.4)
    eye = (mx - 0.27 * A["moth"].width * s * wx, my - 0.12 * A["moth"].height * s)
    cx, cy = lerp(640, eye[0], k), lerp(430, eye[1], k)

    def moon():
        d = ImageDraw.Draw(im)
        ell(d, cx, cy, mr * 1.6, mr * 1.6, (220, 220, 205, 40))
        ell(d, cx, cy, mr, mr, col((0.88, 0.86, 0.78)))
        ell(d, cx - mr * 0.25, cy - mr * 0.2, mr * 0.3, mr * 0.25, col((0.78, 0.76, 0.70)))

    behind = math.sin(th) < 0 and k < 0.01
    if not behind:
        moon()
    comp_c(im, m, mx, my)
    if behind or k > 0:
        moon()
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def v4(t):
    """Hair into rivers: a sleeper's hair grows into the rivers of an old map; towns light up."""
    ts = step(t)
    im = void(False)
    comp(im, fade(A["map"], ramp(ts, 2.0, 5.0)), 0, 0)
    d = ImageDraw.Draw(im)
    fr = lerp(0.25, 1.0, ease(ts / 4.5))
    k = ramp(ts, 2.2, 5.0)
    for s in A["strands"]:
        c = lerp(np.array((0.10, 0.08, 0.08)) * s["g"], np.array((0.42, 0.56, 0.72)) * s["g"], k)
        polyline(d, s["pts"], fr, col(c), s["w"])
    comp_c(im, fade(A["face"], 1 - ramp(ts, 3.6, 5.4)), 640, 320)
    glow = Image.new("L", (W, H), 0)
    dg = ImageDraw.Draw(glow)
    for tw in A["towns"]:
        if ts >= tw["t"]:
            for x, y in tw["dots"]:
                ell(d, x, y, 2.6, 2.6, col(AMBER))
                ell(dg, x, y, 6, 6, 255)
    f = from_pil(im.convert("RGB"))
    f = f + np.array(AMBER) * (from_pil(glow.filter(ImageFilter.GaussianBlur(9))) * 0.6)[..., None]
    return [(f, (640, 480, 1.0), 1.0, None)]


def v5(t):
    """Lanterns: the town lights lift into a constellation that becomes a running horse."""
    ts = step(t)
    im = void(True)
    comp(im, fade(A["map"], 1 - ramp(ts, 0.0, 3.0)), 0, 0)
    d = ImageDraw.Draw(im)
    for s in A["strands"]:
        c = np.array((0.42, 0.56, 0.72)) * s["g"]
        polyline(d, s["pts"], 1.0, col(c, 1 - ramp(ts, 0.0, 2.5)), s["w"]) if ts < 2.5 else None
    stars = A["stars"]
    ln = ramp(ts, 3.4, 4.6)
    order = A["star_order"]
    nseg = int(ln * (len(order) - 1))
    for i in range(nseg):
        a, b = stars[order[i]], stars[order[i + 1]]
        d.line([tuple(a), tuple(b)], fill=col((0.70, 0.74, 0.86), 0.8), width=2)
    ha = ramp(ts, 4.4, 5.6)
    if ha > 0:
        dy, rot = horse_pose(ts, ts > 5.0)
        comp_c(im, fade(xform(A["horse"], rot=rot), ha), HORSE_C[0], HORSE_C[1] + dy)
    glow = Image.new("L", (W, H), 0)
    dg = ImageDraw.Draw(glow)
    for tw in A["towns"]:
        q = ease((ts - tw["lift"]) / 2.2)
        x = lerp(tw["p"][0], tw["star"][0], q)
        y = lerp(tw["p"][1], tw["star"][1], q) - 120 * math.sin(math.pi * q)
        c = lerp(np.array(AMBER), np.array((0.95, 0.93, 0.80)), q)
        tw_ = 0.8 + 0.2 * math.sin(ts * 9 + x)
        ell(d, x, y, 3.2 * tw_, 3.2 * tw_, col(c))
        ell(dg, x, y, 8, 8, 255)
    f = from_pil(im.convert("RGB"))
    f = f + np.array((0.9, 0.75, 0.45)) * (from_pil(glow.filter(ImageFilter.GaussianBlur(10))) * 0.5)[..., None]
    return [(f, (640, 480, 1.0), 1.0, None)]


def v6(t):
    """Horse into wave: the horse dissolves into foam while a hatched wave curls up."""
    ts = step(t)
    im = night()
    sea(ImageDraw.Draw(im), HOR, ts)
    h = arr(A["horse"]).copy()
    e = -0.05 + 1.15 * ease((ts - 1.5) / 4.5)
    h[..., 3] *= (A["h_score"] > e) * (1 - 0.0)
    dy, rot = horse_pose(ts, True)
    hp = img(h)
    hx, hy = HORSE_C[0] - hp.width / 2, HORSE_C[1] - hp.height / 2 + dy
    comp(im, xform(hp, rot=rot), hx, hy)
    d = ImageDraw.Draw(im)
    fr = ramp(ts, 1.3, 5.8)
    for k, pts in enumerate(A["waves"]):
        c = lerp(np.array((0.36, 0.46, 0.62)), np.array((0.86, 0.88, 0.92)), k / 25)
        polyline(d, pts, fr * (1.05 - 0.002 * k), col(c), 3)
    for fx, fy, tf, vx, vy in A["foam"]:
        if ts > tf:
            k = ts - tf
            if k < 1.4:
                ell(d, hx + fx + vx * k, hy + fy + vy * k + 120 * k * k, 2.5, 2.5, col((0.92, 0.94, 0.96), 1 - k / 1.4))
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def draw_branches(d, ts, a=1.0):
    if a < 0.03:
        return
    c = col(lerp(np.array((0.09, 0.10, 0.18)), np.array((0.20, 0.16, 0.14)), a))
    for br in A["branches"]:
        w = 6 if br["d"] == 0 else 3
        polyline(d, br["pts"], ramp(ts, br["t0"], br["t0"] + 2.2), c, w)


def flower(d, x, y, r, a0, k=1.0, skip=()):
    for j in range(5):
        if j in skip:
            continue
        a = a0 + j * 2 * math.pi / 5
        px, py = x + math.cos(a) * r * 0.55, y + math.sin(a) * r * 0.55
        ell(d, px, py, r * 0.5, r * 0.5, col((0.92, 0.40, 0.32) if j % 2 else (0.98, 0.60, 0.50), k))
    ell(d, x, y, r * 0.22, r * 0.22, col((0.96, 0.86, 0.50), k))


def v7(t):
    """Hands: two stone hands rise; their fingers grow into branches that bloom coral."""
    ts = step(t)
    im = night()
    rise = ease(ts / 2.4)
    by = lerp(1240, 820, rise)
    comp(im, A["hand_l"], 430 - A["hand_l"].width / 2, by - A["hand_l"].height)
    comp(im, A["hand_r"], 850 - A["hand_r"].width / 2, by - A["hand_r"].height)
    sea_front(im, ts)
    d = ImageDraw.Draw(im)
    draw_branches(d, ts)
    for fl in A["flowers"]:
        k = ease((ts - fl["t"]) / 0.8)
        if k > 0:
            flower(d, *fl["p"], fl["r"] * k, fl["a"])
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), 1.0, None)]


def v8(t):
    """Rain: petals fall into rain; the Gandhara head sinks through it, smiling."""
    ts = step(t)
    im = night()
    sink = ease((ts - 0.5) / 2.5)
    by = lerp(820, 1240, sink)
    comp(im, A["hand_l"], 430 - A["hand_l"].width / 2, by - A["hand_l"].height)
    comp(im, A["hand_r"], 850 - A["hand_r"].width / 2, by - A["hand_r"].height)
    ga = ramp(ts, 1.2, 2.6)
    if ga > 0:
        gy = lerp(560, 1000, ease((ts - 3.4) / 3.4))
        comp_c(im, fade(A["gandhara"], ga), 640, gy)
    sea_front(im, ts)
    d = ImageDraw.Draw(im)
    draw_branches(d, 99, 1 - ramp(ts, 0.4, 2.6))
    for fl in A["flowers"]:
        gone = [j for j, p in enumerate(fl["petals"]) if ts > p["t"]]
        if len(gone) < 5:
            flower(d, *fl["p"], fl["r"], fl["a"], skip=gone)
        for j, p in enumerate(fl["petals"]):
            if ts <= p["t"]:
                continue
            k = ts - p["t"]
            x = fl["p"][0] + p["vx"] * k + 14 * math.sin(k * 4 + p["sw"])
            y = fl["p"][1] + 0.5 * 520 * k * k
            if y > HOR + 30:
                continue
            stretch = clip((y - 420) / 260)
            c = lerp(np.array((0.95, 0.50, 0.42)), np.array((0.70, 0.78, 0.90)), stretch)
            rr = fl["r"] * 0.45 * (1 - 0.6 * stretch)
            d.line([(x, y - rr - 30 * stretch), (x, y + rr)], fill=col(c), width=max(2, int(rr * (1 - stretch) + 2)))
    rn = ramp(ts, 1.6, 4.0)
    for i, (x, ph, sp, ln) in enumerate(A["rain"]):
        if i / len(A["rain"]) > rn:
            continue
        y = ((ph + ts * sp * 0.35) % 1.0) * (HOR + 60) - 40
        d.line([(x, y), (x - 3, y + ln)], fill=col((0.66, 0.74, 0.88), 0.85), width=2)
        if y + ln > HOR - 4 and i % 3 == 0:
            ell(d, x, HOR + 8 + (i % 7) * 20, 14, 3, col((0.55, 0.64, 0.80), 0.6))
    w = 1 - ramp(t, 5.4, 7.0)
    return [(from_pil(im.convert("RGB")), (640, 480, 1.0), w, None)]


SHOTS = [(v1, 0, 7, 0), (v2, 7, 7, 1.0), (v3, 14, 7, 1.0), (v4, 21, 7, 1.0), (v5, 28, 7, 0),
         (v6, 35, 7, 0), (v7, 42, 7, 1.0), (v8, 49, 7, 0)]
TOTAL = 56 * FPS

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[2:]]
    engine.run(SHOTS, TOTAL, sys.argv[1], frames,
               [build_world, build_v1, build_v2, build_v4, build_v5, build_v7], "pastel", boil=2)
