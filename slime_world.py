"""SLIM-E WORLD, BALASA CUT — seamless 15 s homepage loop of his clay world, repainted as Balasa stop-motion.

Takes a still frame of the existing world film, cuts every character/cloud/sun ray into its own piece,
rebuilds the sky and ground as pastel on toned paper, and animates everything on twos in a perfect loop.
One cloud breaks into paper birds that circle the sky and land back as the cloud.

Usage: python3 slime_world.py <world_frame.png> <out_dir> <coin_cx> <coin_cy> [rembg_cut.png]
"""
import json
import math
import os
import sys
from collections import deque
from multiprocessing import get_context

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import relief as rl
from birds import bird
from lib import clip, col, ease, stipple, stroke_mod, vnoise

FPS, LOOP = 12, 15.0
BS = 20
NF = int(FPS * LOOP)
G = {}


def tau(t, k=1, ph=0.0):
    return math.sin(2 * math.pi * k * t / LOOP + ph)


# ------------------------------------------------------------------ extraction
def extract(frame_path, cut_path=None):
    src = np.asarray(Image.open(frame_path).convert("RGB"), np.float32) / 255
    H, W = src.shape[:2]
    lum = src.mean(-1)
    rows = np.median(lum, axis=1)
    sky_end = int(np.argmax(rows < 0.45)) if (rows < 0.45).any() else H // 2
    med = np.median(src, axis=1, keepdims=True)
    diff = np.abs(src - med).max(-1)
    yy = np.arange(H)[:, None]
    key = (diff > 0.16) & (yy > sky_end - 0.25 * H) & (yy < H * 0.86)
    sat = src.max(-1) - src.min(-1)
    warm = (src[..., 0] > src[..., 2] + 0.06) & (yy < sky_end)          # cream clouds / pink sun vs blue sky
    key |= warm & (sat > 0.10)
    if cut_path and os.path.exists(cut_path):
        cut = np.asarray(Image.open(cut_path), np.float32) / 255
        if cut.shape[:2] == (H, W):
            key |= cut[..., 3] > 0.05
    m = Image.fromarray((key * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MedianFilter(5))
    mk = np.asarray(m) > 128
    lab = np.zeros((H, W), np.int32)
    n = 0
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            if mk[y, x] and not lab[y, x]:
                n += 1
                q = deque([(y, x)])
                lab[y, x] = n
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = cy + dy, cx + dx
                        if 0 <= a < H and 0 <= b < W and mk[a, b] and not lab[a, b]:
                            lab[a, b] = n
                            q.append((a, b))
    pieces = []
    r = np.random.default_rng(3)
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 0.0004 * H * W:
            continue
        x0, x1, y0, y1 = xs.min() - 3, xs.max() + 4, ys.min() - 3, ys.max() + 4
        x0, y0 = max(0, x0), max(0, y0)
        sub = (lab[y0:y1, x0:x1] == i).astype(np.float32)
        h_, w_ = sub.shape
        nz = np.asarray(Image.fromarray((r.random((max(2, h_ // 5), max(2, w_ // 5))) * 255).astype(np.uint8))
                        .resize((w_, h_), Image.BICUBIC), np.float32) / 255
        soft = rl.blur(sub, 1.5)
        al = clip((soft - 0.5 + 0.22 * (nz - 0.5)) * 5)                      # torn paper edge
        rgb = src[y0:y1, x0:x1] * (0.94 + 0.12 * r.random((h_, w_, 1)).astype(np.float32))   # pastel grain
        img = Image.fromarray((np.dstack([clip(rgb), al]) * 255).astype(np.uint8), "RGBA")
        pieces.append(dict(img=img, x=int(x0), y=int(y0), w=w_, h=h_, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                           bottom=int(y1), area=int(sub.sum()), mean=tuple(np.median(src[ys, xs], 0))))
    # background with every piece removed (normalised blur fill)
    hole = rl.blur(mk.astype(np.float32), 3) > 0.02
    keep = (~hole).astype(np.float32)
    acc = np.dstack([rl.blur(src[..., c] * keep, 40) for c in range(3)])
    wgt = rl.blur(keep, 40)[..., None] + 1e-4
    fill = acc / wgt
    acc2 = np.dstack([rl.blur(src[..., c] * keep, 10) for c in range(3)])
    fill2 = acc2 / (rl.blur(keep, 10)[..., None] + 1e-4)
    near = clip(rl.blur(keep, 10) * 3)[..., None]
    fill = fill2 * near + fill * (1 - near)
    bg = np.where(hole[..., None], fill, src)
    return dict(bg=bg.astype(np.float32), pieces=pieces, W=W, H=H, horizon=sky_end)


# ------------------------------------------------------------------ build
def classify(P):
    W, H, hz = P["W"], P["H"], P["horizon"]
    sky = [p for p in P["pieces"] if p["bottom"] < hz + 8]
    ground = [p for p in P["pieces"] if p["bottom"] >= hz + 8]

    def pinkness(p):
        r_, g_, b_ = p["mean"]
        return r_ - g_
    sun = max(sky, key=lambda p: pinkness(p) * (1 if 0.6 < p["w"] / max(1, p["h"]) < 1.6 else 0.2) * min(p["area"], 9e4))
    for p in sky:
        p["kind"] = "cloud"
        if p is sun:
            p["kind"] = "sun"
        elif math.hypot(p["cx"] - sun["cx"], p["cy"] - sun["cy"]) < 1.9 * max(sun["w"], sun["h"]) and pinkness(p) > 0.02 \
                and p["area"] < 0.35 * sun["area"]:
            p["kind"] = "ray"
    clouds = [p for p in sky if p["kind"] == "cloud"]
    flock = max(clouds, key=lambda p: p["area"] * (1.3 if p["cx"] < W / 2 else 1.0))
    flock["kind"] = "flock"
    for p in ground:
        p["kind"] = "friend"
    merge_touching(P, [p for p in P["pieces"] if p["kind"] == "friend"])
    return sun


def merge_touching(P, fr):
    """Pieces that touch (a head on its stem) are fused so they move as one."""
    par = list(range(len(fr)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i, a in enumerate(fr):
        for j, b in enumerate(fr[:i]):
            ov = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            stacked = ov > 0.5 * min(a["w"], b["w"]) and max(a["y"], b["y"]) - min(a["y"] + a["h"], b["y"] + b["h"]) < 24
            if stacked:
                par[find(i)] = find(j)
    groups = {}
    for i, p in enumerate(fr):
        groups.setdefault(find(i), []).append(p)
    for g in groups.values():
        if len(g) < 2:
            continue
        x0, y0 = min(p["x"] for p in g), min(p["y"] for p in g)
        x1, y1 = max(p["x"] + p["w"] for p in g), max(p["y"] + p["h"] for p in g)
        img = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        for p in g:
            img.alpha_composite(p["img"], (p["x"] - x0, p["y"] - y0))
            P["pieces"].remove(p)
        P["pieces"].append(dict(img=img, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                                bottom=y1, area=sum(p["area"] for p in g), mean=g[0]["mean"], kind="friend"))


def make_flock(p, cell):
    img = p["img"]
    a = np.asarray(img.getchannel("A"))
    r = np.random.default_rng(9)
    tiles = []
    for gy in range(0, p["h"], cell):
        for gx in range(0, p["w"], cell):
            if a[gy:gy + cell, gx:gx + cell].mean() < 25:
                continue
            pad = cell // 3
            box = (gx - pad, gy - pad, gx + cell + pad, gy + cell + pad)
            crop = img.crop(box)
            m = Image.new("L", crop.size, 0)
            c = crop.size[0] / 2
            pts = [(c + math.cos(2 * math.pi * k / 9 + r.uniform(-.2, .2)) * r.uniform(cell * .62, cell * .8),
                    c + math.sin(2 * math.pi * k / 9 + r.uniform(-.2, .2)) * r.uniform(cell * .62, cell * .8)) for k in range(9)]
            ImageDraw.Draw(m).polygon(pts, fill=255)
            crop.putalpha(Image.fromarray(np.minimum(np.asarray(crop.getchannel("A")), np.asarray(m))))
            hx, hy = p["x"] + gx + cell / 2, p["y"] + gy + cell / 2
            tiles.append(dict(img=crop, hx=hx, hy=hy, k=len(tiles), d=r.uniform(0, 1), spin=r.uniform(-200, 200),
                              ph=r.uniform(0, 6.28), lane=r.uniform(-1, 1)))
    order = sorted(tiles, key=lambda t: t["hx"] + 0.3 * t["hy"])
    for j, t in enumerate(order):
        t["d"] = j / max(1, len(order) - 1)
    p["tiles"] = tiles
    p["bird_c"] = col(tuple(min(1, v * 1.05) for v in p["mean"]))


def build(frame, out, coin, cut=None):
    P = extract(frame, cut)
    W, H = P["W"], P["H"]
    sun = classify(P)
    for p in P["pieces"]:
        if p["kind"] == "flock":
            make_flock(p, max(18, int(min(W, H) * 0.024)))
    G.update(P)
    G["sun"] = sun
    G["coin"] = coin
    global BS
    BS = max(14, min(W, H) * 0.021)
    # paper + pastel finish, three boil variants
    r = np.random.default_rng(1)
    G["warp"], G["paint"], G["tooth"] = [], [], []
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(3):
        dx = (vnoise(300 + i, 14, 2, W, H) - 0.5) * 5
        dy = (vnoise(310 + i, 14, 2, W, H) - 0.5) * 5
        G["warp"].append((np.clip(Y + dy, 0, H - 1).astype(np.int32), np.clip(X + dx, 0, W - 1).astype(np.int32)))
        strokes = stroke_mod(340 + i, int(W * H / 75), 22, -1.0, 0.35, 2, W, H)
        G["paint"].append((1 + 0.07 * strokes + 0.10 * (vnoise(350 + i, 6, 3, W, H) - 0.5)).astype(np.float32))
        g = r.random((H, W)).astype(np.float32)
        tooth = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
        G["tooth"].append((0.95 + 0.10 * tooth + 0.05 * stipple(360 + i, 0.03, W, H)).astype(np.float32))
    # sky and ground repainted: hatching in the sky, pastel field strokes on the ground
    bg = G["bg"]
    hz = G["horizon"]
    sky_h = stroke_mod(400, int(W * H / 45), 34, -0.15, 0.08, 3, W, H)
    grd_h = stroke_mod(401, int(W * H / 50), 16, -1.3, 0.5, 2, W, H)
    wsky = (Y < hz)[..., None]
    bg = bg * np.where(wsky, 1 + 0.10 * sky_h[..., None], 1 + 0.14 * grd_h[..., None])
    vig = 1 - 0.32 * clip(((X - W / 2) / (W / 2)) ** 2 + ((Y - H / 2) / (H / 2)) ** 2) ** 1.4
    G["bg"] = clip(bg * vig[..., None]).astype(np.float32)
    G["vig"] = vig[..., None].astype(np.float32)
    G["out"] = out
    os.makedirs(out, exist_ok=True)


# ------------------------------------------------------------------ animation
def jit(k, ts, a=1.0):
    s = round(ts * FPS)
    return a * math.sin(k * 12.9898 + s * 78.233), a * math.sin(k * 39.346 + s * 11.135)


def put(im, img, x, y, rot=0.0, pivot=None):
    if abs(rot) > 0.2:
        w, h = img.size
        px, py = pivot if pivot else (w / 2, h / 2)
        pad = int(max(w, h) * 0.5)
        big = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        big.paste(img, (pad, pad))
        big = big.rotate(rot, resample=Image.BICUBIC, center=(px + pad, py + pad))
        img, x, y = big, x - pad, y - pad
    im.alpha_composite(img, (int(round(x)), int(round(y)))) if x >= 0 and y >= 0 and x + img.width <= im.width and y + img.height <= im.height \
        else _clip_put(im, img, int(round(x)), int(round(y)))


def _clip_put(im, img, x, y):
    l, t = max(0, -x), max(0, -y)
    r_, b = min(img.width, im.width - x), min(img.height, im.height - y)
    if r_ > l and b > t:
        im.alpha_composite(img.crop((l, t, r_, b)), (x + l, y + t))


def flock_state(t):
    """0 = whole cloud, 1 = all tiles are birds; also returns the flight phase."""
    u = t / LOOP
    return u


def bird_path(t, tl, W, H):
    cx, cy = G["coin"]
    hz = G["horizon"]
    ex, ey = W * 0.5, min(hz * 0.36, cy - 0.30 * min(W, H) - 0.10 * hz)
    rx, ry = W * 0.40, max(40, min(hz * 0.20, ey - 30))
    a = 2 * math.pi * ((t - 0.30 * LOOP) / (0.55 * LOOP)) * 1.0 + tl["ph"] * 0.15
    lane = tl["lane"]
    return ex + (rx + 26 * lane) * math.cos(a + math.pi), ey + (ry + 18 * lane) * math.sin(a + math.pi) + 8 * math.sin(3 * a + tl["ph"])


def render(i):
    t = i / FPS
    ts = t
    W, H = G["W"], G["H"]
    v = i % 3
    im = Image.fromarray((G["bg"] * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    sun = G["sun"]
    scx, scy = sun["cx"], sun["cy"]
    friends = []
    for k, p in enumerate(G["pieces"]):
        ph = (k * 1.618) % 6.28
        kind = p["kind"]
        if kind == "cloud":
            dx, dy = 14 * tau(t, 1, ph), 6 * tau(t, 2, ph)
            jx, jy = jit(k, ts, 0.8)
            put(im, p["img"], p["x"] + dx + jx, p["y"] + dy + jy, 2.5 * tau(t, 1, ph + 1))
        elif kind == "sun":
            put(im, p["img"], p["x"], p["y"], 6 * tau(t, 1, 0.3))
        elif kind == "ray":
            ang = math.atan2(p["cy"] - scy, p["cx"] - scx)
            rr = 9 * tau(t, 3, ph) + 4
            spin = math.radians(7 * tau(t, 1, 0.3))
            dist = math.hypot(p["cx"] - scx, p["cy"] - scy)
            nx = scx + (dist + rr) * math.cos(ang + spin) - p["cx"]
            ny = scy + (dist + rr) * math.sin(ang + spin) - p["cy"]
            put(im, p["img"], p["x"] + nx, p["y"] + ny, -math.degrees(spin))
        elif kind == "flock":
            u = t / LOOP
            for tl in p["tiles"]:
                lift = 0.20 + 0.12 * tl["d"]          # when this tile leaves the cloud
                land = 0.70 + 0.12 * tl["d"]          # when it is back
                th, tw = tl["img"].size
                hx, hy = tl["hx"] + 14 * tau(t, 1, (k * 1.618) % 6.28), tl["hy"] + 6 * tau(t, 2, (k * 1.618) % 6.28)
                if u < lift or u > land + 0.08:
                    jx, jy = jit(tl["k"] + 500, ts, 0.8)
                    put(im, tl["img"], hx - th / 2 + jx, hy - tw / 2 + jy)
                    continue
                bx, by = bird_path(t, tl, W, H)
                if u < lift + 0.08:                  # paper tile folds into a bird
                    q = ease((u - lift) / 0.08)
                    x, y = hx + (bx - hx) * q, hy + (by - hy) * q - 60 * math.sin(math.pi * q)
                    if q < 0.5:
                        put(im, tl["img"], x - th / 2, y - tw / 2, tl["spin"] * q)
                    else:
                        bird(d, x, y, BS * (1 + 0.4 * (tl["k"] % 3)), t * 24 + tl["k"], c=_c01(p["bird_c"]))
                elif u > land:                       # bird lands and unfolds into its tile
                    q = ease((u - land) / 0.08)
                    x, y = bx + (hx - bx) * q, by + (hy - by) * q - 60 * math.sin(math.pi * q)
                    if q > 0.5:
                        put(im, tl["img"], x - th / 2, y - tw / 2, tl["spin"] * (1 - q))
                    else:
                        bird(d, x, y, BS * (1 + 0.4 * (tl["k"] % 3)), t * 24 + tl["k"], c=_c01(p["bird_c"]))
                else:
                    bird(d, bx, by, BS * (1 + 0.4 * (tl["k"] % 3)), t * 24 + tl["k"], c=_c01(p["bird_c"]))
        else:
            friends.append((k, p))
    # friends stand on the ground: stop-motion sway around their feet, small hops, soft shadows
    for k, p in sorted(friends, key=lambda kp: kp[1]["bottom"]):
        ph = (k * 2.399) % 6.28
        hop = 16 * max(0.0, tau(t, 3 if k % 2 else 2, ph)) ** 3 if k % 3 == 0 else 0.0
        sway = 4.0 * tau(t, 2 if k % 2 else 3, ph)
        fx, fy = p["cx"], p["bottom"]
        sw = p["w"] * 0.42 * (1 - 0.25 * hop / 16)
        sh = Image.new("RGBA", (int(sw * 2 + 40), 60), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse([20, 20, 20 + 2 * sw, 40], fill=(0, 0, 0, 110))
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        put(im, sh, fx - sw - 20, fy - 34)
        jx, jy = jit(k + 99, ts, 0.9)
        put(im, p["img"], p["x"] + jx, p["y"] - hop + jy, sway, pivot=(p["w"] / 2, p["h"]))
    # finishing: line boil, pastel strokes, paper tooth
    f = np.asarray(im.convert("RGB"), np.float32) / 255
    yi, xi = G["warp"][v]
    f = f[yi, xi] * G["paint"][v][..., None] * G["tooth"][v][..., None]
    f = f * 0.97 + 0.015
    Image.fromarray((clip(f) * 255).astype(np.uint8)).save(os.path.join(G["out"], f"f_{i:04d}.png"), compress_level=1)
    return i


def _c01(rgb255):
    return tuple(c / 255 for c in rgb255[:3])


if __name__ == "__main__":
    frame, out, cx, cy = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    cut = sys.argv[5] if len(sys.argv) > 5 else None
    only = [int(x) for x in sys.argv[6:]]
    build(frame, out, (cx, cy), cut)
    print("pieces:", json.dumps({k: sum(1 for p in G["pieces"] if p["kind"] == k) for k in ("cloud", "sun", "ray", "flock", "friend")}),
          "horizon", G["horizon"], flush=True)
    with get_context("fork").Pool(4) as pool:
        for n, _ in enumerate(pool.imap_unordered(render, only or range(NF), chunksize=2)):
            pass
    print("done")
