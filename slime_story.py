"""SLIM-E AND FRIENDS — 50 s, 4:3, silent stop-motion story in the style of Sabin Balasa.

Slim-E (the vinyl figure photos) wakes in his clay world. His friends are grey. The eight hoodie sets fall
from the clouds and catch on a clothesline; each one brings a friend's colour back. The sets walk off
the line like empty people behind Slim-E, and at night their star eyes rise and become the stars.

Usage: python3 slime_story.py <frames_dir> [frame numbers...]
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

import engine
import slime_world as sw
from birds import step
from lib import FPS, H, W, clip, col, ease, from_pil, lerp, ramp

A = {}
CUT = "/home/user/balasa_assets/assets/site_cut/"
SETS = ["sets_rednavy", "sets_pinkpurple", "sets_blackyellow", "sets_yellowgreen"]
S43 = 960 / 1080
X0 = 240                      # crop of the 16:9 world into 4:3
HZ = int(604 * S43)
LINES = (175, 345)
LINE_X = (110, 1170)


def jit(k, t, a=1.0):
    s = round(t * 12)
    return a * math.sin(k * 12.9898 + s * 78.233), a * math.sin(k * 39.346 + s * 11.135)


def scaled(img, h):
    s = h / img.height
    return img.resize((max(1, int(img.width * s)), max(1, int(h))), Image.LANCZOS)


def put(im, img, x, y, rot=0.0, pivot=None):
    sw.put(im, img, x, y, rot, pivot)


def grey(img):
    g = ImageEnhance.Color(img).enhance(0.0)
    return ImageEnhance.Brightness(g).enhance(0.92)


def mix(a, b, k):
    if k <= 0.001:
        return a
    if k >= 0.999:
        return b
    return Image.blend(a, b, k)


def tear(img, seed):
    """Torn paper edge + pastel grain on a cut-out."""
    a = np.asarray(img, np.float32) / 255
    r = np.random.default_rng(seed)
    h, w = a.shape[:2]
    nz = np.asarray(Image.fromarray((r.random((max(2, h // 6), max(2, w // 6))) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                    np.float32) / 255
    al = clip((a[..., 3] - 0.5 + 0.25 * (nz - 0.5)) * 4)
    rgb = a[..., :3] * (0.94 + 0.12 * r.random((h, w, 1)).astype(np.float32))
    return Image.fromarray((np.dstack([clip(rgb), al]) * 255).astype(np.uint8), "RGBA")


# =====================================================================================
def build():
    P = sw.extract("/home/user/balasa/work/site/d0.png", CUT + "world_d0.png")
    sw.G.update(P)
    sw.classify(P)
    bg = Image.fromarray((P["bg"] * 255).astype(np.uint8)).crop((X0, 0, X0 + 1440, 1080)).resize((W, H), Image.LANCZOS)
    A["bg"] = np.asarray(bg, np.float32) / 255
    sky = np.zeros((H, W, 1), np.float32)
    sky[:HZ] = 1
    night = A["bg"] * np.array([0.22, 0.26, 0.42], np.float32)
    night = night * (1 - sky) + (A["bg"] * np.array([0.10, 0.13, 0.30], np.float32) + np.array([0.02, 0.03, 0.08])) * sky
    A["night"] = night
    r = np.random.default_rng(5)
    A["stars_bg"] = [(r.uniform(0, W), r.uniform(0, HZ - 30), r.uniform(1, 2.4)) for _ in range(140)]
    pieces = []
    for p in P["pieces"]:
        img = p["img"].resize((max(1, int(p["w"] * S43)), max(1, int(p["h"] * S43))), Image.LANCZOS)
        q = dict(kind=p["kind"], img=img, x=(p["x"] - X0) * S43, y=p["y"] * S43, w=img.width, h=img.height,
                 cx=(p["cx"] - X0) * S43, cy=p["cy"] * S43, bottom=p["bottom"] * S43)
        if p["kind"] == "flock":
            q["kind"] = "cloud"
        if q["kind"] == "friend":
            q["grey"] = grey(img)
        pieces.append(q)
    A["pieces"] = pieces
    fr = sorted([p for p in pieces if p["kind"] == "friend"], key=lambda p: p["cx"])
    A["friends"] = fr
    # Slim-E, the vinyl figure, in four poses
    for pose, h in (("front", 300), ("back", 300), ("side", 300), ("lying", 150)):
        A[pose] = tear(scaled(Image.open(CUT + f"slime_{pose}.png").convert("RGBA"), h), 11)
    # the eight sets as flat-lay cut-outs, plus each back-hoodie face print
    sets, faces = [], []
    for name in SETS:
        im = Image.open(CUT + name + ".png").convert("RGBA")
        a = np.asarray(im)[..., 3] > 100
        rows = np.nonzero(a.any(1))[0]
        split = rows[np.argmax(np.diff(rows))] + 1 if np.diff(rows).max() > 20 else im.height // 2
        for y0, y1 in ((0, split), (split, im.height)):
            card = im.crop((0, y0, im.width, y1))
            card = card.crop(card.getbbox())
            sets.append(tear(scaled(card, 175), len(sets)))
            cw, ch = card.size
            fx, fy, fr_ = cw * 0.265, ch * 0.42, ch * 0.24
            face = card.crop((int(fx - fr_), int(fy - fr_), int(fx + fr_), int(fy + fr_)))
            m = Image.new("L", face.size, 0)
            ImageDraw.Draw(m).ellipse([0, 0, face.width - 1, face.height - 1], fill=255)
            face.putalpha(Image.fromarray(np.minimum(np.asarray(face.getchannel("A")), np.asarray(m))))
            faces.append(scaled(face, 70))
    A["sets"], A["faces"] = sets, faces
    order = [0, 2, 4, 6, 1, 3, 5, 7]          # alternate colours along the line
    A["order"] = order
    n = len(sets)
    # soften the paper finish for his bright world
    for i in range(len(engine.A["paint"])):
        engine.A["paint"][i] = 1 + 0.45 * (engine.A["paint"][i] - 1)
        engine.A["tooth"][i] = 1 + 0.35 * (engine.A["tooth"][i] - 1)


def line_y(x, row=1):
    u = (x - LINE_X[0]) / (LINE_X[1] - LINE_X[0])
    return LINES[row] + 40 * math.sin(math.pi * clip(u))


def slot(i):
    row, idx = i % 2, i // 2
    return lerp(LINE_X[0] + 150, LINE_X[1] - 150, idx / 3), row


# =====================================================================================
def world(t, night=0.0, colour=None, skip_friends=False):
    f = A["bg"] if night <= 0 else A["bg"] * (1 - night) + A["night"] * night
    im = Image.fromarray((clip(f) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    if night > 0:
        for i, (x, y, rr) in enumerate(A["stars_bg"]):
            a = night * (0.5 + 0.5 * math.sin(t * 2 + i))
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(250, 245, 220, int(220 * a)))
    sun_y = 0.0
    for k, p in enumerate(A["pieces"]):
        ph = (k * 1.618) % 6.28
        if p["kind"] == "cloud":
            img = p["img"] if night < 0.05 else Image.blend(p["img"], ImageEnhance.Brightness(p["img"]).enhance(0.35), night)
            put(im, img, p["x"] + 10 * math.sin(t * 0.4 + ph) + 3 * t, p["y"] + 5 * math.sin(t * 0.8 + ph))
        elif p["kind"] in ("sun", "ray"):
            rise = A.get("sun_rise", 0.0)
            if night > 0.95:
                continue
            img = p["img"] if night < 0.05 else Image.blend(p["img"], Image.new("RGBA", p["img"].size, (0, 0, 0, 0)), night)
            put(im, img, p["x"], p["y"] + rise, 5 * math.sin(t * 0.5) if p["kind"] == "sun" else 0)
    if not skip_friends:
        draw_friends(im, t, night, colour)
    return im


def draw_friends(im, t, night, colour):
    for k, p in enumerate(A["friends"]):
        ph = (k * 2.399) % 6.28
        c = 1.0 if colour is None else colour[k] if k < len(colour) else 1.0
        img = mix(p["grey"], p["img"], c)
        if night > 0:
            img = ImageEnhance.Brightness(img).enhance(1 - 0.6 * night)
        sway = 3.5 * math.sin(t * 1.6 + ph)
        hop = 12 * max(0.0, math.sin(t * 2.2 + ph)) ** 3 if (k % 3 == 0 and c > 0.9) else 0
        fx, fy = p["cx"], p["bottom"]
        swd = p["w"] * 0.4
        sh = Image.new("RGBA", (int(swd * 2 + 40), 50), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse([20, 18, 20 + 2 * swd, 34], fill=(0, 0, 0, 100))
        put(im, sh, fx - swd - 20, fy - 30)
        jx, jy = jit(k + 99, t, 0.9)
        put(im, img, p["x"] + jx, p["y"] - hop + jy, sway, pivot=(p["w"] / 2, p["h"]))


def slime(im, pose, x, foot, t, k=7, rot=0.0, bob=0.0):
    img = A[pose]
    jx, jy = jit(k, t, 1.0)
    swd = img.width * 0.42
    sh = Image.new("RGBA", (int(swd * 2 + 40), 50), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([20, 18, 20 + 2 * swd, 34], fill=(0, 0, 0, 110))
    put(im, sh, x - swd - 20, foot - 26)
    put(im, img, x - img.width / 2 + jx, foot - img.height - bob + jy, rot, pivot=(img.width / 2, img.height))


def frame(im, cam=(640, 480, 1.0), w=1.0):
    return [(from_pil(im.convert("RGB")), cam, w, None)]


# =====================================================================================
GREY = [0.0] * 9


def s1(t):
    """Morning: Slim-E asleep in the grass, the sun rises."""
    ts = step(t)
    A["sun_rise"] = lerp(260, 0, ease(ts / 5.5))
    im = world(ts, colour=GREY)
    slime(im, "lying", 640, 800, ts, rot=0)
    z = lerp(1.6, 1.15, ease(ts / 7))
    return frame(im, (640, lerp(700, 560, ease(ts / 7)), z))


def s2(t):
    """He gets up and walks through the grey field."""
    ts = step(t)
    A["sun_rise"] = 0
    im = world(ts, colour=GREY)
    if ts < 1.2:
        slime(im, "lying" if ts < 0.6 else "front", 640, 800, ts)
    elif ts < 2.0:
        slime(im, "front", 640, 800, ts)
    else:
        x = lerp(640, 900, ease((ts - 2.0) / 5.0))
        walk = round(ts * 12) % 4
        slime(im, "side", x, 800, ts, rot=[0, 3, 0, -3][walk], bob=[0, 8, 0, 8][walk])
    return frame(im, (lerp(640, 720, ease(ts / 8)), 560, 1.15))


SPOT = (760, 820)


def caught(i, ts):
    """Time at which set i (in line order) lands on the line."""
    return 1.4 + i * 1.35


def draw_line(im, ts, upto):
    d = ImageDraw.Draw(im)
    x1 = lerp(LINE_X[0], LINE_X[1], upto)
    for px in LINE_X:
        if px <= x1 + 1:
            d.line([(px, LINES[0] - 20), (px + (8 if px < 640 else -8), HZ + 230)], fill=(70, 50, 40, 255), width=8)
    for row in (0, 1):
        pts = [(x, line_y(x, row)) for x in np.linspace(LINE_X[0], x1, 40)]
        if len(pts) > 1:
            d.line(pts, fill=(245, 240, 228, 255), width=3)


def hang(im, i, ts, swing=1.0):
    img = A["sets"][A["order"][i]]
    x, row = slot(i)
    y = line_y(x, row)
    rot = swing * 5 * math.sin(ts * 1.7 + i)
    put(im, img, x - img.width / 2, y - 8, rot, pivot=(img.width / 2, 8))
    d = ImageDraw.Draw(im)
    for dx in (-img.width * 0.32, img.width * 0.32):
        d.rectangle([x + dx - 3, y - 9, x + dx + 3, y + 7], fill=(210, 180, 130, 255))


def s3(t):
    """The sets fall from the clouds like paper leaves and catch on a clothesline; colour returns."""
    ts = step(t)
    colour = [ramp(ts, caught(i, ts), caught(i, ts) + 0.8) for i in range(9)]
    if len(A["friends"]) > 8:
        colour[8] = colour[7]
    im = world(ts, colour=colour)
    draw_line(im, ts, ease(ts / 1.4))
    for i in range(len(A["sets"])):
        tc = caught(i, ts)
        sx, row = slot(i)
        start = tc - 2.2
        if ts < start:
            continue
        if ts < tc:
            q = (ts - start) / 2.2
            x = sx + 120 * math.sin(q * 5 + i) * (1 - q)
            y = lerp(-200, line_y(sx, row), ease(q))
            img = A["sets"][A["order"][i]]
            flip = max(0.12, abs(math.cos(q * 7 + i)))
            fl = img.resize((max(2, int(img.width * flip)), img.height))
            put(im, fl, x - fl.width / 2, y - 8, 25 * math.sin(q * 4 + i) * (1 - q))
        else:
            hang(im, i, ts)
    look = "front" if ts < 8 else "back"
    slime(im, look, SPOT[0], SPOT[1], ts)
    return frame(im, (640, 480, 1.0))


def s4(t):
    """The sets step off the line and walk across the field behind Slim-E."""
    ts = step(t)
    im = world(ts)
    draw_line(im, ts, 1.0)
    walk = round(ts * 12) % 4
    n = len(A["sets"])
    for i in range(n):
        leave = 0.4 + i * 0.35
        sx, row = slot(i)
        if ts < leave:
            hang(im, i, ts)
            continue
        q = ease((ts - leave) / 1.2)
        march = max(0.0, ts - leave - 1.2) * 22
        x = lerp(sx, 70 + i * 112, q) + march - 60
        foot = lerp(line_y(sx, row) + 175, 770 + (i % 2) * 45, q)
        img = A["sets"][A["order"][i]]
        bob = [0, 10, 0, 10][(walk + i) % 4] * q
        put(im, img, x - img.width / 2, foot - img.height - bob, [0, 3, 0, -3][(walk + i) % 4] * q, pivot=(img.width / 2, img.height))
    lead = 1010 + max(0.0, ts - 1.0) * 22 - 120 * (1 - ease(ts / 1.5))
    slime(im, "side", lead, 860, ts, rot=[0, 3, 0, -3][walk], bob=[0, 8, 0, 8][walk])
    return frame(im, (lerp(640, 700, ease(ts / 10)), 520, lerp(1.0, 1.08, ease(ts / 10))))


def s5(t):
    """Night: the star eyes rise off the hoodies and become the stars; Slim-E lies down under them."""
    ts = step(t)
    nt = ease(ts / 2.5)
    im = world(ts, night=nt)
    n = len(A["sets"])
    rng = np.random.default_rng(3)
    for i in range(n):
        x = 140 + i * 140
        img = A["sets"][A["order"][i]]
        dark = ImageEnhance.Brightness(img).enhance(1 - 0.45 * nt)
        put(im, dark, x - img.width / 2, 800 - img.height + (i % 2) * 30)
        q = ease((ts - 2.5 - i * 0.35) / 3.0)
        if q > 0:
            face = A["faces"][A["order"][i]]
            fx = lerp(x - img.width * 0.23, 120 + rng.uniform(0, 1040), q)
            fy = lerp(800 - img.height * 0.58 + (i % 2) * 30, 70 + rng.uniform(0, 260), q)
            s = lerp(0.7, 0.55, q)
            fi = face.resize((max(2, int(face.width * s)), max(2, int(face.height * s))))
            glow = Image.new("RGBA", (fi.width + 40, fi.height + 40), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse([0, 0, glow.width, glow.height], fill=(255, 245, 210, int(70 * q)))
            put(im, glow, fx - glow.width / 2, fy - glow.height / 2)
            put(im, fi, fx - fi.width / 2, fy - fi.height / 2, 20 * q)
    slime(im, "front" if ts < 4.5 else "lying", 640, 880, ts)
    w = 1 - ramp(t, 8.6, 10.0)
    return frame(im, (640, lerp(480, 420, ease(ts / 10)), 1.0), w)


SHOTS = [(s1, 0, 7, 0), (s2, 7, 8, 0.8), (s3, 15, 15, 0.8), (s4, 30, 10, 0.8), (s5, 40, 10, 0.8)]
TOTAL = 50 * FPS

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[2:]]
    engine.run(SHOTS, TOTAL, sys.argv[1], frames, [build], "pastel", boil=2)
