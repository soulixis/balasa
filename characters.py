"""Character drawing. All figures are drawn in local units, scaled by s, onto a PIL draw."""
import math

from lib import almond_pts, col

UNIFORM = (0.66, 0.22, 0.18)
UNIFORM_WET = (0.42, 0.10, 0.10)
PANTS = (0.40, 0.10, 0.10)
GOLD = (0.82, 0.64, 0.28)
WHITE = (0.90, 0.89, 0.85)
SILVER = (0.78, 0.80, 0.83)


def _P(cx, cy, s):
    return lambda x, y: (cx + x * s, cy + y * s)


def _poly(d, P, pts, c, sil):
    d.polygon([P(x, y) for x, y in pts], fill=sil or c)


def horse_head(d, cx, cy, s, sil=None):
    """Frontal white horse head, centred at (cx, cy). About 260 units tall."""
    P = _P(cx, cy, s)
    c = lambda x: sil or col(x)
    for m in (-1, 1):  # ears
        d.polygon([P(m * 42, -100), P(m * 30, -165), P(m * 14, -104)], fill=c(WHITE))
        d.polygon([P(m * 36, -104), P(m * 29, -145), P(m * 20, -106)], fill=c((0.62, 0.55, 0.52)))
    head = [(-36, -112), (36, -112), (54, -80), (48, -10), (34, 48), (42, 92), (30, 122),
            (-30, 122), (-42, 92), (-34, 48), (-48, -10), (-54, -80)]
    _poly(d, P, head, col(WHITE), sil)
    if sil:
        return
    # soft grey modelling down the long nose
    d.polygon([P(-14, -60), P(14, -60), P(10, 60), P(-10, 60)], fill=col((0.82, 0.82, 0.79)))
    for m in (-1, 1):
        d.line([P(m * 30, -40), P(m * 22, 50)], fill=col((0.72, 0.72, 0.70)), width=max(1, int(3 * s)))
        d.polygon(almond_pts(*P(m * 44, -48), 11 * s, 5 * s, 10), fill=col((0.06, 0.05, 0.06)))
        d.ellipse([*P(m * 18 - 9, 92), *P(m * 18 + 9, 110)], fill=col((0.25, 0.2, 0.2)))
    for i in range(7):  # forelock
        x = -18 + i * 6
        d.line([P(x, -112), P(x * 0.6, -70 + (i % 3) * 6)], fill=col((0.55, 0.53, 0.52)), width=max(1, int(2 * s)))
    # pillbox cap
    d.rectangle([*P(-30, -142), *P(30, -112)], fill=col(UNIFORM))
    d.rectangle([*P(-30, -122), *P(30, -114)], fill=col(GOLD))


def bellhop(d, cx, fy, s, walk=None, tray=0.0, glass=True, wet=False, sil=None,
            key_arm=0.0, head=True):
    """Bellhop standing at foot point (cx, fy). Height about 1060 units.
    walk: walk-cycle phase or None. tray: 0 = held high, 1 = lowered to floor.
    key_arm: 0 = arm hanging, 1 = extended toward camera."""
    P = _P(cx, fy, s)
    c = lambda x: sil or col(x)
    uni = UNIFORM_WET if wet else UNIFORM
    bob = 0.0
    lifts = (0.0, 0.0)
    if walk is not None:
        sw = math.sin(walk)
        lifts = (max(0, sw) * 30, max(0, -sw) * 30)
        bob = -abs(sw) * 10
    # legs
    for m, lift in ((-1, lifts[0]), (1, lifts[1])):
        d.polygon([P(m * 12, -340 + bob), P(m * 72, -340 + bob), P(m * 62, -lift), P(m * 16, -lift)], fill=c(PANTS))
        d.ellipse([*P(m * 40 - 34, -lift - 16), *P(m * 40 + 34, -lift + 10)], fill=c((0.06, 0.05, 0.05)))
    B = lambda x, y: P(x, y + bob)
    # neck, jacket, collar
    d.polygon([B(-44, -700), B(44, -700), B(36, -770), B(-36, -770)], fill=c(WHITE))
    d.polygon([B(-130, -700), B(130, -700), B(110, -560), B(100, -400), B(116, -320), B(-116, -320),
               B(-100, -400), B(-110, -560)], fill=c(uni))
    if not sil:
        d.polygon([B(-42, -700), B(42, -700), B(0, -610)], fill=col(WHITE))
        d.polygon([B(-14, -688), B(14, -688), B(0, -676)], fill=col((0.05, 0.05, 0.05)))
        for m in (-1, 1):
            d.rectangle([*B(m * 130 - 30 * (m > 0), -706), *B(m * 130 + 30 * (m < 0), -694)], fill=col(GOLD))
            for i in range(6):
                y = -650 + i * 50
                d.ellipse([*B(m * 34 - 8, y - 8), *B(m * 34 + 8, y + 8)], fill=col(GOLD))
        if wet:  # darker wet streaks running down the coat
            for i in range(9):
                x = -100 + i * 25
                d.line([B(x, -690), B(x + 6, -330)], fill=col((0.30, 0.06, 0.07)), width=max(1, int(5 * s)))
    w = max(2, int(36 * s))
    # left arm hangs
    d.line([B(-118, -690), B(-150, -560), B(-138, -440)], fill=c(uni), width=w, joint="curve")
    d.ellipse([*B(-162, -462), *B(-114, -414)], fill=c(WHITE))
    # right arm: tray / key / hanging
    if key_arm > 0:
        hx, hy = 150 - 120 * key_arm, -440 - 140 * key_arm
        d.line([B(118, -690), B(150 - 40 * key_arm, -560), B(hx, hy)], fill=c(uni), width=w, joint="curve")
        r = 24 + 26 * key_arm
        d.ellipse([*B(hx - r, hy - r * 0.7), *B(hx + r, hy + r * 0.7)], fill=c(WHITE))
    elif tray is None:
        d.line([B(118, -690), B(150, -560), B(138, -440)], fill=c(uni), width=w, joint="curve")
        d.ellipse([*B(114, -462), *B(162, -414)], fill=c(WHITE))
    else:
        hy = -640 + 560 * tray
        hx = 175 + 40 * tray
        ey = -540 + 280 * tray
        d.line([B(118, -690), B(185, ey), B(hx, hy)], fill=c(uni), width=w, joint="curve")
        d.ellipse([*B(hx - 85, hy - 30), *B(hx + 85, hy + 2)], fill=c(SILVER))
        if not sil:
            d.ellipse([*B(hx - 85, hy - 30), *B(hx + 85, hy + 2)], outline=col((0.45, 0.47, 0.5)), width=max(1, int(3 * s)))
        d.ellipse([*B(hx - 22, hy - 14), *B(hx + 22, hy + 10)], fill=c(WHITE))
        if glass:
            milk_glass(d, *B(hx, hy - 16), s, sil)
    if head:
        horse_head(d, *B(0, -890), s, sil)


def milk_glass(d, x, y, s, sil=None):
    """Glass of milk standing with its base at (x, y)."""
    c = lambda v: sil or col(v)
    d.polygon([(x - 17 * s, y - 80 * s), (x + 17 * s, y - 80 * s), (x + 14 * s, y), (x - 14 * s, y)], fill=c((0.94, 0.94, 0.90)))
    d.ellipse([x - 17 * s, y - 86 * s, x + 17 * s, y - 74 * s], fill=c((0.98, 0.98, 0.96)))


def steam(d, x, y, s, t, alpha=0.5):
    for k in range(3):
        pts = []
        for i in range(18):
            q = i / 17
            pts.append((x + (k - 1) * 9 * s + 12 * s * math.sin(q * 6 + t * 2.5 + k * 2),
                        y - q * 110 * s))
        d.line(pts, fill=(235, 235, 230, int(255 * alpha * 0.8)), width=max(1, int(4 * s)))


def face(d, cx, cy, s, skin, hair, eyes="open", hair_len=1.0, child=False, iris=(0.15, 0.12, 0.10)):
    """Frontal Balasa-style face: long oval, almond eyes, stone stillness.
    eyes: 'open', 'down', 'black'."""
    rx, ry = (62, 74) if child else (68, 104)
    P = _P(cx, cy, s)
    # hair mass behind
    if hair_len > 0:
        hl = ry * (0.6 + 1.6 * hair_len)
        d.polygon([P(-rx * 1.25, -ry * 0.4), P(-rx * 0.9, -ry * 1.12), P(0, -ry * 1.22), P(rx * 0.9, -ry * 1.12),
                   P(rx * 1.25, -ry * 0.4), P(rx * 1.35, hl), P(-rx * 1.35, hl)], fill=col(hair))
    pts = []
    for i in range(48):
        a = 2 * math.pi * i / 48
        x, y = rx * math.sin(a), -ry * math.cos(a)
        if y > 0:
            x *= 1 - (0.18 if child else 0.32) * (y / ry) ** 2
        pts.append(P(x, y))
    d.polygon(pts, fill=col(skin))
    # fringe
    d.chord([*P(-rx * 1.02, -ry * 1.15), *P(rx * 1.02, -ry * 0.25)], 180, 360, fill=col(hair))
    shade = col(tuple(v * 0.78 for v in skin))
    ink = col((0.18, 0.12, 0.10))
    lw = max(1, int(3 * s))
    for m in (-1, 1):
        ex, ey = m * rx * 0.38, -ry * 0.02
        d.arc([*P(ex - rx * 0.26, ey - ry * 0.2), *P(ex + rx * 0.26, ey - ry * 0.02)], 200, 340, fill=shade, width=lw)
        if eyes == "down":
            d.arc([*P(ex - rx * 0.2, ey - ry * 0.07), *P(ex + rx * 0.2, ey + ry * 0.07)], 10, 170, fill=ink, width=lw)
        else:
            d.polygon(almond_pts(*P(ex, ey), rx * 0.21 * s, ry * 0.075 * s, 12), fill=col((0.92, 0.9, 0.86)))
            if eyes == "black":
                d.polygon(almond_pts(*P(ex, ey), rx * 0.21 * s, ry * 0.075 * s, 12), fill=col((0.01, 0.01, 0.01)))
            else:
                r = ry * 0.065
                d.ellipse([*P(ex - r, ey - r), *P(ex + r, ey + r)], fill=col(iris))
            d.line(almond_pts(*P(ex, ey), rx * 0.21 * s, ry * 0.075 * s, 12)[:13], fill=ink, width=lw)
    d.line([P(0, -ry * 0.05), P(-rx * 0.06, ry * 0.36)], fill=shade, width=lw)
    d.arc([*P(-rx * 0.16, ry * 0.3), *P(rx * 0.12, ry * 0.42)], 20, 160, fill=shade, width=lw)
    d.line([P(-rx * 0.22, ry * 0.6), P(rx * 0.22, ry * 0.6)], fill=col((0.55, 0.28, 0.26)), width=max(1, int(4 * s)))
    d.arc([*P(-rx * 0.2, ry * 0.58), *P(rx * 0.2, ry * 0.7)], 20, 160, fill=col((0.62, 0.36, 0.33)), width=lw)


def back_figure(d, cx, fy, s, body, hair, hat=None, kind=0):
    """A person seen from behind, standing still."""
    P = _P(cx, fy, s)
    if kind == 1:  # dress
        d.polygon([P(-55, -330), P(55, -330), P(95, -40), P(-95, -40)], fill=col(body))
        d.rectangle([*P(-30, -40), *P(-10, 0)], fill=col((0.15, 0.12, 0.1)))
        d.rectangle([*P(10, -40), *P(30, 0)], fill=col((0.15, 0.12, 0.1)))
    else:  # coat / suit
        d.polygon([P(-70, -330), P(70, -330), P(62, -120), P(-62, -120)], fill=col(body))
        d.rectangle([*P(-50, -125), *P(-8, 0)], fill=col(tuple(v * 0.7 for v in body)))
        d.rectangle([*P(8, -125), *P(50, 0)], fill=col(tuple(v * 0.7 for v in body)))
    d.rectangle([*P(-15, -355), *P(15, -325)], fill=col((0.75, 0.66, 0.58)))
    d.ellipse([*P(-38, -430), *P(38, -345)], fill=col(hair))
    if hat:
        d.rectangle([*P(-50, -420), *P(50, -410)], fill=col(hat))
        d.rectangle([*P(-32, -460), *P(32, -415)], fill=col(hat))
