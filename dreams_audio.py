"""Soundtrack for THE PLACE BETWEEN DREAMS (55 s). No voices.
Usage: python3 dreams_audio.py out.wav
"""
import sys
import wave

import numpy as np

SR, DUR = 44100, 55.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(11)
L, R = np.zeros(N), np.zeros(N)
MBL, MBR = np.zeros(N), np.zeros(N)      # music box on its own bus so it can stop abruptly


def add(sig, start, gain=1.0, pan=0.0, bus=None):
    l_, r_ = (MBL, MBR) if bus == "mb" else (L, R)
    i = int(start * SR)
    if i >= N or i < 0:
        return
    sig = sig[: N - i] * gain
    l_[i:i + sig.size] += sig * np.sqrt((1 - pan) / 2)
    r_[i:i + sig.size] += sig * np.sqrt((1 + pan) / 2)


def fftconv(a, b):
    n = a.size + b.size - 1
    m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, m) * np.fft.rfft(b, m), m)[:n]


def lowpass(x, n):
    n = max(1, int(n))
    return fftconv(x, np.ones(n) / n)[: x.size]


def reverb(x, secs=2.0, wet=0.4, seed=1):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = lowpass(r.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 4.5 / secs), 5)
    y = fftconv(x, ir)
    y = y / (np.max(np.abs(y)) + 1e-9) * (np.max(np.abs(x)) + 1e-9)
    out = np.zeros(y.size)
    out[: x.size] += x * (1 - wet)
    return out + y * wet


def env(n, a, r):
    e = np.ones(n)
    ai, ri = int(a * SR), int(r * SR)
    if ai:
        e[:ai] = np.linspace(0, 1, ai)
    if ri:
        e[-ri:] *= np.linspace(1, 0, ri)
    return e


def tone(freqs, secs, decay=3.0):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * tt) for f, a in freqs) * np.exp(-tt * decay)


# ---- bed: warm drone + soft wind ------------------------------------------------------
drone = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in
            [(73.42, .5, 0), (73.7, .35, 1), (110.0, .3, 2), (146.8, .16, 3), (185.0, .07, 4), (220.0, .05, 5)])
drone *= 0.65 + 0.35 * np.sin(2 * np.pi * t / 11.0) ** 2
nz = rng.normal(0, 1, N)
wind = lowpass(nz, 22) - lowpass(nz, 300)
wind *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 5.3 + 1) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * t / 13))
wind /= np.max(np.abs(wind))
bed_level = np.interp(t, [0, 23, 32, 41, 50, 53.3, 55], [1, 1, 0.8, 0.9, 1, 1, 0])
wind_level = np.interp(t, [0, 16, 23, 23.5, 32, 41, 55], [1, 0.8, 0.8, 0.2, 0.3, 0.7, 0.8])
add(0.30 * drone * bed_level + 0.22 * wind * wind_level * bed_level, 0)


# ---- distant birds ---------------------------------------------------------------------
def chirp():
    n = int(rng.uniform(0.05, 0.11) * SR)
    tt = np.arange(n) / SR
    f = np.linspace(rng.uniform(2300, 3200), rng.uniform(3600, 4800), n)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.04 * np.sin(2 * np.pi * 45 * tt))) / SR
    return np.sin(ph) * np.sin(np.linspace(0, np.pi, n))


tm = 0.4
while tm < 50:
    dens = np.interp(tm, [0, 8, 16, 23, 32, 41, 50], [1.0, 0.7, 0.4, 0.05, 0.1, 0.3, 0.4])
    if rng.random() < dens:
        grp = np.concatenate([np.concatenate([chirp(), np.zeros(int(rng.uniform(0.04, 0.12) * SR))]) for _ in range(rng.integers(2, 5))])
        add(reverb(lowpass(grp, 3), 1.5, 0.5, int(tm * 10)), tm, 0.05, rng.uniform(-0.8, 0.8))
    tm += rng.uniform(0.6, 1.6)

# ---- shot 1: birds dissolving into clouds; a cloud brushes the lens -------------------
add(reverb(tone([(1318, 1), (1976, 0.6), (2637, 0.4)], 2.5, 1.2), 2.5, 0.6), 4.0, 0.05, 0.3)
n = int(2.2 * SR)
w = lowpass(rng.normal(0, 1, n), 14) * np.sin(np.linspace(0, np.pi, n)) ** 2 * 4
add(w, 6.2, 0.35, 0.0)

# ---- shot 2: a tower falls upward ------------------------------------------------------
n = int(2.0 * SR)
tt = np.arange(n) / SR
f = np.geomspace(500, 2600, n)
gl = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.6) * env(n, 0.02, 0.4)
add(reverb(gl, 2.0, 0.5, 3), 12.6, 0.06, 0.2)
for k in range(5):
    add(reverb(tone([(2093 * 2 ** (k / 5), 1)], 0.6, 6), 1.0, 0.5, 4 + k), 12.8 + k * 0.25, 0.03, 0.3)


# ---- shot 3: steps in the grass, a door that opens by itself, light ------------------
def step():
    n = int(0.09 * SR)
    x = rng.normal(0, 1, n)
    return (lowpass(x, 8) - lowpass(x, 60)) * np.exp(-np.arange(n) / SR * 40)


for k in range(10):
    add(step(), 16.3 + k * 0.45, 0.25 * (0.4 + 0.6 * k / 9), -0.3 + 0.03 * k)


def creak(secs, f0=95, f1=70):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    f = np.linspace(f0, f1, n) * (1 + 0.15 * np.sin(2 * np.pi * 7 * tt) + 0.05 * rng.normal(0, 1, n))
    ph = np.cumsum(f) / SR
    saw = 2 * (ph % 1) - 1
    x = lowpass(saw, 4) - lowpass(saw, 25)
    return x * env(n, 0.05, 0.2)


add(creak(0.8), 18.6, 0.18, 0.2)
add(creak(0.9, 110, 75), 21.6, 0.25, 0.2)
n = int(2.6 * SR)
tt = np.arange(n) / SR
swell = sum(np.sin(2 * np.pi * f * tt) for f in (587.3, 880, 1174.7, 1480)) / 4
swell *= np.linspace(0, 1, n) ** 2 * np.exp(-np.clip(tt - 1.2, 0, None) * 2)
swell += lowpass(rng.normal(0, 1, n), 3) * 0.3 * np.linspace(0, 1, n) ** 3 * np.exp(-np.clip(tt - 1.2, 0, None) * 3)
add(reverb(swell, 2.5, 0.5, 5), 21.8, 0.16)

# ---- shot 4: the world upside down — reversed sounds, water falling upward ---------
n = int(9 * SR)
water = lowpass(rng.normal(0, 1, n), 5) - lowpass(rng.normal(0, 1, n), 60)
water *= env(n, 1.5, 1.5)
rev = np.zeros(n)
for k in range(12):  # reversed droplets: swell, then cut
    m = int(0.5 * SR)
    d_ = tone([(rng.uniform(900, 1700), 1)], 0.5, 9)[::-1]
    i = int(rng.uniform(0, 8.4) * SR)
    rev[i:i + m] += d_ * 0.6
add(water * 0.12 + rev * 0.08, 23.2, 1.0, 0.1)
n = int(8 * SR)
tt = np.arange(n) / SR
whirl = lowpass(rng.normal(0, 1, n), 40) * 6 * np.sin(np.linspace(0, np.pi, n)) * (0.6 + 0.4 * np.sin(2 * np.pi * tt / 2.0))
add(whirl, 23.8, 0.2, 0.0)

# ---- shot 5: a door emerging from a painting, mirrored footsteps, the door opens ---
for k in range(26):
    add(tone([(rng.uniform(2500, 5000), 1)], 0.25, 18), 33.2 + k * 0.1 + rng.uniform(0, 0.08), 0.02, rng.uniform(-0.6, 0.6))
for k in range(7):
    add(step(), 34.5 + k * 0.45, 0.2, 0.5)
    add(step(), 34.5 + k * 0.45, 0.12, -0.5)
add(creak(1.0, 100, 72), 38.4, 0.22, 0.3)

# ---- shot 6: the smile; the long pull away ------------------------------------------
n = int(3.5 * SR)
tt = np.arange(n) / SR
pad = sum(np.sin(2 * np.pi * f * tt) for f in (293.7, 370.0, 440.0, 587.3)) / 4 * np.sin(np.linspace(0, np.pi, n)) ** 2
add(reverb(pad, 2.5, 0.5, 7), 42.4, 0.12)
n = int(5.5 * SR)
air = lowpass(rng.normal(0, 1, n), 10) * np.linspace(0, 1, n) ** 2 * env(n, 0, 1.5) * 2
add(air, 45.0, 0.18)

# ---- shot 7: someone on the other side ----------------------------------------------
for k, at in enumerate((51.3, 51.9, 52.4)):
    n = int(0.25 * SR)
    th = lowpass(rng.normal(0, 1, n), 80) * np.exp(-np.arange(n) / SR * 18) * 8
    add(th, at, 0.12 * (1 - 0.2 * k))
add(tone([(3100, 1), (4400, 0.6), (1900, 0.4)], 0.15, 40), 52.9, 0.12, 0.1)
add(creak(0.4, 160, 140), 53.0, 0.06, 0.1)

# ---- the music box: slowly emerging, stopping without warning -----------------------
MEL = [74, 78, 81, 83, 81, 78, 76, 74, 69, 71, 74, 76, 78, 76, 74, 71]
tm, k = 9.0, 0
while tm < 54.3:
    p = np.interp(tm, [9, 16, 23, 32, 41, 50], [0.25, 0.45, 0.6, 0.8, 0.95, 1.0])
    g = np.interp(tm, [9, 16, 23, 32, 41, 50], [0.05, 0.07, 0.09, 0.11, 0.12, 0.12])
    if rng.random() < p:
        f = 440 * 2 ** ((MEL[k % len(MEL)] - 69) / 12) * 2 ** (rng.normal(0, 12) / 1200)
        note = tone([(f, 1), (f * 3.9, 0.3), (f * 6.1, 0.12)], 1.8, 2.6)
        if 23 < tm < 32 and k % 3 == 0:     # upside-down world: some notes play backward
            note = note[::-1] * np.linspace(0.3, 1, note.size)
        add(note, tm, g * rng.uniform(0.7, 1.0), 0.15, bus="mb")
    tm += 0.62 * rng.uniform(0.95, 1.08)
    k += 1
stop = int(54.3 * SR)
MBL[stop:] = 0
MBR[stop:] = 0
L += MBL
R += MBR

# ---- master ---------------------------------------------------------------------------
L *= np.clip(t / 1.0, 0, 1)
R *= np.clip(t / 1.0, 0, 1)
peak = max(np.max(np.abs(L)), np.max(np.abs(R)))
pcm = (np.stack([L, R], 1) / peak * 0.89 * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("ok")
