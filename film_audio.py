"""Soundtrack for THE HOTEL THAT DREAMS OF ME (60 s, stereo, 44.1 kHz).

Drone, underwater rumble, distant machinery, a detuned music box, the mother's line,
sounds behind the blue doors, one telephone ring, hard silence.
Needs espeak-ng for the voice. Usage: python3 film_audio.py out.wav
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

SR, DUR = 44100, 60.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
L = np.zeros(N)
R = np.zeros(N)


def add(sig, start, gain=1.0, pan=0.0):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    L[i:i + sig.size] += sig * np.sqrt((1 - pan) / 2)
    R[i:i + sig.size] += sig * np.sqrt((1 + pan) / 2)


def lowpass(x, n):
    n = max(1, int(n))
    k = np.ones(n) / n
    return fftconv(x, k)[: x.size]


def fftconv(a, b):
    n = a.size + b.size - 1
    m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, m) * np.fft.rfft(b, m), m)[:n]


def reverb(x, secs=1.6, wet=0.35, seed=1):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = r.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 4.5 / secs)
    ir = lowpass(ir, 6)
    y = fftconv(x, ir)
    y = y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))
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


def seg(a, b):
    return slice(int(a * SR), int(b * SR))


def load_tts(text, voice, speed, pitch, gap=4):
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.wav")
            subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-g", str(gap), "-w", p, text],
                           check=True, capture_output=True)
            with wave.open(p) as w:
                sr = w.getframerate()
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    except (OSError, subprocess.CalledProcessError):
        return np.zeros(SR)
    nz = np.nonzero(np.abs(x) > 0.01)[0]
    x = x[nz[0]:nz[-1]] if nz.size else x
    return np.interp(np.arange(0, x.size, sr / SR), np.arange(x.size), x)


def stretch(x, factor):
    """Resample: factor < 1 = slower and lower."""
    return np.interp(np.arange(0, x.size - 1, factor), np.arange(x.size), x)


# ---- bed: drone + underwater rumble + distant machinery -----------------------------
drone = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in
            [(43.65, .5, 0), (43.9, .4, 1), (65.4, .25, 2), (87.3, .18, 3), (130.8, .06, 4)])
drone *= 0.6 + 0.4 * np.sin(2 * np.pi * t / 9.0) ** 2
brown = np.cumsum(rng.normal(0, 1, N))
brown -= lowpass(brown, 4000)
rumble = lowpass(brown, 120)
rumble /= np.max(np.abs(rumble))
hum = (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 150 * t + 1)
       + 0.15 * np.sin(2 * np.pi * 200 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t))
throb = np.sin(2 * np.pi * 31 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 0.8 * t))) * 0.5
level = np.interp(t, [0, 7, 14, 21, 29, 38, 47, 60], [1.0, 0.9, 0.8, 0.6, 0.4, 0.7, 0.6, 0.6])
water = np.interp(t, [0, 7, 14, 29, 38, 47, 60], [1.0, 0.5, 0.4, 0.15, 0.4, 0.35, 0.35])
mach = np.interp(t, [0, 7, 14, 21, 29, 38, 47, 60], [0.2, 0.5, 0.7, 0.15, 0.1, 0.3, 0.25, 0.25])
bed = 0.30 * drone * level + 0.55 * rumble * water + 0.06 * hum * mach + 0.10 * throb * mach
add(bed, 0)

# ---- shot 1: a whale call ------------------------------------------------------------
n = int(3.2 * SR)
tt = np.arange(n) / SR
f0 = np.interp(tt, [0, 1.2, 3.2], [190, 115, 145]) * (1 + 0.01 * np.sin(2 * np.pi * 5 * tt))
ph = 2 * np.pi * np.cumsum(f0) / SR
call = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * env(n, 0.6, 1.2)
add(reverb(call, 2.5, 0.6), 1.4, 0.18, -0.3)


# ---- detuned music box --------------------------------------------------------------
def tine(freq, secs=1.6):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 2.8)
            + 0.3 * np.sin(2 * np.pi * freq * 3.9 * tt) * np.exp(-tt * 9)
            + 0.15 * np.sin(2 * np.pi * freq * 6.1 * tt) * np.exp(-tt * 16))


MEL = [69, 72, 76, 74, 72, 71, 69, 64, 65, 69, 72, 71, 69, 68, 69, None]


def music_box(start, end, step, transpose=1.0, gain=0.12, pan=0.2, seed=3):
    r = np.random.default_rng(seed)
    k, tm = 0, start
    while tm < end:
        m = MEL[k % len(MEL)]
        if m is not None and r.random() > 0.12:
            f = 440 * 2 ** ((m - 69) / 12) * transpose * 2 ** (r.normal(0, 22) / 1200)
            add(tine(f), tm + r.normal(0, 0.02), gain * r.uniform(0.6, 1.0), pan)
        tm += step * r.uniform(0.92, 1.12)
        k += 1


music_box(0.5, 6.5, 0.62, gain=0.07)
music_box(14.5, 28.5, 0.55, gain=0.10, seed=4)
music_box(47.5, 57.6, 0.85, transpose=0.93, gain=0.11, seed=5)

# ---- shot 2: wet exhale through the keyhole, then the cut ---------------------------
n = int(0.9 * SR)
nz = rng.normal(0, 1, n)
breath = lowpass(nz, 6) - lowpass(nz, 40)
breath *= np.sin(np.linspace(0, np.pi, n)) ** 1.5 * (0.75 + 0.25 * np.sin(2 * np.pi * 26 * np.arange(n) / SR))
add(breath, 13.2, 0.9)


# ---- shot 3: elevator floor clicks, doors opening ----------------------------------
def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def click(freqs=(2300, 3150, 4100), secs=0.12):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * f * tt) for f in freqs) * np.exp(-tt * 45) / len(freqs) + \
        np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 30) * 0.8


ts = np.arange(0, 7, 0.001)
fl = -5 - 7 * ease((ts - 0.5) / 3.5)
for i in range(1, ts.size):
    if np.floor(fl[i]) != np.floor(fl[i - 1]):
        add(click(), 14 + ts[i], 0.35, 0.1)
n = int(1.4 * SR)
door = lowpass(rng.normal(0, 1, n), 30) * 6 * env(n, 0.2, 0.6)
door += 0.2 * np.sin(2 * np.pi * np.cumsum(np.linspace(260, 180, n)) / SR) * env(n, 0.3, 0.6)
add(door, 18.2, 0.5)

# ---- shot 4: wind over the field, the balloon rushing in ----------------------------
n = int(8 * SR)
wind = lowpass(rng.normal(0, 1, n), 25) * 5
wind *= (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / SR / 3.1) ** 2) * env(n, 1.5, 0.5)
add(wind, 21, 0.25, -0.2)
n = int(2.0 * SR)
nz = rng.normal(0, 1, n)
whoosh = (lowpass(nz, 30) * 3 * (1 - np.linspace(0, 1, n)) + nz * 0.25 * np.linspace(0, 1, n) ** 2)
whoosh *= np.linspace(0, 1, n) ** 2
add(whoosh, 27, 0.45)

# ---- shot 5: clock ticking backward, the mother's line ------------------------------
tick = click((1800, 2600), 0.06)[::-1]
for k in range(9):
    add(tick, 29.6 + k, 0.25, 0.5)
line = load_tts("You're late. Your room has been waiting for you.", "en-us+f3", 150, 38, 3)
line = stretch(line, 0.9)
line = lowpass(line, 3)
voice = reverb(line, 1.4, 0.3, 2)
voice = 0.75 * voice + 0.25 * np.concatenate([np.zeros(int(0.018 * SR)), voice])[: voice.size]
add(voice / (np.max(np.abs(voice)) + 1e-9), 33.6, 0.45)
VOICE_SECS = line.size / SR

# ---- shot 6: behind the doors ------------------------------------------------------
def muffle(x, n=18):
    return reverb(lowpass(x, n), 1.8, 0.5, 3)


laugh = load_tts("ha ha ha ha, ha ha", "en-us+f2", 190, 75)
add(muffle(laugh), 39.0, 0.25, -0.7)
n = int(6.5 * SR)
waves = lowpass(rng.normal(0, 1, n), 14) * 4
waves *= (np.sin(2 * np.pi * np.arange(n) / SR / 2.6) ** 2) ** 1.5 * env(n, 1, 1)
add(waves, 39.6, 0.3, 0.6)


def cry(secs=1.0, f_a=430, f_b=540):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    f0 = np.interp(tt, [0, secs * 0.3, secs], [f_a, f_b, f_a * 0.85]) * (1 + 0.03 * np.sin(2 * np.pi * 6 * tt))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((1 / k) * np.sin(k * ph) * (1.4 if 2 <= k <= 3 else 1) for k in range(1, 10))
    return x * env(n, 0.08, 0.3)


add(muffle(np.concatenate([cry(), np.zeros(SR // 4), cry(0.8, 470, 560)]), 22), 41.4, 0.22, 0.0)
for txt, sp, p, at, pan in (("remember", 120, 40, 42.0, -0.5), ("come in", 110, 30, 43.6, 0.6), ("it was always you", 120, 45, 44.6, -0.2)):
    w = load_tts(txt, "en-us+whisper", sp, p)
    add(reverb(w, 1.2, 0.4, 4), at, 0.35, pan)
n = int(0.5 * SR)
tt = np.arange(n) / SR
slam = np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 9) + rng.normal(0, 1, n) * np.exp(-tt * 40) * 0.6
add(slam, 45.92, 0.9)
L[seg(46.2, 47.0)] = 0
R[seg(46.2, 47.0)] = 0
L[seg(14.0, 14.25)] *= 0.05      # the hard cut after the keyhole
R[seg(14.0, 14.25)] *= 0.05

# ---- shot 7: room 217 ---------------------------------------------------------------
n = int(1.2 * SR)
tt = np.arange(n) / SR
shimmer = sum(np.sin(2 * np.pi * f * tt) for f in (1568, 2093, 2637)) * np.exp(-tt * 3) / 3
add(shimmer, 49.4, 0.10, 0.2)
n = int(0.4 * SR)
tt = np.arange(n) / SR
add(np.sin(2 * np.pi * 40 * tt) * np.exp(-tt * 10), 57.8, 0.6)   # the frame where his face is yours
L[seg(58.2, DUR)] = 0
R[seg(58.2, DUR)] = 0
n = int(0.95 * SR)
tt = np.arange(n) / SR
bell = sum(a * np.sin(2 * np.pi * f * tt) for f, a in ((880, 1), (1175, 0.7), (2350, 0.3)))
ring = bell * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 20 * tt))) * env(n, 0.01, 0.15)
add(ring, 58.5, 0.35, 0.3)

# ---- master ------------------------------------------------------------------------
fade = np.clip(t / 1.0, 0, 1)
L *= fade
R *= fade
peak = max(np.max(np.abs(L)), np.max(np.abs(R)))
pcm = (np.stack([L, R], 1) / peak * 0.89 * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"voice line {VOICE_SECS:.2f}s")
