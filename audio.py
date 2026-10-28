"""Ambient drone + soft bells timed to the visual beats. Usage: python3 audio.py out.wav"""
import sys
import wave

import numpy as np

SR, DUR = 44100, 20.0
t = np.arange(int(SR * DUR)) / SR
rng = np.random.default_rng(1)

# low drone: detuned fifths with slow swells
drone = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in
            [(55, .5, 0), (55.4, .4, 1), (82.4, .3, 2), (110.2, .2, 3), (164.8, .08, 4)])
drone *= 0.6 + 0.4 * np.sin(2 * np.pi * t / 7.0) ** 2

# airy noise, smoothed, like wind or sea
noise = np.convolve(rng.normal(0, 1, t.size), np.ones(60) / 60, "same")
noise *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 4.3) ** 2
noise *= np.clip((t - 13) / 2, 0.3, 1)          # sea swells in scene 3


def bell(at, f, amp=0.35):
    env = np.where(t >= at, np.exp(-(t - at) * 1.6), 0)
    return amp * env * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.76 * t))


bells = bell(2.6, 392) + bell(4.6, 294) + bell(8.0, 523, .25) + bell(15.9, 349) + bell(17.2, 466, .2)
mix = 0.35 * drone / 1.4 + 0.25 * noise + 0.3 * bells
fade = np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 2.5, 0, 1)
mix *= fade / np.max(np.abs(mix)) * 0.8
left = mix + 0.02 * np.roll(noise, 900)
right = mix + 0.02 * np.roll(noise, -900)
pcm = (np.stack([left, right], 1) / 1.05 * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
