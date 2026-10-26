"""Music features that drive the reactive visuals: energy, spectrum bands, onsets, beats."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from .audio import read_audio

SR = 22050


@dataclass
class Features:
    fps: float
    rms: np.ndarray        # [frames] 0..1 loudness (smoothed)
    bands: np.ndarray      # [frames, n_bands] 0..1 spectrum
    onset: np.ndarray      # [frames] 0..1 onset strength
    beats: np.ndarray      # beat times in seconds
    tempo: float
    beat_env: np.ndarray   # [frames] 0..1 decaying pulse after each beat
    downbeats: np.ndarray  # every 4th beat (bar starts), seconds

    def at(self, t):
        return min(len(self.rms) - 1, max(0, int(t * self.fps)))


def _normalize(x, lo=5, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def _smooth(x, attack, release):
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = acc + (attack if v > acc else release) * (v - acc)
        out[i] = acc
    return out


def track_beats(onset_env, fps, tightness=100.0):
    """Dynamic-programming beat tracker (Ellis 2007) on an onset envelope sampled at ``fps``."""
    env = onset_env - onset_env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lags = np.arange(len(ac)) / fps
    bpm = np.where(lags > 0, 60.0 / np.maximum(lags, 1e-9), 0)
    weight = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-9) / 110.0) / 0.9) ** 2)
    valid = (bpm >= 60) & (bpm <= 180)
    score = np.where(valid, ac * weight, -np.inf)
    period = int(np.argmax(score)) if np.isfinite(score).any() else int(fps * 0.55)
    period = max(1, period)
    tempo = 60.0 * fps / period
    n = len(onset_env)
    cumulative = onset_env.astype(float).copy()
    back = np.full(n, -1)
    window = np.arange(-2 * period, -period // 2)
    penalty = -tightness * (np.log(-window / period) ** 2)
    for i in range(n):
        idx = i + window
        ok = idx >= 0
        if not ok.any():
            continue
        cand = cumulative[idx[ok]] + penalty[ok]
        j = int(np.argmax(cand))
        cumulative[i] = onset_env[i] + cand[j]
        back[i] = idx[ok][j]
    tail = cumulative[max(0, n - period):]
    i = int(np.argmax(tail)) + max(0, n - period)
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1]) / fps, tempo


def analyze(path, fps=30.0, n_bands=64):
    samples = read_audio(path, SR)
    hop = SR / fps
    nper = 2048
    f, t, Z = stft(samples, fs=SR, nperseg=nper, noverlap=nper - int(round(hop)), boundary=None, padded=True)
    mag = np.abs(Z).T  # [frames, freqs]
    frames = int(np.ceil(len(samples) / SR * fps))
    if mag.shape[0] < frames:
        mag = np.pad(mag, ((0, frames - mag.shape[0]), (0, 0)))
    mag = mag[:frames]
    power = mag ** 2
    rms = np.sqrt(power.mean(1))
    rms = _smooth(_normalize(np.log1p(rms * 100)), 0.5, 0.08)
    edges = np.geomspace(40, 12000, n_bands + 1)
    bins = np.searchsorted(f, edges)
    bands = np.stack([power[:, bins[i]:max(bins[i] + 1, bins[i + 1])].mean(1) for i in range(n_bands)], 1)
    bands = np.log1p(bands * 1e3)
    bands = bands / (np.percentile(bands, 99, axis=0, keepdims=True) + 1e-9)
    bands = np.clip(bands, 0, 1)
    logmag = np.log1p(mag * 50)
    flux = np.maximum(0, np.diff(logmag, axis=0, prepend=logmag[:1])).sum(1)
    onset = _normalize(flux, 10, 99.5)
    beats, tempo = track_beats(onset, fps)
    beats = beats + (nper / 2) / SR  # STFT frames are labelled by window start
    env = np.zeros(frames)
    period = 60.0 / tempo if tempo else 0.5
    for b in beats:
        i = int(b * fps)
        span = np.arange(i, min(frames, i + int(fps * period)))
        env[span] = np.maximum(env[span], np.exp(-(span - i) / (fps * 0.12)))
    downbeats = beats[::4] if len(beats) else beats
    return Features(fps, rms.astype(np.float32), bands.astype(np.float32), onset.astype(np.float32),
                    beats, float(tempo), env.astype(np.float32), downbeats)


def snap(t, grid, tolerance):
    """Snap a time to the nearest grid point within tolerance."""
    if len(grid) == 0:
        return t
    i = int(np.argmin(np.abs(grid - t)))
    return float(grid[i]) if abs(grid[i] - t) <= tolerance else t
