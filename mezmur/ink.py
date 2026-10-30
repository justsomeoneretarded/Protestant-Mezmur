"""Hand-drawn ink transitions that carry each lyric line change.

Each new line gets one short flourish around the lyric area: a brush splash, a whipping thin line, or a
small burst of drops. Character (deliberately loose, varied per line):
- curved paths (bezier arcs), strokes tapered at both ends, flat solid colour;
- fast arrival (~0.1-0.2 s), short hold, then it breaks into drops that keep their momentum and shrink
  away (~0.5-0.9 s);
- small and local: it only plays around the lyric line.

Purely decorative: any failure must be swallowed by the caller and the video rendered without it.
"""
import math

import cv2
import numpy as np


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _ease_in_out(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Flourish:
    """One ink event, fully determined by its seed. Draws into a uint8 mask."""

    def __init__(self, t0, W, H, center_y, seed):
        rng = np.random.default_rng(seed)
        self.t0 = t0
        self.kind = ["splash", "line", "drops"][int(rng.choice(3, p=[0.45, 0.35, 0.20]))]
        s = H / 1080.0
        self.s = s
        side = 1 if rng.random() < 0.5 else -1          # sweep left->right or right->left
        span = W * rng.uniform(0.34, 0.55)
        cx = W / 2 + rng.uniform(-0.12, 0.12) * W
        cy = center_y + rng.uniform(-25, 35) * s
        a = np.array([cx - side * span / 2, cy + rng.uniform(10, 70) * s])
        d = np.array([cx + side * span / 2, cy + rng.uniform(-70, 10) * s])
        lift = rng.uniform(40, 110) * s * (1 if rng.random() < 0.5 else -1)   # the arc crosses the line
        b = a + (d - a) * 0.33 + np.array([0, lift])
        c = a + (d - a) * 0.66 + np.array([0, lift * rng.uniform(0.4, 1.1)])
        self.path = _bezier(a, b, c, d, 90)
        # Ink is never perfectly even: the width wobbles along the stroke, and a splash throws a few lobes.
        self.wobble = (rng.uniform(1.5, 4.0), rng.uniform(0, 6.3), rng.uniform(5.0, 9.0), rng.uniform(0, 6.3))
        self.lobes = []
        if self.kind == "splash":
            self.width = rng.uniform(18, 30) * s           # half-thickness at the widest point
            self.t_in, self.t_hold, self.t_out = rng.uniform(0.10, 0.18), rng.uniform(0.03, 0.08), rng.uniform(0.55, 0.85)
            n_drops = int(rng.integers(10, 20))
        elif self.kind == "line":
            self.width = rng.uniform(3.0, 5.5) * s
            self.t_in, self.t_hold, self.t_out = rng.uniform(0.16, 0.26), rng.uniform(0.0, 0.05), rng.uniform(0.35, 0.55)
            n_drops = int(rng.integers(0, 4))
        else:
            self.width = rng.uniform(6, 12) * s
            self.t_in, self.t_hold, self.t_out = rng.uniform(0.08, 0.14), 0.02, rng.uniform(0.5, 0.8)
            n_drops = int(rng.integers(14, 26))
        self.duration = self.t_in + self.t_hold + self.t_out
        if self.kind == "splash":
            for _ in range(int(rng.integers(2, 5))):
                j = int(rng.integers(len(self.path) // 4, len(self.path) - 6))
                tan = self.path[j + 3] - self.path[j]
                tan /= np.linalg.norm(tan) + 1e-6
                nrm = np.array([-tan[1], tan[0]]) * (1 if rng.random() < 0.5 else -1)
                length = rng.uniform(40, 110) * s
                end = self.path[j] + nrm * length + tan * length * rng.uniform(0.2, 0.9)
                ctrl = self.path[j] + tan * length * 0.5 + nrm * length * 0.3
                self.lobes.append((j / (len(self.path) - 1), _bezier(self.path[j], ctrl, ctrl, end, 18),
                                   self.width * rng.uniform(0.25, 0.5)))
        # Drops leave along the path, keep its direction (a bit slower), spread, and shrink to nothing.
        idx = rng.integers(len(self.path) // 3, len(self.path) - 1, max(0, n_drops))
        tang = self.path[np.minimum(idx + 1, len(self.path) - 1)] - self.path[idx]
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
        normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
        speed = rng.uniform(120, 420, (n_drops, 1)) * s
        self.d_pos = self.path[idx] + normal * rng.normal(0, 18 * s, (n_drops, 1))
        self.d_vel = tang * speed + normal * rng.normal(0, 90 * s, (n_drops, 1)) + np.array([0, rng.uniform(-30, 30) * s])
        lo, hi = (2.0, 3.5) if self.kind == "line" else (2.5, 11.0)
        self.d_r = rng.uniform(lo, hi, n_drops) * s
        self.d_delay = rng.uniform(0.0, 0.12, n_drops)

    def active(self, t):
        return self.t0 <= t <= self.t0 + self.duration

    def bbox(self, W, H):
        pts = np.vstack([self.path, self.d_pos + self.d_vel * self.duration])
        pad = int(self.width * 1.6 + 30 * self.s)
        x0, y0 = np.floor(pts.min(0)).astype(int) - pad
        x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    def _stroke(self, mask, pts, width, ox, oy, u0=0.0, taper_start=True):
        """A continuous, tapered, slightly uneven brush stroke along pts."""
        m = len(pts)
        if m < 2 or width < 0.6:
            return
        a1, p1, a2, p2 = self.wobble
        u = np.linspace(0, 1, m)
        taper = np.sin(np.pi * np.clip(u if taper_start else 0.5 + u / 2, 0, 1)) ** 0.7
        wob = 1 + 0.12 * np.sin(a1 * 6.28 * (u0 + u) + p1) + 0.08 * np.sin(a2 * 6.28 * (u0 + u) + p2)
        radii = width * (0.12 + 0.88 * taper) * wob
        pts = np.round(pts - np.array([ox, oy])).astype(int)
        for j in range(m - 1):
            r = float(min(radii[j], radii[j + 1]))
            if r >= 0.6:
                cv2.line(mask, tuple(pts[j]), tuple(pts[j + 1]), 255, max(1, int(round(2 * r))), cv2.LINE_AA)
        for j in range(0, m, 3):
            if radii[j] >= 1.0:
                cv2.circle(mask, tuple(pts[j]), int(round(radii[j])), 255, -1, cv2.LINE_AA)

    def draw(self, mask, t, ox=0, oy=0):
        e = t - self.t0
        n = len(self.path)
        if e < self.t_in:                       # head races in, tail stays at the start
            head = _ease_out(e / self.t_in)
            tail = 0.0
            fat = 0.35 + 0.65 * _ease_out(e / self.t_in)
        elif e < self.t_in + self.t_hold:
            head, tail, fat = 1.0, 0.0, 1.0
        else:                                   # tail catches up, body thins and breaks up
            k = (e - self.t_in - self.t_hold) / self.t_out
            head = 1.0
            tail = _ease_in_out(min(1.0, k * 1.25))
            fat = max(0.0, 1.0 - _ease_in_out(k) * 1.1)
        if self.kind != "drops" and head > tail and fat > 0.02:
            i0, i1 = int(tail * (n - 1)), max(int(tail * (n - 1)) + 1, int(head * (n - 1)))
            self._stroke(mask, self.path[i0:i1 + 1], self.width * fat, ox, oy, i0 / (n - 1))
            for at, lobe, w in self.lobes:      # side lobes of a splash appear once the body reaches them
                if tail < at < head:
                    grow = min(1.0, (head - at) * 4)
                    m = max(2, int(len(lobe) * grow))
                    self._stroke(mask, lobe[:m], w * fat, ox, oy, 0.0, taper_start=False)
        # Drops start when the body starts breaking up (immediately for a burst).
        start = self.t_in * (0.4 if self.kind == "drops" else 0.85)
        for p, v, r0, dl in zip(self.d_pos, self.d_vel, self.d_r, self.d_delay):
            le = e - start - dl
            if le <= 0:
                continue
            life = max(0.15, self.t_out * 0.95 + self.t_hold)
            k = le / life
            if k >= 1:
                continue
            pos = p + v * le * (1 - 0.45 * k)                     # momentum, slowing down
            r = r0 * (1 - k) ** 1.3                               # shrink to a point, never fade
            if r >= 0.7:
                stretch = 1.0 + 1.6 * (1 - k) * min(1.0, np.linalg.norm(v) / (300 * self.s))
                ang = math.degrees(math.atan2(v[1], v[0]))
                cv2.ellipse(mask, (int(pos[0]) - ox, int(pos[1]) - oy),
                            (max(1, int(r * stretch)), max(1, int(r))), ang, 0, 360, 255, -1, cv2.LINE_AA)


def _noop():
    return None


class InkTransitions:
    """Schedules one flourish per lyric line change and composites it onto frames."""

    def __init__(self, W, H, line_times, color_bgr, center_y=None, seed=0, min_gap=1.1):
        self.W, self.H = W, H
        self.color = np.array(color_bgr, np.float32)
        cy = center_y if center_y is not None else H * 0.52
        rng = np.random.default_rng(seed)
        self.events = []
        last = -10.0
        for k, t in enumerate(sorted(line_times)):
            start = t - 0.12                     # fully grown right as the new line appears
            if start - last < min_gap or start < 0:
                continue
            ev = Flourish(start, W, H, cy, int(rng.integers(1 << 30)))
            if self.events and ev.kind == self.events[-1].kind and rng.random() < 0.6:
                ev = Flourish(start, W, H, cy, int(rng.integers(1 << 30)))   # prefer variety
            self.events.append(ev)
            last = start + ev.duration * 0.6

    def apply(self, frame, t):
        for ev in self.events:
            if not ev.active(t):
                continue
            x0, y0, x1, y1 = ev.bbox(self.W, self.H)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            ev.draw(mask, t, x0, y0)
            a = (mask.astype(np.float32) / 255.0)[..., None]
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            frame[y0:y1, x0:x1] = (roi * (1 - a) + self.color * a).astype(np.uint8)
        return frame
