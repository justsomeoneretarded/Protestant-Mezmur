"""Hand-drawn ink transitions that carry each lyric line change.

Every line change gets its own flourish, built from random combinations so they rarely repeat:

* path shapes   — arc, S-curve, wave, loop, zigzag, spiral, ring around the line, underline, swoosh,
                  hook, figure-eight, radial spray;
* brush styles  — thick splash (uneven edges, side lobes), thin whip line, double line, dashed line,
                  comet (fat head, thin tail), ribbon (width pulses), scribble (several quick strokes),
                  drop burst, splat (spikes from a point);
* variation     — 1-3 layered strokes, direction, mirroring, rotation, size, speed, timing offsets,
                  how much it breaks into drops and which way they fly.

Common character: curved tapered strokes in flat colour, fast arrival, short hold, then the ink breaks
into drops that keep their momentum and shrink away. Small and local to the lyric line.

Purely decorative: any failure must be swallowed by the caller and the video rendered without it.
"""
import math

import cv2
import numpy as np

SHAPES = ["arc", "s_curve", "wave", "loop", "zigzag", "spiral", "ring", "underline", "swoosh", "hook",
          "figure8", "spray"]
STYLES = ["splash", "line", "double", "dashed", "comet", "ribbon", "scribble", "burst", "splat"]


def _ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _ease_in_out(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _catmull(points, n):
    """Smooth curve through control points."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.repeat(p, n, 0)
    p = np.vstack([p[0] * 2 - p[1], p, p[-1] * 2 - p[-2]])
    segs = len(p) - 3
    out = []
    for k in range(segs):
        p0, p1, p2, p3 = p[k], p[k + 1], p[k + 2], p[k + 3]
        t = np.linspace(0, 1, max(2, n // segs), endpoint=k == segs - 1)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def _shape(name, rng, span, height):
    """Unit-ish path (centred on 0,0) for a shape, in pixels."""
    w, h = span / 2, height
    if name == "arc":
        lift = rng.uniform(0.4, 1.0) * h * rng.choice([-1, 1])
        return _catmull([(-w, h * 0.3), (-w * 0.3, -lift), (w * 0.4, -lift * 0.8), (w, -h * 0.2)], 90)
    if name == "s_curve":
        a = rng.uniform(0.5, 1.0) * h
        return _catmull([(-w, a), (-w * 0.35, -a), (w * 0.35, a), (w, -a)], 100)
    if name == "wave":
        x = np.linspace(-w, w, 120)
        k = rng.uniform(1.5, 3.5)
        return np.stack([x, np.sin(x / w * math.pi * k + rng.uniform(0, 6)) * h * rng.uniform(0.25, 0.6)], 1)
    if name == "loop":
        r = h * rng.uniform(0.5, 0.9)
        lx = rng.uniform(-0.2, 0.3) * w
        return _catmull([(-w, h * 0.4), (lx - r, 0), (lx, -r), (lx + r, 0), (lx, r * 0.6), (lx - r * 0.4, -r * 0.2),
                         (w, -h * 0.5)], 130)
    if name == "zigzag":
        k = int(rng.integers(3, 6))
        xs = np.linspace(-w, w, k)
        ys = [h * rng.uniform(0.35, 0.8) * (1 if i % 2 else -1) for i in range(k)]
        return _catmull(list(zip(xs, ys)), 120)
    if name == "spiral":
        turns = rng.uniform(1.2, 2.2)
        t = np.linspace(0, 1, 140)
        ang = t * turns * 2 * math.pi + rng.uniform(0, 6)
        rad = (0.15 + 0.85 * t) * h * rng.uniform(0.8, 1.3)
        return np.stack([np.cos(ang) * rad * 1.6, np.sin(ang) * rad], 1)
    if name == "ring":
        a0 = rng.uniform(0, 2 * math.pi)
        sweep = rng.uniform(1.3, 1.9) * math.pi
        ang = a0 + np.linspace(0, sweep, 140)
        return np.stack([np.cos(ang) * w * rng.uniform(0.75, 0.95), np.sin(ang) * h * rng.uniform(0.9, 1.3)], 1)
    if name == "underline":
        y = h * rng.uniform(0.6, 0.95)
        return _catmull([(-w, y + h * 0.1), (-w * 0.2, y - h * 0.15), (w * 0.5, y + h * 0.05), (w * 1.05, y - h * 0.3)], 90)
    if name == "swoosh":
        return _catmull([(-w * 1.05, h * 0.9), (-w * 0.4, h * 0.2), (w * 0.3, -h * 0.2), (w * 1.05, -h * 0.9)], 90)
    if name == "hook":
        return _catmull([(-w, h * 0.2), (w * 0.3, h * 0.1), (w * 0.8, -h * 0.4), (w * 0.55, -h * 0.9),
                         (w * 0.3, -h * 0.5)], 100)
    if name == "figure8":
        t = np.linspace(0, 2 * math.pi * rng.uniform(0.7, 1.0), 150)
        return np.stack([np.sin(t) * w * 0.8, np.sin(t) * np.cos(t) * h * 1.2], 1)
    # spray: a short straight-ish flick, drops do the rest
    ang = rng.uniform(-0.6, 0.6)
    L = w * rng.uniform(0.5, 0.9)
    return _catmull([(-L * math.cos(ang), -L * math.sin(ang)), (0, h * 0.1), (L * math.cos(ang), L * math.sin(ang))], 60)


class _Stroke:
    def __init__(self, path, width, t_off, t_in, t_hold, t_out, style, rng, s):
        self.path, self.width, self.style = path, width, style
        self.t_off, self.t_in, self.t_hold, self.t_out = t_off, t_in, t_hold, t_out
        self.end = t_off + t_in + t_hold + t_out
        self.wobble = (rng.uniform(1.5, 4.0), rng.uniform(0, 6.3), rng.uniform(5.0, 9.0), rng.uniform(0, 6.3))
        self.dash = (int(rng.integers(5, 10)), rng.uniform(0.45, 0.7))
        self.pulse = rng.uniform(2.0, 5.0)
        self.tail_mode = rng.choice(["chase", "fade_thin", "both_ends"])
        self.lobes = []
        if style in ("splash", "comet"):
            for _ in range(int(rng.integers(1, 5))):
                j = int(rng.integers(len(path) // 5, len(path) - 6))
                tan = path[j + 3] - path[j]
                tan /= np.linalg.norm(tan) + 1e-6
                nrm = np.array([-tan[1], tan[0]]) * rng.choice([-1, 1])
                L = rng.uniform(30, 110) * s
                end = path[j] + nrm * L + tan * L * rng.uniform(0.2, 0.9)
                ctrl = path[j] + tan * L * 0.5 + nrm * L * 0.3
                self.lobes.append((j / (len(path) - 1), _catmull([path[j], ctrl, end], 18), width * rng.uniform(0.2, 0.5)))

    def profile(self, u, u0):
        a1, p1, a2, p2 = self.wobble
        if self.style == "comet":
            base = (1 - u) ** 0.6 * 0.95 + 0.05            # fat head at the leading end
            base = base[::-1]
        elif self.style == "ribbon":
            base = 0.35 + 0.65 * np.abs(np.sin((u0 + u) * math.pi * self.pulse))
        else:
            base = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.7
        wob = 1 + 0.13 * np.sin(a1 * 6.28 * (u0 + u) + p1) + 0.08 * np.sin(a2 * 6.28 * (u0 + u) + p2)
        return (0.12 + 0.88 * base) * wob

    def draw(self, mask, e, ox, oy):
        e = e - self.t_off
        if e <= 0 or e >= self.t_in + self.t_hold + self.t_out:
            return
        n = len(self.path)
        if e < self.t_in:
            head, tail, fat = _ease_out(e / self.t_in), 0.0, 0.35 + 0.65 * _ease_out(e / self.t_in)
        elif e < self.t_in + self.t_hold:
            head, tail, fat = 1.0, 0.0, 1.0
        else:
            k = (e - self.t_in - self.t_hold) / self.t_out
            if self.tail_mode == "chase":
                head, tail, fat = 1.0, _ease_in_out(min(1.0, k * 1.25)), max(0.0, 1 - _ease_in_out(k) * 1.1)
            elif self.tail_mode == "fade_thin":
                head, tail, fat = 1.0, _ease_in_out(k) * 0.5, max(0.0, 1 - _ease_in_out(k) * 1.05)
            else:  # shrinks from both ends toward the middle
                head, tail, fat = 1 - 0.5 * _ease_in_out(k), 0.5 * _ease_in_out(k), max(0.0, 1 - _ease_in_out(k))
        if head <= tail or fat <= 0.02:
            return
        i0, i1 = int(tail * (n - 1)), max(int(tail * (n - 1)) + 1, int(head * (n - 1)))
        pts = self.path[i0:i1 + 1]
        m = len(pts)
        if m < 2:
            return
        u = np.linspace(0, 1, m)
        radii = self.width * fat * self.profile(u, i0 / (n - 1))
        P = np.round(pts - np.array([ox, oy])).astype(int)
        dashes, on = self.dash
        for j in range(m - 1):
            if self.style == "dashed" and ((u[j] * dashes) % 1.0) > on:
                continue
            r = float(min(radii[j], radii[j + 1]))
            if r >= 0.6:
                cv2.line(mask, tuple(P[j]), tuple(P[j + 1]), 255, max(1, int(round(2 * r))), cv2.LINE_AA)
        if self.style != "dashed":
            for j in range(0, m, 3):
                if radii[j] >= 1.0:
                    cv2.circle(mask, tuple(P[j]), int(round(radii[j])), 255, -1, cv2.LINE_AA)
        for at, lobe, w in self.lobes:
            if tail < at < head:
                grow = min(1.0, (head - at) * 4)
                q = max(2, int(len(lobe) * grow))
                L = np.round(lobe[:q] - np.array([ox, oy])).astype(int)
                lr = w * fat * (1 - np.linspace(0, 1, q)) ** 0.8
                for j in range(q - 1):
                    if lr[j] >= 0.6:
                        cv2.line(mask, tuple(L[j]), tuple(L[j + 1]), 255, max(1, int(round(2 * lr[j]))), cv2.LINE_AA)


class Flourish:
    """One ink event (1-3 layered strokes + drops), fully determined by its seed."""

    def __init__(self, t0, W, H, center_y, seed):
        rng = np.random.default_rng(seed)
        self.t0 = t0
        s = H / 1080.0
        self.s = s
        self.style = str(rng.choice(STYLES, p=[0.17, 0.15, 0.1, 0.08, 0.12, 0.1, 0.1, 0.1, 0.08]))
        self.shape = str(rng.choice(SHAPES))
        self.kind = f"{self.style}/{self.shape}"
        span = W * rng.uniform(0.28, 0.55)
        height = rng.uniform(55, 120) * s
        cx = W / 2 + rng.uniform(-0.14, 0.14) * W
        cy = center_y + rng.uniform(-30, 40) * s
        rot = rng.uniform(-0.35, 0.35)
        flip_x, flip_y = rng.choice([-1, 1]), rng.choice([-1, 1])
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])

        def place(path, dx=0.0, dy=0.0):
            p = path * np.array([flip_x, flip_y])
            if rng.random() < 0.5:
                p = p[::-1]
            return p @ R.T + np.array([cx + dx, cy + dy])

        speed = rng.uniform(0.75, 1.35)
        base_w = {"splash": rng.uniform(16, 30), "line": rng.uniform(2.5, 5.5), "double": rng.uniform(2.0, 4.0),
                  "dashed": rng.uniform(3.0, 6.0), "comet": rng.uniform(12, 22), "ribbon": rng.uniform(7, 14),
                  "scribble": rng.uniform(2.0, 4.0), "burst": rng.uniform(6, 12), "splat": rng.uniform(8, 16)}[self.style] * s
        t_in, t_hold, t_out = rng.uniform(0.09, 0.24) * speed, rng.uniform(0.0, 0.09), rng.uniform(0.4, 0.9) * speed
        self.strokes = []
        main = place(_shape(self.shape, rng, span, height))
        if self.style == "double":
            off = rng.uniform(8, 18) * s
            nrm = np.gradient(main, axis=0)[:, ::-1] * np.array([1, -1])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
            for k, sign in enumerate((-1, 1)):
                self.strokes.append(_Stroke(main + nrm * off * sign, base_w, 0.03 * k, t_in, t_hold, t_out, "line", rng, s))
        elif self.style == "scribble":
            for k in range(int(rng.integers(3, 6))):
                sub = place(_shape(str(rng.choice(["arc", "zigzag", "wave", "hook", "swoosh"])), rng,
                                   span * rng.uniform(0.4, 0.8), height * rng.uniform(0.4, 0.9)),
                            rng.uniform(-0.2, 0.2) * span, rng.uniform(-0.4, 0.4) * height)
                self.strokes.append(_Stroke(sub, base_w * rng.uniform(0.8, 1.3), 0.05 * k, t_in * 0.8, 0.0, t_out * 0.7,
                                            "line", rng, s))
        elif self.style not in ("burst", "splat"):
            self.strokes.append(_Stroke(main, base_w, 0.0, t_in, t_hold, t_out, self.style, rng, s))
            if rng.random() < 0.35:  # a thinner echo stroke following the main one
                echo = place(_shape(str(rng.choice(SHAPES[:-1])), rng, span * 0.7, height * 0.7),
                             rng.uniform(-0.15, 0.15) * span, rng.uniform(-0.5, 0.5) * height)
                self.strokes.append(_Stroke(echo, max(2.0 * s, base_w * 0.25), rng.uniform(0.04, 0.12), t_in, t_hold,
                                            t_out * 0.8, "line", rng, s))
        elif self.style == "splat":   # short fat spikes radiating from a point
            origin = np.array([cx, cy])
            for k in range(int(rng.integers(5, 10))):
                ang = rng.uniform(0, 2 * math.pi)
                L = rng.uniform(40, 120) * s
                tip = origin + np.array([math.cos(ang), math.sin(ang) * 0.7]) * L
                path = _catmull([origin, (origin + tip) / 2 + rng.normal(0, 8 * s, 2), tip], 20)
                self.strokes.append(_Stroke(path, base_w * rng.uniform(0.5, 1.1), rng.uniform(0, 0.05), t_in * 0.7,
                                            t_hold, t_out * 0.8, "comet", rng, s))
        self.width = max([st.width for st in self.strokes] or [base_w])
        stroke_end = max([st.end for st in self.strokes] or [t_in + t_hold])

        # Drops: how many, where from and where to depend on the style.
        n_drops = {"splash": (10, 22), "line": (0, 5), "double": (0, 4), "dashed": (2, 8), "comet": (6, 14),
                   "ribbon": (4, 10), "scribble": (2, 8), "burst": (16, 32), "splat": (8, 18)}[self.style]
        n_drops = int(rng.integers(*n_drops))
        if self.style in ("burst", "splat") or not self.strokes:
            origin = place(np.zeros((1, 2)))[0] if self.style == "burst" else np.array([cx, cy])
            ang = rng.uniform(0, 2 * math.pi, n_drops)
            spread = rng.uniform(0.4, 2 * math.pi) if self.style == "burst" else 2 * math.pi
            ang = rng.uniform(0, 2 * math.pi) + (ang - math.pi) * spread / (2 * math.pi)
            dirs = np.stack([np.cos(ang), np.sin(ang) * 0.75], 1)
            self.d_pos = origin + dirs * rng.uniform(0, 15 * s, (n_drops, 1))
            self.d_vel = dirs * rng.uniform(150, 520, (n_drops, 1)) * s
        else:
            src = self.strokes[0].path
            idx = rng.integers(len(src) // 4, len(src) - 1, max(0, n_drops))
            tang = src[np.minimum(idx + 1, len(src) - 1)] - src[idx]
            tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
            normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
            spin = rng.choice([-1, 1]) * rng.uniform(0, 0.6)
            self.d_pos = src[idx] + normal * rng.normal(0, 16 * s, (n_drops, 1))
            self.d_vel = (tang * rng.uniform(100, 420, (n_drops, 1)) + normal * rng.normal(spin * 200, 90, (n_drops, 1))) * s
        self.d_vel += np.array([0, rng.uniform(-40, 60)]) * s          # slight drift up or down
        lo, hi = {"line": (2.0, 3.5), "double": (2.0, 3.0), "scribble": (2.0, 3.5)}.get(self.style, (2.5, 11.0))
        self.d_r = rng.uniform(lo, hi, n_drops) * s
        self.d_start = (t_in * rng.uniform(0.2, 0.9)) + rng.uniform(0.0, 0.14, n_drops)
        self.d_life = rng.uniform(0.35, 0.8, n_drops) * speed
        self.t_in, self.t_hold, self.t_out = t_in, t_hold, t_out
        drops_end = float((self.d_start + self.d_life).max()) if n_drops else 0.0
        self.duration = max(stroke_end, drops_end)

    def active(self, t):
        return self.t0 <= t <= self.t0 + self.duration

    def bbox(self, W, H):
        pts = [st.path for st in self.strokes] + [self.d_pos + self.d_vel * 1.0, self.d_pos]
        pts = np.vstack([p for p in pts if len(p)])
        pad = int(self.width * 1.6 + 40 * self.s)
        x0, y0 = np.floor(pts.min(0)).astype(int) - pad
        x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    def draw(self, mask, t, ox=0, oy=0):
        e = t - self.t0
        for st in self.strokes:
            st.draw(mask, e, ox, oy)
        for p, v, r0, st, life in zip(self.d_pos, self.d_vel, self.d_r, self.d_start, self.d_life):
            le = e - st
            if le <= 0 or le >= life:
                continue
            k = le / life
            pos = p + v * le * (1 - 0.45 * k)                      # momentum, slowing down
            r = r0 * (1 - k) ** 1.3                                # shrink to a point, never fade
            if r >= 0.7:
                stretch = 1.0 + 1.6 * (1 - k) * min(1.0, np.linalg.norm(v) / (300 * self.s))
                ang = math.degrees(math.atan2(v[1], v[0]))
                cv2.ellipse(mask, (int(pos[0]) - ox, int(pos[1]) - oy),
                            (max(1, int(r * stretch)), max(1, int(r))), ang, 0, 360, 255, -1, cv2.LINE_AA)


class InkTransitions:
    """Schedules one flourish per lyric line change and composites it onto frames."""

    def __init__(self, W, H, line_times, color_bgr, center_y=None, seed=0, min_gap=1.1):
        self.W, self.H = W, H
        self.color = np.array(color_bgr, np.float32)
        cy = center_y if center_y is not None else H * 0.52
        rng = np.random.default_rng(seed)
        self.events = []
        last = -10.0
        recent = []
        for t in sorted(line_times):
            start = t - 0.12                     # fully grown right as the new line appears
            if start - last < min_gap or start < 0:
                continue
            ev = None
            for _ in range(8):                   # avoid repeating a recent style or shape
                ev = Flourish(start, W, H, cy, int(rng.integers(1 << 30)))
                if ev.style not in [r[0] for r in recent[-2:]] and ev.shape not in [r[1] for r in recent[-3:]]:
                    break
            recent.append((ev.style, ev.shape))
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
