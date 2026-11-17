"""The letter a snake slithers along: a centerline with rounded corners, and
boundary lines `half_width` to either side that the snake should stay between.

Internally the centerline is extended backwards by a straight approach line
(s < s_letter) so the part of the body that has not yet entered the letter has
somewhere to be; that line is not part of the letter and is not scored."""
import numpy as np

from .kinematics import wrap
from .letters import letter_polyline, resample


def fillet_polyline(poly, radius, spacing, min_turn_deg=15.0):
    """Replace every sharp vertex (turn > min_turn_deg) with a ~circular arc of
    `radius` (shrunk where neighbouring corners leave too little room), then
    resample uniformly at `spacing`."""
    poly = np.asarray(poly, dtype=np.float64)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    poly, seg, seg_len = poly[np.r_[True, seg_len > 1e-12]], seg[seg_len > 1e-12], seg_len[seg_len > 1e-12]
    s_v = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = s_v[-1]
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = wrap(np.diff(heading))  # turn[i-1] is the turn at vertex i
    corners = [i for i in range(1, len(poly) - 1) if abs(turn[i - 1]) > np.deg2rad(min_turn_deg)]

    def point(s):
        return np.stack([np.interp(s, s_v, poly[:, 0]), np.interp(s, s_v, poly[:, 1])], axis=-1)

    def tangent(s):
        j = np.clip(np.searchsorted(s_v, s, side="right") - 1, 0, len(seg) - 1)
        return seg[j] / seg_len[j]

    pieces, cur = [], 0.0
    for k, i in enumerate(corners):
        prev_s, fp = (s_v[corners[k - 1]], 0.5) if k > 0 else (0.0, 0.9)
        next_s, fn = (s_v[corners[k + 1]], 0.5) if k + 1 < len(corners) else (total, 0.9)
        t = min(radius * np.tan(abs(turn[i - 1]) / 2), fp * (s_v[i] - prev_s), fn * (next_s - s_v[i]))
        a, b = s_v[i] - t, s_v[i] + t
        if a > cur:
            pieces.append(point(np.arange(cur, a, spacing)))
        A, B, dA, dB = point(a), point(b), tangent(a - 1e-9), tangent(b + 1e-9)
        theta = abs(wrap(np.arctan2(dB[1], dB[0]) - np.arctan2(dA[1], dA[0])))
        R = np.linalg.norm(B - A) / (2 * np.sin(theta / 2))
        h = 4.0 / 3.0 * np.tan(theta / 4) * R  # cubic Bezier approximation of a circular arc
        c1, c2 = A + h * dA, B - h * dB
        u = np.linspace(0.0, 1.0, max(8, int(np.ceil(R * theta / spacing))), endpoint=False)[:, None]
        pieces.append((1 - u) ** 3 * A + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u**2 * c2 + u**3 * B)
        cur = b
    pieces.append(point(np.linspace(cur, total, max(2, int(np.ceil((total - cur) / spacing)) + 1))))
    return resample(np.concatenate(pieces), spacing)


class Corridor:
    """Dense centerline: approach line (s in [0, s_letter]) then the letter
    (s in [s_letter, length]). Body points on the letter should stay within
    `half_width` of the centerline."""

    def __init__(self, letter, scale=1.0, center=(0.0, 0.0), lead_in=0.7,
                 fillet_radius=0.05, half_width=0.025, spacing=0.001):
        self.letter = letter.upper()
        self.half_width = half_width
        self.spacing = spacing
        letter_pts = fillet_polyline(letter_polyline(letter, scale, center), fillet_radius, spacing)

        start = letter_pts[0]
        d0 = letter_pts[min(10, len(letter_pts) - 1)] - start
        d0 /= np.linalg.norm(d0)
        n_lead = int(np.ceil(lead_in / spacing))
        lead = start - d0 * np.linspace(lead_in, 0.0, n_lead + 1)[:-1, None]

        self.points = np.concatenate([lead, letter_pts])
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))])
        self.s_letter = self.s[n_lead]
        self.length = self.s[-1]
        self.letter_length = self.length - self.s_letter
        self.letter_points = self.points[n_lead:]

    def boundary_lines(self):
        """Left and right boundary lines of the letter (each (M, 2))."""
        p = self.letter_points
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)
        return p + self.half_width * n, p - self.half_width * n

    def points_at(self, s):
        s = np.clip(s, 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.points[:, 0]),
                         np.interp(s, self.s, self.points[:, 1])], axis=-1)

    def _window(self, s_lo, s_hi):
        i0 = max(0, np.searchsorted(self.s, s_lo) - 1)
        i1 = min(len(self.s), np.searchsorted(self.s, s_hi) + 1)
        return i0, i1

    def project(self, p, s_guess, radius=0.03):
        """Arc length of the centerline point nearest to p, searched near s_guess."""
        i0, i1 = self._window(s_guess - radius, s_guess + radius)
        d = np.linalg.norm(self.points[i0:i1] - p, axis=1)
        return self.s[i0 + int(np.argmin(d))]

    def distance(self, pts, s_lo, s_hi):
        """Distance from each point to the centerline section s in [s_lo, s_hi]."""
        i0, i1 = self._window(s_lo, s_hi)
        c = self.points[i0:i1]
        return np.sqrt(((pts[:, None, :] - c[None, :, :]) ** 2).sum(-1).min(axis=1))

    def max_bend_per_link(self, link_len):
        """Largest joint angle needed for a chain of `link_len` links lying on the centerline."""
        p = self.points_at(np.arange(0.0, self.length, link_len))
        d = np.diff(p, axis=0)
        return float(np.abs(wrap(np.diff(np.arctan2(d[:, 1], d[:, 0])))).max())
