"""Single-stroke letter and digit centerlines.

Each letter is one continuous (M, 2) polyline in a [0, 1] x [0, 1] box, y up,
drawn in the direction the snake travels. Only letters that can be drawn in
one line without reversing on themselves are included.
"""
import numpy as np


def _arc(cx, cy, rx, ry, a0, a1, step_deg=2.0):
    """Elliptical arc from angle a0 to a1 (degrees, either direction)."""
    n = max(4, int(np.ceil(abs(a1 - a0) / step_deg)))
    a = np.deg2rad(np.linspace(a0, a1, n + 1))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=-1)


def _cat(*parts):
    return np.concatenate([np.atleast_2d(np.asarray(p, dtype=np.float64)) for p in parts], axis=0)


LETTERS = {
    "C": _arc(0.5, 0.5, 0.5, 0.5, 45, 315),
    "D": _cat([(0, 0), (0, 1), (0.5, 1)], _arc(0.5, 0.5, 0.5, 0.5, 90, -90), [(0, 0)]),
    "G": _cat(_arc(0.5, 0.5, 0.5, 0.5, 45, 360), [(0.55, 0.5)]),
    "I": _cat([(0.5, 1), (0.5, 0)]),
    "J": _cat([(1, 1)], _arc(0.6, 0.3, 0.4, 0.3, 0, -180)),
    "L": _cat([(0, 1), (0, 0), (1, 0)]),
    "M": _cat([(0, 0), (0, 1), (0.5, 0.4), (1, 1), (1, 0)]),
    "N": _cat([(0, 0), (0, 1), (1, 0), (1, 1)]),
    "O": _arc(0.5, 0.5, 0.5, 0.5, 90, 450),
    "S": _cat(_arc(0.5, 0.75, 0.45, 0.25, 20, 270), _arc(0.5, 0.25, 0.45, 0.25, 90, -160)),
    "U": _cat([(0, 1)], _arc(0.5, 0.4, 0.5, 0.4, 180, 360), [(1, 1)]),
    "V": _cat([(0, 1), (0.5, 0), (1, 1)]),
    "W": _cat([(0, 1), (0.25, 0), (0.5, 0.6), (0.75, 0), (1, 1)]),
    "Z": _cat([(0, 1), (1, 1), (0, 0), (1, 0)]),
}

def _figure_eight(n=361):
    """Smooth figure-eight: starts at the top heading left, crosses itself in the middle."""
    t = np.linspace(0.0, 2 * np.pi, n)
    return np.stack([0.5 - 0.42 * np.sin(2 * t), 0.5 + 0.5 * np.cos(t)], axis=-1)


# Digits drawn in one stroke without reversing on themselves (4 needs two strokes).
# 6 and 9 leave their loops slightly open so the head doesn't end on its own body;
# 9 is drawn stem-first, since loop-first would need a reversal at the junction.
DIGITS = {
    "1": _cat([(0.2, 0.7), (0.6, 1), (0.6, 0)]),
    "2": _cat(_arc(0.5, 0.7, 0.42, 0.3, 160, -30), [(0, 0), (1, 0)]),
    "3": _cat([(0.05, 1), (0.9, 1), (0.45, 0.58)], _arc(0.45, 0.29, 0.45, 0.29, 90, -140)),
    "5": _cat([(0.9, 1), (0.2, 1)], _arc(0.48, 0.32, 0.42, 0.32, 130, -140)),
    "6": _cat(_arc(0.55, 0.5, 0.45, 0.5, 60, 180), [(0.1, 0.3)], _arc(0.45, 0.3, 0.35, 0.3, 180, 480)),
    "7": _cat([(0, 1), (1, 1), (0.35, 0)]),
    "8": _figure_eight(),
    "9": _cat([(0.85, 0)], _arc(0.5, 0.7, 0.35, 0.3, 0, 310)),
}

GLYPHS = {**LETTERS, **DIGITS}
SUPPORTED = "".join(sorted(DIGITS)) + "".join(sorted(LETTERS))


def letter_polyline(letter, scale=1.0, center=(0.0, 0.0)):
    """The glyph's centerline in world coordinates (glyph box is scale x scale)."""
    key = letter.upper() if isinstance(letter, str) else letter
    if not isinstance(key, str) or key not in GLYPHS:
        raise ValueError(f"Unsupported character {letter!r}; supported single-stroke characters: {SUPPORTED}")
    return np.asarray(center, dtype=np.float64) + scale * (GLYPHS[key] - 0.5)


def resample(polyline, spacing):
    """Resample a polyline at uniform arc-length spacing (endpoints kept)."""
    pts = np.asarray(polyline, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    pts, seg = pts[keep], seg[seg > 1e-12]
    if len(pts) < 2:
        return pts[:1].copy()
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(1, int(np.ceil(s[-1] / spacing)))
    s_new = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])], axis=-1)
