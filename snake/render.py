"""Rendering of SnakeTraceEnv rollouts to frames and video files."""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402


def snapshot(env):
    """The per-step state needed to draw a frame."""
    return {
        "joints": env.joint_positions.copy(),
        "t": env.t,
        "s_head": env.s_head,
        "progress": env.progress,
        "max_dev": env.max_deviation,
        "outside": env.outside_amount > 0,
    }


def _view_bounds(snapshots, letter_pts, margin=0.06):
    pts = np.concatenate([letter_pts] + [s["joints"] for s in snapshots])
    lo, hi = pts.min(0), pts.max(0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 + margin
    return (c[0] - half, c[0] + half), (c[1] - half, c[1] + half)


def view_bounds(env, *snapshot_lists):
    """(xlim, ylim) square view covering the letter and every snapshot in the given rollouts,
    e.g. to draw several rollouts of the same letter at the same zoom."""
    return _view_bounds([s for snaps in snapshot_lists for s in snaps], env.corridor.letter_points)


def render_frames(env, snapshots, frame_skip=2, size_px=640, bounds=None, label=None):
    """Draw RGB frames for a recorded rollout.

    bounds: optional (xlim, ylim) view (default: fit this rollout); label: optional title prefix.
    """
    cor = env.corridor
    i_letter = int(np.searchsorted(cor.s, cor.s_letter))
    letter_pts = cor.letter_points
    left, right = cor.boundary_lines()
    outline = np.concatenate([left, right[::-1], left[:1]])  # closed outline incl. end caps
    xlim, ylim = bounds if bounds is not None else _view_bounds(snapshots, letter_pts)

    dpi = 100
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.93)
    ax_px = 0.96 * size_px
    pts_per_unit = ax_px / (xlim[1] - xlim[0]) * 72.0 / dpi
    body_lw = max(2.0, 1.2 * cor.half_width * pts_per_unit)
    cmap = plt.get_cmap("viridis")

    idxs = list(range(0, len(snapshots), frame_skip))
    if idxs[-1] != len(snapshots) - 1:
        idxs.append(len(snapshots) - 1)

    frames = []
    for t in idxs:
        snap = snapshots[t]
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # letter: boundary lines the body should stay between, dashed centerline
        ax.plot(*outline.T, color="0.3", lw=1.2, zorder=2)
        ax.plot(*letter_pts.T, color="0.7", lw=0.8, ls="--", zorder=3)
        done = cor.points[i_letter : max(i_letter, int(np.searchsorted(cor.s, snap["s_head"]))) + 1]
        if len(done) > 1:
            ax.plot(*done.T, color="tab:orange", lw=1.2, alpha=0.7, zorder=3)

        j = snap["joints"]
        body = np.stack([j[:-1], j[1:]], axis=1)
        ax.add_collection(LineCollection(body, colors=cmap(np.linspace(0.1, 0.9, len(body))),
                                         linewidths=body_lw, capstyle="round", zorder=4))
        if snap["outside"].any():
            ax.plot(*j[snap["outside"]].T, "o", color="tab:red", ms=3, zorder=5)
        ax.plot(*j[-1], "o", color="tab:orange", mec="k", ms=body_lw * 0.9, zorder=6)

        ax.set_title(f"{label + '  ' if label else ''}'{env.letter}'  N={env.n_joints}  step {snap['t']}  "
                     f"progress {100 * snap['progress']:.0f}%  max dev {1000 * snap['max_dev']:.1f} mm",
                     fontsize=11)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


def save_video(frames, out_path, fps=30):
    """Write frames to .mp4 (needs imageio + imageio-ffmpeg) or .gif (Pillow).

    Falls back to .gif if an mp4 writer is unavailable. Returns the path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    root, ext = os.path.splitext(out_path)
    ext = ext.lower()

    if ext == ".mp4":
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401

            imageio.mimsave(out_path, frames, fps=fps, macro_block_size=16)
            return out_path
        except ImportError:
            warnings.warn("imageio/imageio-ffmpeg not installed; writing a GIF instead. "
                          "Install with: python -m pip install imageio imageio-ffmpeg")
            out_path, ext = root + ".gif", ".gif"

    if ext != ".gif":
        raise ValueError(f"Unsupported video extension {ext!r}; use .mp4 or .gif")

    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / fps)), loop=0)
    return out_path
