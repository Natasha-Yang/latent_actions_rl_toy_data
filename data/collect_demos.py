"""Collect expert demonstrations for the tasks in cfgs/data.yaml.

For every task (letter/digit) the scripted FollowTheLeaderExpert is rolled out
`demos_per_task` times, each time on a glyph whose scale is drawn from the
config's `scale` setting. Every step is stored as one row:

    task         (T,)          str      letter/digit being traced, e.g. "O"
    episode      (T,)          int32    1 .. num_tasks * demos_per_task (task-major order)
    step_num     (T,)          int32    0 .. episode_length - 1 within the episode
    observation  (T, H, W)     uint8    grayscale image of the state *before* the action
    action       (T, A)        float32  expert action taken from that state, in [-1, 1]

plus per-episode metadata (`episode_task`, `episode_scale`, `episode_completed`)
and the resolved config as a JSON string (`config`). Load with

    d = np.load("data/demos.npz")
    d["action"], d["observation"], d["task"], ...

Image: black background, the letter's corridor as a dim band, the snake on top
with brightness increasing from tail to head (so the head is identifiable).
`observation.view: letter` frames the whole letter (fixed per episode);
`view: head` is a `view_size` x `view_size` world window centred on the head.

Config (cfgs/data.yaml). Required: tasks, scale, demos_per_task. Optional keys
and their defaults are in DEFAULTS below, e.g.

    scale: {mode: random_uniform, min: 0.5, max: 5}   # or a plain number for a fixed scale
    seed: 0
    observation: {size: 64, view: letter}
    env: {n_joints: 128}         # any SnakeTraceEnv keyword argument
    expert: {speed: 0.01}

    python data/collect_demos.py
    python data/collect_demos.py --config cfgs/data.yaml --out data/demos.npz --workers 16
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snake.env import SnakeTraceEnv  # noqa: E402
from snake.expert import FollowTheLeaderExpert  # noqa: E402

DEFAULTS = {
    "seed": 0,
    "out": "data/demos.npz",  # relative to the repo root
    "observation": {
        "size": 64,          # output image is size x size
        "view": "letter",    # "letter" (whole glyph) or "head" (window around the head)
        "view_size": 1.0,    # world units across, for view: head
        "supersample": 4,    # draw at size * supersample, then box-downsample (anti-aliasing)
    },
    "env": {},               # extra SnakeTraceEnv kwargs; max_steps defaults to 2x the expected length
    "expert": {"speed": 0.01},
}


# ---------------------------------------------------------------- config
def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    missing = [k for k in ("tasks", "scale", "demos_per_task") if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    cfg = _merge(DEFAULTS, raw)
    cfg["tasks"] = [str(t).upper() for t in cfg["tasks"]]  # YAML reads `8` as an int
    if cfg["observation"]["view"] not in ("letter", "head"):
        raise ValueError(f"observation.view must be 'letter' or 'head', got {cfg['observation']['view']!r}")
    return cfg


def sample_scales(scale_cfg, n, rng):
    if isinstance(scale_cfg, (int, float)):
        return np.full(n, float(scale_cfg))
    mode = scale_cfg.get("mode")
    if mode == "random_uniform":
        lo, hi = float(scale_cfg["min"]), float(scale_cfg["max"])
        if not 0 < lo <= hi:
            raise ValueError(f"scale needs 0 < min <= max, got min={lo}, max={hi}")
        return rng.uniform(lo, hi, size=n)
    if mode == "fixed":
        return np.full(n, float(scale_cfg["value"]))
    raise ValueError(f"Unknown scale mode {mode!r}; use 'random_uniform', 'fixed', or a number")


# -------------------------------------------------------------- rendering
class Renderer:
    """Rasterises the letter corridor and the snake into a (size, size) uint8 image.

    The corridor is drawn once per episode onto a large background covering the
    whole letter; each frame crops the view out of it and draws the snake on top.
    """

    CORRIDOR, TAIL, HEAD = 70, 140, 255
    N_BODY_CHUNKS = 16  # body is drawn as this many polylines of increasing brightness

    def __init__(self, env, size=64, view="letter", view_size=1.0, supersample=4):
        c = env.corridor
        self.size, self.ss = size, supersample
        self.canvas = size * supersample
        self.view = view

        hw = c.half_width
        lead = c.points_at(np.linspace(c.s_letter - env.total_length, c.s_letter, 32))
        pts = np.concatenate([c.letter_points, lead])  # everywhere the snake can be
        lo, hi = pts.min(0) - 3 * hw, pts.max(0) + 3 * hw
        if view == "letter":
            span = 1.05 * (hi - lo).max()
            centre = (lo + hi) / 2
            lo, hi = centre - span / 2, centre + span / 2
        else:
            span = view_size
            lo, hi = lo - span / 2, hi + span / 2  # room to crop around a head near the edge
        self.ppu = self.canvas / span  # canvas pixels per world unit
        self.x0, self.y1 = lo[0], hi[1]
        self.view_px = self.canvas

        min_w = supersample  # at least one output pixel wide
        self.corridor_w = max(min_w, int(round(2 * hw * self.ppu)))
        self.body_w = max(min_w, int(round(1.2 * hw * self.ppu)))

        wh = np.ceil((hi - lo) * self.ppu).astype(int)
        self.background = Image.new("L", (int(wh[0]), int(wh[1])), 0)
        # decimate the dense centerline to ~1 canvas pixel spacing
        step = max(1, int(1.0 / (self.ppu * c.spacing)))
        letter = np.concatenate([c.letter_points[::step], c.letter_points[-1:]])
        ImageDraw.Draw(self.background).line(self._px(letter), fill=self.CORRIDOR,
                                             width=self.corridor_w, joint="curve")

    def _px(self, pts, offset=(0.0, 0.0)):
        x = (pts[:, 0] - self.x0) * self.ppu - offset[0]
        y = (self.y1 - pts[:, 1]) * self.ppu - offset[1]
        return list(map(tuple, np.stack([x, y], axis=1)))

    def render(self, env):
        if self.view == "letter":
            off = (0, 0)
            img = self.background.crop((0, 0, self.canvas, self.canvas))
        else:
            hx = (env.head[0] - self.x0) * self.ppu
            hy = (self.y1 - env.head[1]) * self.ppu
            off = (int(round(hx)) - self.canvas // 2, int(round(hy)) - self.canvas // 2)
            img = self.background.crop((off[0], off[1], off[0] + self.canvas, off[1] + self.canvas))

        draw = ImageDraw.Draw(img)
        joints = self._px(env.joint_positions, off)  # tail -> head
        bounds = np.linspace(0, len(joints) - 1, self.N_BODY_CHUNKS + 1).round().astype(int)
        shades = np.linspace(self.TAIL, self.HEAD, self.N_BODY_CHUNKS).round().astype(int)
        for i0, i1, shade in zip(bounds[:-1], bounds[1:], shades):
            draw.line(joints[i0:i1 + 1], fill=int(shade), width=self.body_w, joint="curve")
        hx, hy = joints[-1]
        r = 0.8 * self.body_w
        draw.ellipse((hx - r, hy - r, hx + r, hy + r), fill=self.HEAD)
        return np.asarray(img.reduce(self.ss), dtype=np.uint8)


# ------------------------------------------------------------- collection
def make_env(task, scale, seed, cfg):
    env = SnakeTraceEnv(task, letter_scale=scale, seed=seed, **cfg["env"])
    if "max_steps" not in cfg["env"]:
        # the expert advances `speed` per step (less in tight corners); leave 2x headroom
        env.max_steps = int(np.ceil(2 * env.corridor.letter_length / cfg["expert"]["speed"])) + 100
    return env


def run_episode(job):
    episode, task, scale, seed, cfg = job
    env = make_env(task, scale, seed, cfg)
    expert = FollowTheLeaderExpert(**cfg["expert"])
    renderer = Renderer(env, **cfg["observation"])

    env.reset()
    observations, actions = [], []
    done = False
    while not done:
        observations.append(renderer.render(env))
        action = expert.act(env)
        actions.append(action.astype(np.float32))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return episode, np.stack(observations), np.stack(actions), env.metrics()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "cfgs", "data.yaml"))
    ap.add_argument("--out", default=None, help="output .npz (default: config `out`, relative to repo root)")
    ap.add_argument("--workers", type=int, default=None, help="parallel episodes (default: #CPUs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or os.path.join(ROOT, cfg["out"])
    tasks, n_per = cfg["tasks"], int(cfg["demos_per_task"])
    n_eps = len(tasks) * n_per

    rng = np.random.default_rng(cfg["seed"])
    ep_task = np.repeat(tasks, n_per)
    ep_scale = sample_scales(cfg["scale"], n_eps, rng)
    ep_seed = rng.integers(2**31, size=n_eps)

    # fail fast on letters the snake can't bend around (worst case: the smallest scale)
    for task in tasks:
        make_env(task, ep_scale[ep_task == task].min(), 0, cfg)

    jobs = [(i + 1, ep_task[i], float(ep_scale[i]), int(ep_seed[i]), cfg) for i in range(n_eps)]
    workers = min(args.workers or os.cpu_count(), n_eps)
    print(f"collecting {n_eps} demos ({n_per} x {tasks}) with {workers} workers")

    results = {}
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (ep, obs, act, metrics) in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            results[ep] = (obs, act, metrics)
            if not metrics["completed"]:
                print(f"  warning: episode {ep} ({ep_task[ep - 1]}, scale {ep_scale[ep - 1]:.2f}) "
                      f"truncated at {metrics['steps']} steps, progress {100 * metrics['progress']:.0f}%")
            if k % max(1, n_eps // 20) == 0 or k == n_eps:
                print(f"  {k}/{n_eps} episodes  {time.time() - t0:.0f}s")

    order = range(1, n_eps + 1)
    lengths = np.array([len(results[ep][1]) for ep in order])
    data = {
        "task": np.repeat(ep_task, lengths),
        "episode": np.repeat(np.arange(1, n_eps + 1, dtype=np.int32), lengths),
        "step_num": np.concatenate([np.arange(n, dtype=np.int32) for n in lengths]),
        "observation": np.concatenate([results[ep][0] for ep in order]),
        "action": np.concatenate([results[ep][1] for ep in order]),
        # per-episode metadata (index = episode - 1)
        "episode_task": ep_task,
        "episode_scale": ep_scale.astype(np.float32),
        "episode_completed": np.array([results[ep][2]["completed"] for ep in order]),
        "config": np.array(json.dumps(cfg)),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"saved {len(data['action'])} steps from {n_eps} episodes -> {out} "
          f"({os.path.getsize(out) / 1e6:.0f} MB)")
    print(f"  observation {data['observation'].shape} {data['observation'].dtype}, "
          f"action {data['action'].shape} {data['action'].dtype}, "
          f"episode length {lengths.min()}-{lengths.max()} (mean {lengths.mean():.0f}), "
          f"completed {data['episode_completed'].sum()}/{n_eps}")


if __name__ == "__main__":
    main()
