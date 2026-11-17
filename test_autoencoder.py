"""Roll out a trained action autoencoder on the tasks in cfgs/test_autoencoder.yaml.

For every test episode (task, glyph scale) two rollouts start from the same initial state:

    expert   the scripted FollowTheLeaderExpert acts directly;
    decoded  at every step the expert's action for the *current* state is passed through the
             autoencoder, decode(encode(a)), and that reconstruction is executed instead.

By default the decoded rollout is closed loop: the expert is queried on the states the decoded
actions lead to, so it can correct small reconstruction errors. Drift shows up only if the
autoencoder can't represent the corrective actions. With --teacher-forcing the decoded env is
instead reset to the expert rollout's state before every step, so each decoded frame is a
one-step prediction from the expert trajectory and errors can't compound.
VAEs are run deterministically (z = mu).

Outputs, in <out_dir>/<run>_<checkpoint>[_tf]/:
    <TASK>_ep<k>.mp4   side-by-side video: expert (left), decoded (right), same view
    summary.png        one panel per episode: letter, both head paths, final decoded body
    metrics.json       per-episode metrics for both rollouts + action reconstruction MSE

The env and expert settings (n_joints, expert speed, max_steps rule) are taken from the config stored
in the checkpoint's training dataset, so the action space matches what the autoencoder was trained on.

    python test_autoencoder.py checkpoints/Autoencoder_O-8-S_37v19gj7            # uses best.pt
    python test_autoencoder.py checkpoints/Autoencoder_O_sogf48ny/last.pt --no-video
    python test_autoencoder.py checkpoints/Autoencoder_O-8-S_37v19gj7 --teacher-forcing
    python test_autoencoder.py <ckpt> --config cfgs/test_autoencoder.yaml --frame-skip 4 --workers 8

Config: required `tasks`, `episodes_per_task`, `scale` (same format as cfgs/data.yaml). Optional:
`seed` (default 1; the demos used 0, and the same seed would draw the same scales as the training demos).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import yaml

import model.autoencoder as autoencoders
from data.collect_demos import DEFAULTS as DATA_DEFAULTS
from data.collect_demos import _merge, make_env, sample_scales
from snake.expert import FollowTheLeaderExpert
from snake.render import render_frames, save_video, snapshot, view_bounds

ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ("tasks", "episodes_per_task", "scale")
DEFAULT_SEED = 1


# ---------------------------------------------------------------- config / checkpoint
def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    cfg["tasks"] = [str(t).upper() for t in cfg["tasks"]]  # YAML reads `8` as an int
    cfg.setdefault("seed", DEFAULT_SEED)
    return cfg


def checkpoint_path(path):
    path = resolve(path)
    if os.path.isdir(path):
        path = os.path.join(path, "best.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = getattr(autoencoders, ckpt["model"])(**ckpt["model_kwargs"])
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt


def data_config(train_cfg):
    """Env / expert settings the training demos were collected with (stored in the dataset)."""
    dataset = resolve(train_cfg["dataset"])
    with np.load(dataset) as d:
        stored = json.loads(str(d["config"])) if "config" in d.files else {}
    return _merge(DATA_DEFAULTS, stored)


# ---------------------------------------------------------------- rollouts
class Reconstructor:
    """a -> decode(encode(a)) for one action, on CPU (z = mu for a VAE)."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def __call__(self, action):
        z = self.net.encode(torch.as_tensor(action, dtype=torch.float32)[None])
        if isinstance(z, tuple):  # VariationalAutoencoder: (mu, logvar)
            z = z[0]
        return self.net.decode(z)[0].numpy().astype(np.float64)


def rollout(env, expert, reconstruct=None):
    """Run one episode. Returns snapshots, the expert actions queried at each step, the actions executed."""
    env.reset()
    snaps, expert_actions, executed = [snapshot(env)], [], []
    done = False
    while not done:
        a = expert.act(env)
        a_exec = a if reconstruct is None else reconstruct(a)
        _, _, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated
        snaps.append(snapshot(env))
        expert_actions.append(a)
        executed.append(a_exec)
    return snaps, np.asarray(expert_actions), np.asarray(executed)


def rollout_teacher_forced(env_exp, env_dec, expert, reconstruct):
    """Expert and decoded envs in lockstep; before every step the decoded env is reset to the expert's
    state, so each decoded snapshot is a one-step prediction from the expert trajectory.
    Returns (expert snaps, decoded snaps, expert actions, reconstructed actions)."""
    env_exp.reset()
    env_dec.reset()
    snaps_exp, snaps_dec, expert_actions, executed = [snapshot(env_exp)], [snapshot(env_dec)], [], []
    done = False
    while not done:
        env_dec.set_state(env_exp.get_state())
        a = expert.act(env_exp)
        a_rec = reconstruct(a)
        env_dec.step(a_rec)
        _, _, terminated, truncated, _ = env_exp.step(a)
        done = terminated or truncated
        snaps_exp.append(snapshot(env_exp))
        snaps_dec.append(snapshot(env_dec))
        expert_actions.append(a)
        executed.append(a_rec)
    return snaps_exp, snaps_dec, np.asarray(expert_actions), np.asarray(executed)


_WORKER = {}


def _init_worker(ckpt_path):
    torch.set_num_threads(1)
    net, _ = load_model(ckpt_path)
    _WORKER["reconstruct"] = Reconstructor(net)


def run_episode(job):
    name, task, scale, seed, data_cfg, teacher_forcing, video_path, frame_skip, fps = job
    reconstruct = _WORKER["reconstruct"]
    envs = {k: make_env(task, scale, seed, data_cfg) for k in ("expert", "decoded")}
    expert = FollowTheLeaderExpert(**data_cfg["expert"])

    if teacher_forcing:
        snaps_exp, snaps_dec, a_exp, a_rec = rollout_teacher_forced(envs["expert"], envs["decoded"], expert,
                                                                     reconstruct)
        rollouts = {"expert": (snaps_exp, a_exp, a_rec), "decoded": (snaps_dec, a_exp, a_rec)}
    else:
        rollouts = {}
        for kind in ("expert", "decoded"):
            snaps, a_exp, a_exec = rollout(envs[kind], expert, reconstruct if kind == "decoded" else None)
            # for "expert", reconstruct the clean demo-like actions offline; for "decoded" the executed
            # actions are already the reconstructions of the expert actions at the states they led to
            a_rec = a_exec if kind == "decoded" else np.stack([reconstruct(a) for a in a_exp])
            rollouts[kind] = (snaps, a_exp, a_rec)

    runs = {}
    for kind, (snaps, a_exp, a_rec) in rollouts.items():
        m = envs[kind].metrics()
        m["action_mse"] = float(np.mean((a_rec - a_exp) ** 2))
        runs[kind] = {"env": envs[kind], "snaps": snaps, "metrics": m}
    # head distance between the two rollouts at the same step (over the steps both ran)
    h_exp, h_dec = (np.asarray(envs[k].head_history) for k in ("expert", "decoded"))
    n = min(len(h_exp), len(h_dec))
    runs["decoded"]["metrics"]["head_err"] = float(np.linalg.norm(h_exp[:n] - h_dec[:n], axis=1).mean())

    if video_path is not None:
        exp, dec = runs["expert"], runs["decoded"]
        bounds = view_bounds(exp["env"], exp["snaps"], dec["snaps"])
        n = max(len(exp["snaps"]), len(dec["snaps"]))
        pad = lambda s: s + [s[-1]] * (n - len(s))  # noqa: E731  hold the shorter rollout's last frame
        left = render_frames(exp["env"], pad(exp["snaps"]), frame_skip=frame_skip, bounds=bounds, label="expert")
        right = render_frames(dec["env"], pad(dec["snaps"]), frame_skip=frame_skip, bounds=bounds,
                              label="decoded (TF)" if teacher_forcing else "decoded")
        video_path = save_video([np.concatenate(f, axis=1) for f in zip(left, right)], video_path, fps=fps)

    cor = runs["expert"]["env"].corridor
    left_line, right_line = cor.boundary_lines()
    return {
        "name": name, "task": task, "scale": scale, "seed": seed, "video": video_path,
        "metrics": {k: r["metrics"] for k, r in runs.items()},
        # for the summary plot
        "outline": np.concatenate([left_line, right_line[::-1], left_line[:1]]),
        "heads": {k: np.asarray(r["env"].head_history) for k, r in runs.items()},
        "final_body": {k: r["snaps"][-1]["joints"] for k, r in runs.items()},
    }


# ---------------------------------------------------------------- reporting
def plot_summary(results, tasks, n_per, path, title):
    import matplotlib.pyplot as plt  # snake.render already selected the Agg backend

    fig, axes = plt.subplots(len(tasks), n_per, figsize=(2.6 * n_per, 2.8 * len(tasks)), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for r in results:
        i, j = tasks.index(r["task"]), int(r["name"].rsplit("ep", 1)[1]) - 1
        ax = axes[i, j]
        ax.plot(*r["outline"].T, color="0.6", lw=0.8)
        ax.plot(*r["heads"]["expert"].T, color="tab:blue", lw=1.2, label="expert head")
        ax.plot(*r["heads"]["decoded"].T, color="tab:orange", lw=1.2, ls="--", label="decoded head")
        ax.plot(*r["final_body"]["decoded"].T, color="tab:orange", lw=2.5, alpha=0.5, label="decoded final body")
        ax.set_aspect("equal")
        md = r["metrics"]["decoded"]
        ax.set_title(f"{r['task']} ep{j + 1}  scale {r['scale']:.2f}\n"
                     f"progress {100 * md['progress']:.0f}%  outside {100 * md['frac_steps_outside']:.0f}%",
                     fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_table(results):
    """mse: action reconstruction MSE; dev mm: max body distance from the centerline;
    head mm: mean head distance from the expert rollout's head at the same step."""
    hdr = (f"{'episode':<10}{'scale':>6} | {'expert':^30} | {'decoded':^46}\n"
           f"{'':<10}{'':>6} | {'steps':>6}{'done':>6}{'prog':>6}{'out%':>6}{'mse':>6} | "
           f"{'steps':>6}{'done':>6}{'prog':>6}{'out%':>6}{'dev mm':>8}{'head mm':>8}{'mse':>6}")
    print(hdr)
    for r in results:
        e, d = r["metrics"]["expert"], r["metrics"]["decoded"]
        print(f"{r['name']:<10}{r['scale']:>6.2f} | {e['steps']:>6}{str(e['completed'])[0]:>6}"
              f"{100 * e['progress']:>5.0f}%{100 * e['frac_steps_outside']:>5.0f}%{e['action_mse']:>6.3f} | "
              f"{d['steps']:>6}{str(d['completed'])[0]:>6}{100 * d['progress']:>5.0f}%"
              f"{100 * d['frac_steps_outside']:>5.0f}%{1000 * d['max_body_dev']:>8.1f}"
              f"{1000 * d['head_err']:>8.1f}{d['action_mse']:>6.3f}")
    for task in dict.fromkeys(r["task"] for r in results):
        d = [r["metrics"]["decoded"] for r in results if r["task"] == task]
        print(f"  {task}: decoded completed {sum(m['completed'] for m in d)}/{len(d)}, "
              f"mean progress {100 * np.mean([m['progress'] for m in d]):.0f}%, "
              f"mean steps-outside {100 * np.mean([m['frac_steps_outside'] for m in d]):.0f}%, "
              f"mean head err {1000 * np.mean([m['head_err'] for m in d]):.1f} mm, "
              f"mean action mse {np.mean([m['action_mse'] for m in d]):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="checkpoint .pt, or a run directory (uses its best.pt)")
    ap.add_argument("--config", default=os.path.join(ROOT, "cfgs", "test_autoencoder.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "videos", "test_autoencoder"))
    ap.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    ap.add_argument("--teacher-forcing", action="store_true",
                    help="reset the decoded rollout to the expert's state before every step (one-step predictions)")
    ap.add_argument("--no-video", action="store_true", help="skip the videos (summary.png and metrics only)")
    ap.add_argument("--frame-skip", type=int, default=2, help="render every k-th step")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ext", default="mp4", choices=("mp4", "gif"))
    ap.add_argument("--workers", type=int, default=None, help="parallel episodes (default: #CPUs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    ckpt_path = checkpoint_path(args.checkpoint)
    _, ckpt = load_model(ckpt_path)  # workers load their own copy
    train_cfg = ckpt["config"]
    data_cfg = data_config(train_cfg)

    # the action space must match: rebuild one env and compare
    probe = make_env(cfg["tasks"][0], 1.0, 0, data_cfg)
    if probe.action_dim != ckpt["model_kwargs"]["action_dim"]:
        raise ValueError(f"env action_dim {probe.action_dim} != autoencoder action_dim "
                         f"{ckpt['model_kwargs']['action_dim']}")
    unseen = [t for t in cfg["tasks"] if t not in train_cfg["tasks"]]
    if unseen:
        print(f"note: tasks {unseen} were not in the autoencoder's training tasks {train_cfg['tasks']}")

    tasks, n_per = cfg["tasks"], int(cfg["episodes_per_task"])
    n_eps = len(tasks) * n_per
    rng = np.random.default_rng(cfg["seed"])
    ep_task = np.repeat(tasks, n_per)
    ep_scale = sample_scales(cfg["scale"], n_eps, rng)
    ep_seed = rng.integers(2**31, size=n_eps)
    for task in tasks:  # fail fast on letters the snake can't bend around at the smallest scale
        make_env(task, ep_scale[ep_task == task].min(), 0, data_cfg)

    run_name = f"{os.path.basename(os.path.dirname(ckpt_path))}_{os.path.splitext(os.path.basename(ckpt_path))[0]}"
    if args.teacher_forcing:
        run_name += "_tf"
    out_dir = os.path.join(resolve(args.out_dir), run_name)
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for i in range(n_eps):
        name = f"{ep_task[i]}_ep{i % n_per + 1}"
        video = None if args.no_video else os.path.join(out_dir, f"{name}.{args.ext}")
        jobs.append((name, str(ep_task[i]), float(ep_scale[i]), int(ep_seed[i]), data_cfg, args.teacher_forcing,
                     video, args.frame_skip, args.fps))

    workers = min(args.workers or os.cpu_count(), n_eps)
    print(f"{ckpt['model']} (epoch {ckpt.get('epoch')}, trained on {train_cfg['tasks']}) from {ckpt_path}")
    print(f"testing {n_eps} episodes ({n_per} x {tasks}, seed {cfg['seed']}, "
          f"{'teacher-forced' if args.teacher_forcing else 'closed-loop'}) with {workers} workers -> {out_dir}")
    t0 = time.time()
    results = []
    with Pool(workers, initializer=_init_worker, initargs=(ckpt_path,)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            results.append(r)
            print(f"  {k}/{n_eps} {r['name']} done  {time.time() - t0:.0f}s")
    order = {j[0]: i for i, j in enumerate(jobs)}
    results.sort(key=lambda r: order[r["name"]])

    print()
    print_table(results)
    plot_summary(results, tasks, n_per, os.path.join(out_dir, "summary.png"),
                 f"{run_name}: expert vs decode(encode(expert action))"
                 + ("  [teacher-forced: one-step predictions]" if args.teacher_forcing else ""))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"checkpoint": ckpt_path, "config": cfg, "teacher_forcing": args.teacher_forcing,
                   "episodes": [{k: r[k] for k in ("name", "task", "scale", "seed", "video", "metrics")}
                                for r in results]},
                  f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(f"\nsaved summary.png, metrics.json{'' if args.no_video else ' and videos'} -> {out_dir}")


if __name__ == "__main__":
    main()
