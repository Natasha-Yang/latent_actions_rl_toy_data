"""Run the scripted follow-the-leader expert on one letter and save a video.

The snake slithers head-first along the letter, staying between its boundary
lines, so its body takes on the letter's shape (e.g. for 'O' it curls into a circle).

    python test_env.py O
    python test_env.py 8 --snake-length 0.3 --scale 1.5
    python test_env.py s --n-joints 200 --seed 3 --out videos/S.gif
"""
import argparse
import sys
import time

import numpy as np

from snake.env import SnakeTraceEnv
from snake.expert import FollowTheLeaderExpert
from snake.letters import SUPPORTED
from snake.render import render_frames, save_video, snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("letter", help=f"single-stroke letter or digit, one of {SUPPORTED} (case-insensitive)")
    ap.add_argument("--out", default=None, help="output video (.mp4 or .gif); default videos/<LETTER>.mp4")
    ap.add_argument("--n-joints", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snake-length", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=1.0, help="glyph box size (same units as snake length)")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--speed", type=float, default=0.01, help="expert head advance per step")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-skip", type=int, default=2, help="render every k-th step")
    args = ap.parse_args()

    try:
        env = SnakeTraceEnv(args.letter, n_joints=args.n_joints, total_length=args.snake_length,
                            letter_scale=args.scale, max_steps=args.max_steps, seed=args.seed)
    except ValueError as e:
        sys.exit(f"error: {e}")
    expert = FollowTheLeaderExpert(speed=args.speed)

    t0 = time.time()
    obs, info = env.reset()
    snaps = [snapshot(env)]
    ret, max_action = 0.0, 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = expert.act(env)
        max_action = max(max_action, float(np.abs(action).max()))
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        snaps.append(snapshot(env))
    sim_time = time.time() - t0

    m = env.metrics()
    print(f"letter {env.letter}  joints {env.n_joints}  obs_dim {env.obs_dim}  action_dim {env.action_dim}  "
          f"letter length {env.corridor.letter_length:.2f} (snake {env.total_length:.2f})")
    print(f"steps {m['steps']}  completed {m['completed']}  steps with body outside lines {100 * m['frac_steps_outside']:.0f}%  "
          f"progress {100 * m['progress']:.1f}%  return {ret:.2f}  max|a| {max_action:.2f}  sim {sim_time:.2f}s")
    print(f"body deviation from centerline (on-letter part): max {1000 * m['max_body_dev']:.2f} mm, "
          f"lines at ± {1000 * env.corridor.half_width:.0f} mm")

    out = args.out or f"videos/{env.letter}.mp4"
    frames = render_frames(env, snaps, frame_skip=args.frame_skip)
    written = save_video(frames, out, fps=args.fps)
    print(f"saved {len(frames)} frames -> {written}")


if __name__ == "__main__":
    main()
