"""Sanity tests. Run with pytest, or directly: python tests/test_kinematics.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake.env import SnakeTraceEnv  # noqa: E402
from snake.expert import FollowTheLeaderExpert  # noqa: E402
from snake.kinematics import chain_from_points, fk  # noqa: E402
from snake.letters import SUPPORTED, letter_polyline  # noqa: E402


def test_fk_straight_chain():
    n, L = 128, 1.0
    pts = fk(np.zeros(n), L / (n + 1), base=(0.0, 0.0), base_angle=np.pi / 2)
    assert pts.shape == (n + 2, 2)
    np.testing.assert_allclose(pts[-1], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(pts, axis=0), axis=1), L / (n + 1))


def test_chain_from_points_inverts_fk():
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.3, 0.3, 64)
    pts = fk(q, 0.01, base=(0.2, -0.1), base_angle=1.0)
    base, phi, q2 = chain_from_points(pts)
    np.testing.assert_allclose(base, [0.2, -0.1])
    np.testing.assert_allclose(phi, 1.0)
    np.testing.assert_allclose(q2, q, atol=1e-12)


def test_unsupported_letter_raises():
    for bad in ("A", "4", "0", "ab"):
        try:
            letter_polyline(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_corridors_are_feasible():
    for c in SUPPORTED:
        env = SnakeTraceEnv(c)  # raises if the glyph needs more bend than joint_limit
        cor = env.corridor
        steps = np.linalg.norm(np.diff(cor.points, axis=0), axis=1)
        assert steps.max() < 2 * cor.spacing, c
        assert cor.max_bend_per_link(env.link_len) < 0.95 * env.joint_limit, c


def test_expert_follows_all_letters():
    for c in SUPPORTED:
        for seed in (0, 1):
            env = SnakeTraceEnv(c, seed=seed)
            expert = FollowTheLeaderExpert()
            env.reset()
            done = False
            while not done:
                obs, r, term, trunc, _ = env.step(expert.act(env))
                assert np.all(np.isfinite(obs))
                done = term or trunc
            m = env.metrics()
            assert m["completed"] and m["frac_steps_outside"] == 0, (c, seed, m)
            assert m["max_body_dev"] < 0.1 * env.corridor.half_width, (c, seed, m)


def test_leaving_lines_is_penalised_not_terminal():
    env = SnakeTraceEnv("O", seed=0)
    expert = FollowTheLeaderExpert()
    env.reset()
    for _ in range(100):  # get part of the body onto the letter
        env.step(expert.act(env))
    a = np.zeros(env.action_dim)
    a[: env.n_joints] = 1.0  # curl hard -> body leaves the lines
    _, r, term, trunc, info = env.step(a)
    assert info["n_outside"] > 0 and not term and not trunc and r < 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
