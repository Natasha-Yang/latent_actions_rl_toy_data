"""Scripted follow-the-leader expert for SnakeTraceEnv."""
import numpy as np

from .kinematics import chain_from_points, wrap


class FollowTheLeaderExpert:
    """Moves the head `speed` further along the corridor centerline each step and
    places every joint on the centerline behind it, so the body follows the
    exact route the head took (like a train on a track).

    The action is the difference between that target configuration and the
    current one. When a step would exceed the env's per-step action limits
    (e.g. while joints are entering a tight corner), the advance is shortened.
    """

    def __init__(self, speed=0.01, n_bisect=8):
        self.speed = speed
        self.n_bisect = n_bisect

    def target_config(self, env, s_head):
        s = s_head - env.link_len * np.arange(env.n_joints + 1, -1, -1)  # tail -> head
        return chain_from_points(env.corridor.points_at(s))

    def _raw_action(self, env, s_head):
        base, phi, q = self.target_config(env, s_head)
        d = base - env.base
        c, s = np.cos(env.phi), np.sin(env.phi)
        d_local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        return np.concatenate([
            wrap(q - env.q) / env.dq_max,
            d_local / env.dpos_max,
            [wrap(phi - env.phi) / env.dphi_max],
        ])

    def act(self, env):
        # never target past the end: clamped points would coincide and give the head links a garbage heading
        step = max(0.0, min(self.speed, env.corridor.length - env.s_head))
        a = self._raw_action(env, env.s_head + step)
        if np.abs(a).max() > 1.0:
            lo, hi = 0.0, step
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo + hi)
                if np.abs(self._raw_action(env, env.s_head + mid)).max() <= 1.0:
                    lo = mid
                else:
                    hi = mid
            a = self._raw_action(env, env.s_head + lo)
        return np.clip(a, -1.0, 1.0)
