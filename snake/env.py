"""Kinematic environment: a free snake slithers along a letter, staying between its lines."""
import numpy as np

from .corridor import Corridor
from .kinematics import fk, wrap


class SnakeTraceEnv:
    """A planar snake (N joints, N+1 links, total length `total_length`) starts
    straight with its head at the first point of a letter or digit (glyph box
    `letter_scale` x `letter_scale`), and must slither
    head-first along the letter so that its body follows the letter's shape
    (for 'O', the snake ends up curled in a circle).

    The letter is drawn as two boundary lines `corridor_width` apart. Body
    points that have entered the letter (i.e. lie behind the head by less than
    the distance the head has travelled along it) should stay between them;
    going outside is penalised in the reward but does not end the episode.
    The part of the body not yet on the letter is unconstrained.

    State: tail position (x, y), tail heading phi, joint angles q (N,).
    Action (N+3,) in [-1, 1]:
        a[:N]    joint angle changes,                scaled by dq_max (rad)
        a[N:N+2] tail translation in the tail frame, scaled by dpos_max
        a[N+2]   tail rotation,                      scaled by dphi_max (rad)
    Observation (float32): [sin q, cos q, tail x, y, sin phi, cos phi, head x, y,
        `lookahead` centerline points ahead of the head relative to it (flattened),
        progress].
    Termination: the head reaches the end of the letter. Truncation: max_steps.
    """

    def __init__(
        self,
        letter,
        n_joints=128,
        total_length=0.5,
        joint_limit=0.35,
        dq_max=0.1,
        dpos_max=0.02,
        dphi_max=0.15,
        letter_scale=1.0,
        corridor_width=0.05,
        fillet_radius=0.05,
        max_steps=2000,
        lookahead=8,
        lookahead_spacing=0.04,
        start_offset_max=0.0,
        init_noise=0.0,
        progress_reward=10.0,
        deviation_cost=0.05,
        outside_cost=1.0,
        action_cost=1e-3,
        seed=None,
    ):
        self.letter = letter.upper()
        self.n_joints = n_joints
        self.total_length = total_length
        self.link_len = total_length / (n_joints + 1)
        self.joint_limit = joint_limit
        self.dq_max, self.dpos_max, self.dphi_max = dq_max, dpos_max, dphi_max
        self.max_steps = max_steps
        self.lookahead = lookahead
        self.lookahead_spacing = lookahead_spacing
        self.start_offset_max = start_offset_max
        self.init_noise = init_noise
        self.progress_reward = progress_reward
        self.deviation_cost = deviation_cost
        self.outside_cost = outside_cost
        self.action_cost = action_cost

        self.corridor = Corridor(
            self.letter, scale=letter_scale, lead_in=total_length + 0.2,
            fillet_radius=fillet_radius, half_width=corridor_width / 2,
        )
        bend = self.corridor.max_bend_per_link(self.link_len)
        if bend > 0.95 * joint_limit:
            raise ValueError(
                f"Letter {self.letter!r} needs {bend:.3f} rad per joint but joint_limit={joint_limit}; "
                "use more joints, a larger fillet_radius, or a larger letter_scale."
            )

        # arc length from each body point (tail -> head) back from the head
        self._body_offsets = self.link_len * np.arange(n_joints + 1, -1, -1)
        self.action_dim = n_joints + 3
        self.obs_dim = 2 * n_joints + 4 + 2 + 2 * lookahead + 1
        self.rng = np.random.default_rng(seed)
        self.q = None

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.corridor
        s_head = c.s_letter - self.rng.uniform(0.0, self.start_offset_max)
        tail, head = c.points_at(s_head - self.total_length), c.points_at(s_head)
        self.base = tail
        self.phi = float(np.arctan2(*(head - tail)[::-1]))
        self.q = np.zeros(self.n_joints)
        if self.init_noise > 0:
            self.q = np.clip(self.init_noise * self.rng.standard_normal(self.n_joints),
                             -self.joint_limit, self.joint_limit)
        self.t = 0
        self._update_kinematics()
        self.s_head = c.project(self.head, s_head)
        self._update_deviation()
        self.head_history = [self.head.copy()]
        self.dev_history = [self.max_deviation]
        self.outside_history = [self.n_outside]
        return self._obs(), self._info()

    def step(self, action):
        if self.q is None:
            raise RuntimeError("Call reset() before step().")
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {a.shape}")
        n = self.n_joints

        self.q = np.clip(self.q + self.dq_max * a[:n], -self.joint_limit, self.joint_limit)
        c, s = np.cos(self.phi), np.sin(self.phi)
        dx, dy = self.dpos_max * a[n], self.dpos_max * a[n + 1]
        self.base = self.base + np.array([c * dx - s * dy, s * dx + c * dy])
        self.phi = float(wrap(self.phi + self.dphi_max * a[n + 2]))
        self.t += 1

        prev_progress = self.progress
        self._update_kinematics()
        self.s_head = self.corridor.project(self.head, self.s_head)
        self._update_deviation()
        self.head_history.append(self.head.copy())
        self.dev_history.append(self.max_deviation)
        self.outside_history.append(self.n_outside)

        terminated = self.reached_end
        truncated = (not terminated) and self.t >= self.max_steps
        hw = self.corridor.half_width
        n_pts = len(self.deviation)
        reward = (
            self.progress_reward * (self.progress - prev_progress)
            - self.deviation_cost * float(self.deviation.sum()) / (n_pts * hw)
            - self.outside_cost * float(self.outside_amount.sum()) / (n_pts * hw)
            - self.action_cost * float(np.mean(a**2))
        )
        return self._obs(), reward, terminated, truncated, self._info()

    def get_state(self):
        """Copy of the full simulator state (tail pose, joint angles, step count, head arc length)."""
        return {"base": self.base.copy(), "phi": self.phi, "q": self.q.copy(), "t": self.t, "s_head": self.s_head}

    def set_state(self, state):
        """Restore a state from get_state() (e.g. of another env on the same letter). Histories are kept."""
        self.base = np.array(state["base"], dtype=np.float64)
        self.phi = float(state["phi"])
        self.q = np.array(state["q"], dtype=np.float64)
        self.t = int(state["t"])
        self.s_head = float(state["s_head"])
        self._update_kinematics()
        self._update_deviation()

    # ------------------------------------------------------------ properties
    @property
    def progress(self):
        """Fraction of the letter the head has travelled."""
        c = self.corridor
        if self.reached_end:
            return 1.0
        return float(np.clip((self.s_head - c.s_letter) / c.letter_length, 0.0, 1.0))

    @property
    def reached_end(self):
        return self.s_head >= self.corridor.length - 0.005

    @property
    def max_deviation(self):
        return float(self.deviation.max())

    @property
    def n_outside(self):
        return int((self.outside_amount > 0).sum())

    def metrics(self):
        dev = np.asarray(self.dev_history)
        outside = np.asarray(self.outside_history)
        return {
            "steps": self.t,
            "progress": self.progress,
            "completed": self.reached_end,
            "max_body_dev": float(dev.max()),
            "frac_steps_outside": float(np.mean(outside > 0)),
            "max_abs_q": float(np.abs(self.q).max()),
        }

    # ------------------------------------------------------------- internals
    def _update_kinematics(self):
        self.joint_positions = fk(self.q, self.link_len, self.base, self.phi)
        self.head = self.joint_positions[-1]
        self.tail = self.joint_positions[0]

    def _update_deviation(self):
        """Per body point: distance to the letter centerline (0 for points not yet on
        the letter) and how far beyond the boundary lines it is."""
        c = self.corridor
        self.on_letter = self._body_offsets <= self.s_head - c.s_letter
        self.deviation = np.zeros(len(self.joint_positions))
        if self.on_letter.any():
            lo = max(c.s_letter, self.s_head - self.total_length - 0.1)
            self.deviation[self.on_letter] = c.distance(
                self.joint_positions[self.on_letter], lo, self.s_head + 0.05
            )
        self.outside_amount = np.clip(self.deviation - c.half_width, 0.0, None)

    def _obs(self):
        ahead = self.corridor.points_at(self.s_head + self.lookahead_spacing * np.arange(1, self.lookahead + 1))
        return np.concatenate([
            np.sin(self.q), np.cos(self.q),
            self.base, [np.sin(self.phi), np.cos(self.phi)],
            self.head, (ahead - self.head).ravel(), [self.progress],
        ]).astype(np.float32)

    def _info(self):
        return {
            "t": self.t,
            "head": self.head.copy(),
            "s_head": self.s_head,
            "progress": self.progress,
            "max_deviation": self.max_deviation,
            "n_outside": self.n_outside,
        }
