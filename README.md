# Snake letter mazes (latent-action RL toy)

A free-floating planar snake with many joints (default 128) slithers head-first along a letter or digit,
staying between its boundary lines, so its body follows the glyph's shape (e.g. for `O` it goes around a
circle). By default the glyph is 1.0 x 1.0 and the snake is 0.5 long.

```
snake/
  letters.py     # single-stroke centerlines: digits 1 2 3 5 6 7 8 9, letters C D G I J L M N O S U V W Z
  corridor.py    # letter centerline with rounded corners, boundary lines; projection / distance queries
  kinematics.py  # fk(q, tail pose) -> body points; chain_from_points (inverse)
  env.py         # SnakeTraceEnv(letter, ...): reset() / step(action), gymnasium-style returns
  expert.py      # FollowTheLeaderExpert: every joint follows the head's route
  render.py      # rollout -> frames -> .mp4 / .gif
test_env.py      # run the expert on a letter and save a video
tests/           # python tests/test_kinematics.py (or pytest)
```

## Usage
```
python test_env.py O                      # -> videos/O.mp4 (GIF if imageio-ffmpeg is missing)
python test_env.py 8 --snake-length 0.3 --scale 1.5 --out videos/8.gif
```

```python
from snake.env import SnakeTraceEnv
env = SnakeTraceEnv("O", n_joints=128, seed=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action in [-1, 1]^(n_joints + 3)
```

- **State**: tail position (x, y), tail heading `phi`, joint angles `q` (N,), each the bend of a link relative to
  the previous one. N joints, N+1 links, total length 1.
- **Action** (N+3): `a[:N]` joint angle changes (× `dq_max`), `a[N:N+2]` tail translation in the tail frame
  (× `dpos_max`), `a[N+2]` tail rotation (× `dphi_max`).
- **Start**: the snake lies straight with its head at the letter's first point, body sticking out behind.
- **Lines**: the letter is two boundary lines `corridor_width` apart. Body points that have entered the letter
  should stay between them; points not yet on the letter are unconstrained. Leaving is penalised, not terminal.
- **Termination**: the head reaches the end of the letter (truncation at `max_steps`).
- **Reward**: progress along the letter − mean centerline deviation − amount outside the lines − small action cost.
- **Obs**: `[sin q, cos q, tail xy, sin/cos phi, head xy, next 8 centerline points relative to head, progress]`.

Sharp letter corners are rounded (`fillet_radius`) so a chain with the given joint limit can follow them;
the env raises a `ValueError` if a letter would need more bend per joint than `joint_limit` allows.
Glyphs are drawn without reversing on themselves (the snake can't turn back), so `4` (two strokes) is not
supported, `3` is flat-topped, `6`/`9` leave their loops slightly open, and `9` is drawn stem-first.
Self-collision is not modelled (e.g. the snake crosses itself in `8`).

## Environment
```
conda env create -f environment.yml   # python 3.11, numpy, matplotlib, imageio(+ffmpeg), pytest, torch cu128
conda activate snake
```
The env code itself only needs numpy, matplotlib and Pillow; mp4 output needs imageio + imageio-ffmpeg
(otherwise videos fall back to GIF).
