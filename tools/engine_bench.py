"""Throughput of the C engine vs catanatron, one core.

  PYTHONHASHSEED=0 python -m tools.engine_bench
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

import catan_kit  # noqa: E402,F401
from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402

from engine import Engine, VecEnv, STATE_SIZE  # noqa: E402
from engine import cengine, obs_np  # noqa: E402


def bench_vec(n, obs, seconds=3.0):
    env = VecEnv(n, seed=1, autoplay=True, obs=obs)
    rng = np.random.default_rng(0)
    decisions = steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        # random legal action per env: argmax of noise over the mask
        a = np.argmax(rng.random(env.mask.shape, np.float32) * env.mask, axis=1)
        steps += env.step(a)
        decisions += n
    dt = time.perf_counter() - t0
    return decisions / dt, steps / dt


def bench_vec_c_only(n, obs, seconds=3.0):
    """Same, but time only the C call (excludes numpy action sampling)."""
    env = VecEnv(n, seed=1, autoplay=True, obs=obs)
    rng = np.random.default_rng(0)
    tc = 0.0
    decisions = steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = np.argmax(rng.random(env.mask.shape, np.float32) * env.mask, axis=1)
        t1 = time.perf_counter()
        steps += env.step(a)
        tc += time.perf_counter() - t1
        decisions += n
    return decisions / tc, steps / tc


def main():
    print(f"State: {STATE_SIZE} bytes")
    # full random games inside C
    st = np.zeros((64, STATE_SIZE), np.uint8)
    t0 = time.perf_counter()
    steps = cengine.lib.vec_random_games(st.ctypes.data, 64, 20000, 3, 0)
    dt = time.perf_counter() - t0
    print(f"C random self-play: {20000 / dt:,.0f} games/s, {steps / dt / 1e6:.2f} M steps/s "
          f"({steps / 20000:.0f} steps/game)")
    for obs in (False, True):
        d, s = bench_vec(1024, obs)
        dc, sc = bench_vec_c_only(1024, obs)
        print(f"VecEnv(1024) {'with' if obs else 'no  '} obs: end-to-end {d / 1e6:.2f} M decisions/s "
              f"({s / 1e6:.2f} M steps/s); C call alone {dc / 1e6:.2f} M decisions/s ({sc / 1e6:.2f} M steps/s)")
    # copy cost
    e = Engine(seed=5)
    t0 = time.perf_counter()
    for _ in range(100000):
        st[0] = e.buf
    print(f"state copy (numpy row assign, incl. Python overhead): {(time.perf_counter() - t0) * 10:.2f} us")

    # catanatron baseline: copy+execute of a random legal action, and full random games
    random.seed(1)
    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
    for _ in range(60):
        g.execute(random.choice(g.playable_actions))
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        c = g.copy()
        c.execute(random.choice(c.playable_actions))
        n += 1
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"catanatron copy+execute: {us:.1f} us/step ({1e6 / us:,.0f} steps/s)")
    t0 = time.perf_counter()
    games = steps = 0
    while time.perf_counter() - t0 < 5.0:
        g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=games)
        while g.winning_color() is None and g.state.num_turns < 1000:
            g.execute(random.choice(g.playable_actions))
            steps += 1
        games += 1
    dt = time.perf_counter() - t0
    print(f"catanatron random games: {games / dt:.1f} games/s, {steps / dt:,.0f} steps/s")
    # numpy encoder cost (submission-side)
    t0 = time.perf_counter()
    for _ in range(2000):
        obs_np.encode_game(g, g.state.current_color())
    print(f"obs_np.encode_game: {(time.perf_counter() - t0) / 2000 * 1e6:.0f} us")


if __name__ == "__main__":
    main()
