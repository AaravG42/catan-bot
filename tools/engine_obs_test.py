"""C vs numpy observation encoders: bit-identical on the same positions.

Plays random catanatron games mirrored into the engine and, at every step, compares
  * engine.encode(seat=s) with obs_np.encode_game(game, color_s) for BOTH seats
  * obs_np on the harness's masked view (catan_kit.obs.masked_copy, hidden cards redrawn) for the
    seat to move, with the C encoding of the true state (hidden info must not leak in)

  PYTHONHASHSEED=0 python -m tools.engine_obs_test --games 200
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import catan_kit  # noqa: E402,F401
from catan_kit.obs import masked_copy  # noqa: E402
from catanatron.game import Game, TURNS_LIMIT  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402

from engine import Engine  # noqa: E402
from engine import bridge, obs_np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=7)
    a = ap.parse_args()
    positions = 0
    for seed in range(a.seed0, a.seed0 + a.games):
        random.seed(seed)
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
        eng = bridge.load_into(Engine(), game)
        rng = random.Random(seed)
        while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
            st = game.state
            for seat, color in enumerate(st.colors):
                c = eng.encode(seat=seat)
                p = obs_np.encode_game(game, color)
                assert c.tobytes() == p.tobytes(), (seed, seat, (c != p).nonzero())
            me = st.current_color()
            view = masked_copy(game, me, rng)
            p = obs_np.encode_game(view, me)
            assert eng.encode().tobytes() == p.tobytes(), (seed, "masked view", (eng.encode() != p).nonzero())
            positions += 1
            action = rng.choice(game.playable_actions)
            game.execute(action)
            eng.step(bridge.action_index(action), bridge.outcome_of(st.action_records[-1]))
    print(f"OK: {positions} positions x (2 seats + masked view), C and numpy encoders bit-identical")


if __name__ == "__main__":
    main()
