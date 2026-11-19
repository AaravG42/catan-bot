"""v1: one-ply search over a learned win-probability evaluator, exact over chance outcomes."""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "common")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from core import Value, outcomes  # noqa: E402
from feats import feats  # noqa: E402


def warmup(V):
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer
    state = random.getstate()
    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=7)
    for _ in range(60):
        if g.winning_color() is not None:
            break
        for a in g.playable_actions[:3]:
            for o, _ in outcomes(g, a, Color.RED):
                V([feats(o, Color.RED)])
        g.play_tick()
    random.setstate(state)


class Agent:
    def __init__(self):
        self.V = Value(os.path.join(HERE, "value.npz"))
        self.rng = random.Random(1)
        warmup(self.V)

    def decide(self, obs, playable_actions):
        me = obs.me
        world = obs.sample_world(self.rng)
        rows, owner, probs, fixed = [], [], [], {}
        for i, a in enumerate(playable_actions):
            try:
                outs = outcomes(world, a, me)
            except Exception:
                fixed[i] = -1.0
                continue
            for g, p in outs:
                w = g.winning_color()
                if w is not None:
                    fixed[i] = fixed.get(i, 0.0) + p * (1.0 if w == me else 0.0)
                    continue
                rows.append(feats(g, me)); owner.append(i); probs.append(p)
        score = np.zeros(len(playable_actions))
        for i, v in fixed.items():
            score[i] += v
        if rows:
            v = self.V(rows)
            np.add.at(score, owner, np.asarray(probs) * v)
        return int(np.argmax(score))
