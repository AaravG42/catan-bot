"""v11: v5 search with the value net over efeats2 (76 + 47 features from the 639-float observation). v3: v2 + F-style robber. v2: same-turn beam search over a learned win-probability evaluator, budgeted by cpu_left.

At each decision every legal action is scored by the expected value of its outcomes (exact over
dice, dev draws and steals). Then, while time remains, the best few deterministic actions that keep
the turn ours are expanded recursively (beam), and their score becomes the best continuation found.
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "common"), os.path.join(HERE, "..", "..", "engine")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402
from catanatron.models.enums import ActionType  # noqa: E402

from core import outcomes  # noqa: E402
from core import Value  # noqa: E402
from efeats2 import efeats2  # noqa: E402
from obs_np import encode_game  # noqa: E402
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn  # noqa: E402

F_VALUE = get_value_fn("base_fn", DEFAULT_WEIGHTS)
F_ROBBER = os.environ.get("V4_F_ROBBER", "1") == "1"
ROBBER = ActionType.MOVE_ROBBER
DISCARD = ActionType.DISCARD_RESOURCE

END = ActionType.END_TURN
BEAM = int(os.environ.get("V2_BEAM", "3"))
DEPTH = int(os.environ.get("V2_DEPTH", "3"))
RESERVE = float(os.environ.get("V2_RESERVE", "0.20"))  # seconds of allowance never spent
SPEND = float(os.environ.get("V2_SPEND", "0.12"))      # fraction of spare allowance per decision
CAP = float(os.environ.get("V2_CAP", "0.06"))          # hard cap on one decision's search, seconds


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
                V(efeats2(encode_game(o, Color.RED)[None]))
        g.play_tick()
    random.setstate(state)


class Agent:
    def __init__(self):
        self.V = Value(os.path.join(HERE, "value.npz"))
        self.rng = random.Random(1)
        warmup(self.V)

    def score(self, game, actions, me, deadline=None):
        """Expected value of each action; also returns the single successor of deterministic ones.
        Past the deadline, remaining actions get -inf (only used below the root)."""
        rows, owner, probs = [], [], []
        q = np.zeros(len(actions))
        succ = [None] * len(actions)
        for i, a in enumerate(actions):
            if deadline is not None and time.process_time() > deadline:
                q[i:] = -np.inf
                break
            try:
                outs = outcomes(game, a, me)
            except Exception:
                q[i] = -1.0
                continue
            if len(outs) == 1:
                succ[i] = outs[0][0]
            for g, p in outs:
                w = g.winning_color()
                if w is not None:
                    q[i] += p * (1.0 if w == me else 0.0)
                    continue
                rows.append(encode_game(g, me)); owner.append(i); probs.append(p)
        if rows:
            np.add.at(q, owner, np.asarray(probs) * self.V(efeats2(np.stack(rows))))
        return q, succ

    def deepen(self, game, actions, me, depth, deadline, root=True):
        q, succ = self.score(game, actions, me, None if root else deadline)
        if depth > 0:
            for i in np.argsort(-q)[:BEAM]:
                if time.process_time() > deadline:
                    break
                g = succ[i]
                if (g is None or actions[i].action_type == END or g.winning_color() is not None
                        or g.state.current_color() != me):
                    continue
                acts = g.playable_actions
                if len(acts) == 1:
                    continue
                sub = self.deepen(g, acts, me, depth - 1, deadline, root=False)
                q[i] = max(q[i], sub.max())
        return q

    def robber(self, world, actions, me):
        """F's robber: the placement that leaves the best F value (dominated by enemy production)."""
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a)
            except Exception:
                continue
            v = F_VALUE(g, me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    def decide(self, obs, playable_actions):
        t0 = time.process_time()
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        deadline = t0 + min(CAP, max(0.0, (left - RESERVE) * SPEND))
        world = obs.sample_world(self.rng)
        if F_ROBBER and playable_actions[0].action_type == ROBBER:
            return self.robber(world, playable_actions, obs.me)
        depth = 0 if playable_actions[0].action_type == DISCARD else DEPTH
        q = self.deepen(world, playable_actions, obs.me, depth, deadline)
        return int(np.argmax(q))
