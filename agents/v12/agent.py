"""v12: v11 + reply-aware opening (greedy setup rollouts over the value net for both sides). v11: v5 search with the value net over efeats2 (76 + 47 features from the 639-float observation). v3: v2 + F-style robber. v2: same-turn beam search over a learned win-probability evaluator, budgeted by cpu_left.

At each decision every legal action is scored by the expected value of its outcomes (exact over
dice, dev draws and steals). Then, while time remains, the best few deterministic actions that keep
the turn ours are expanded recursively (beam), and their score becomes the best continuation found.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread: every thread counts against the clock
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
SETTLE = ActionType.BUILD_SETTLEMENT
OPEN_K = int(os.environ.get("V12_OPEN_K", "5"))
OPEN_M = int(os.environ.get("V12_OPEN_M", "10"))   # settlement candidates per rollout step, by pips
from catanatron.models.map import number_probability  # noqa: E402


def _npv(game):
    """node -> pip value, cached per map."""
    m = game.state.board.map
    key = id(m)
    hit = _npv.cache.get(key)
    if hit is not None and hit[0] is m:
        return hit[1]
    d = {n: sum(number_probability(t.number) for t in ts if t.resource is not None)
         for n, ts in m.adjacent_tiles.items()}
    _npv.cache[key] = (m, d)
    return d


_npv.cache = {}


END = ActionType.END_TURN
BEAM = int(os.environ.get("V2_BEAM", "3"))
DEPTH = int(os.environ.get("V2_DEPTH", "3"))
RESERVE = float(os.environ.get("V2_RESERVE", "0.15"))  # seconds of allowance never spent
SPEND = float(os.environ.get("V2_SPEND", "0.15"))      # fraction of spare allowance per decision
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

    def greedy_setup(self, game, me):
        """Play the rest of the initial build phase, each side choosing 1-ply by the value net."""
        g = game
        while g.state.is_initial_build_phase and g.winning_color() is None:
            acts = g.playable_actions
            mover = g.state.current_color()
            if len(acts) > OPEN_M and acts[0].action_type == SETTLE:
                npv = _npv(g)
                acts = sorted(acts, key=lambda a: -npv[a.value])[:OPEN_M]
            succ = []
            for a in acts:
                s = g.copy()
                try:
                    s.execute(a, validate_action=False)
                except Exception:
                    continue
                succ.append(s)
            if not succ:
                break
            if len(succ) == 1:
                g = succ[0]
                continue
            v = self.V(efeats2(np.stack([encode_game(s, mover) for s in succ])))
            g = succ[int(np.argmax(v))]
        return g

    def opening(self, world, actions, me):
        """Settlement placement: top-K by 1-ply value, re-ranked by the value after a greedy
        completion of the setup phase by both players (the opponent's reply included)."""
        q, succ = self.score(world, actions, me)
        top = [int(i) for i in np.argsort(-q)[:OPEN_K] if succ[i] is not None]
        if len(top) < 2:
            return int(np.argmax(q))
        ends = [self.greedy_setup(succ[i], me) for i in top]
        v = self.V(efeats2(np.stack([encode_game(g, me) for g in ends])))
        return top[int(np.argmax(v))]

    def decide(self, obs, playable_actions):
        t0 = time.process_time()
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        deadline = t0 + min(CAP, max(0.0, (left - RESERVE) * SPEND))
        world = obs.sample_world(self.rng)
        if F_ROBBER and playable_actions[0].action_type == ROBBER:
            return self.robber(world, playable_actions, obs.me)
        if world.state.is_initial_build_phase and playable_actions[0].action_type == SETTLE:
            return self.opening(world, playable_actions, obs.me)
        q = self.deepen(world, playable_actions, obs.me, DEPTH, deadline)
        return int(np.argmax(q))
