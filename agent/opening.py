"""Setup placement (both settlements and both roads): a reply-aware rollout over a value net.

choose(game, actions, me), for every initial-phase decision with more than one legal action:

  1. One ply. Play each legal action on a copy and score the result with the setup value net
     (value_123.npz: P(win) for `me` from the 123 features of features123).
  2. Keep the TOP_K best (5).
  3. Rollout. From each of those, play the rest of the setup phase greedily for BOTH players:
     whoever is to move picks the action whose result the same net rates best from that mover's
     seat. Settlement choices are pre-filtered to the ROLLOUT_WIDTH (10) highest-pip nodes.
  4. Play the candidate whose finished setup the net rates best for `me`.

Step 3 is what makes it reply-aware: a spot is judged by the position after the opponent's
answer (and our own later placements), not by its pips alone. Nothing here reads the clock:
the setup phase costs a fixed, small amount of CPU per decision.
"""
import numpy as np
from catanatron.models.enums import ActionType
from catanatron.models.map import number_probability

from features123 import encode_game, features123

#: The initial-phase decisions this module answers (the only two kinds the setup phase has).
SETUP_ACTIONS = (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD)
#: Candidates kept after the one-ply scoring and rolled out (OPEN_K of the shipped v12).
TOP_K = 5
#: Settlement actions considered at each rollout step, by node pips (OPEN_M of the shipped v12).
ROLLOUT_WIDTH = 10


def node_pips(cmap):
    """node -> sum of number_probability over its resource tiles (robber ignored)."""
    return {n: sum(number_probability(t.number) for t in ts if t.resource is not None)
            for n, ts in cmap.adjacent_tiles.items()}


class SetupPlanner:
    def __init__(self, value_net):
        self.net = value_net

    def _rate(self, games, color):
        """The net's P(win) for `color` in each of `games`, as one batch."""
        return self.net(features123(np.stack([encode_game(g, color) for g in games])))

    def _finish_setup(self, game, pips):
        """Play out the rest of the initial phase, each side choosing 1-ply by the value net."""
        g = game
        while g.state.is_initial_build_phase and g.winning_color() is None:
            acts = g.playable_actions
            mover = g.state.current_color()
            if len(acts) > ROLLOUT_WIDTH and acts[0].action_type == ActionType.BUILD_SETTLEMENT:
                # Stable sort: equal-pip nodes keep the engine's action order.
                acts = sorted(acts, key=lambda a: -pips[a.value])[:ROLLOUT_WIDTH]
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
            g = succ[int(np.argmax(self._rate(succ, mover)))]
        return g

    def choose(self, game, actions, me):
        """Index into `actions` of the setup settlement or road to play."""
        # Step 1: one ply. An action the engine rejects scores -1 (below any probability) and is
        # never rolled out; a placement that already wins (possible only with a tiny vps_to_win)
        # scores 1, or 0 if it hands the opponent the win.
        q = np.zeros(len(actions))
        succ = [None] * len(actions)
        rated = []
        for i, a in enumerate(actions):
            s = game.copy()
            try:
                s.execute(a, validate_action=False)
            except Exception:
                q[i] = -1.0
                continue
            succ[i] = s
            winner = s.winning_color()
            if winner is not None:
                q[i] = 1.0 if winner == me else 0.0
            else:
                rated.append(i)
        if rated:
            q[rated] = self._rate([succ[i] for i in rated], me)
        # Step 2. np.argsort's default (non-stable) order is kept on purpose: it decides ties.
        top = [int(i) for i in np.argsort(-q)[:TOP_K] if succ[i] is not None]
        if len(top) < 2:
            return int(np.argmax(q))
        # Steps 3 and 4. Node pips depend only on the map, which the rollout never changes.
        pips = node_pips(game.state.board.map)
        ends = [self._finish_setup(succ[i], pips) for i in top]
        return top[int(np.argmax(self._rate(ends, me)))]
