"""Shared core: numpy value net and exact chance-outcome expansion."""
import math

import numpy as np
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.map import number_probability

from feats import RES, DEVS

ROLL, BUY, ROBBER = ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD, ActionType.MOVE_ROBBER


class Value:
    def __init__(self, path):
        d = np.load(path)
        self.mu, self.sd, n = d["mu"], d["sd"], int(d["n"])
        self.W = [d[f"W{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]

    def __call__(self, X):
        h = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                np.maximum(h, 0, out=h)
        return 1.0 / (1.0 + np.exp(-h[:, 0]))


def outcomes(game, action, me):
    """[(game, prob)] for every random outcome of `action` on `game`, seen from `me`."""
    t = action.action_type
    st = game.state
    if t == ROLL:
        out = []
        for roll in range(2, 13):
            g = game.copy()
            g.execute(Action(action.color, t, (roll // 2, math.ceil(roll / 2))), validate_action=False)
            out.append((g, number_probability(roll)))
        return out
    if t == BUY:
        j = 1 - st.color_to_index[me]
        pile = list(st.development_listdeck)
        for d in DEVS:
            pile += [d] * st.player_state[f"P{j}_{d}_IN_HAND"]
        out = []
        for d in set(pile):
            g = game.copy()
            g.execute(Action(action.color, t, d), validate_action=False)
            out.append((g, pile.count(d) / len(pile)))
        return out
    if t == ROBBER and action.value[1] is not None:
        k = st.color_to_index[action.value[1]]
        hand = [st.player_state[f"P{k}_{r}_IN_HAND"] for r in RES]
        n = sum(hand)
        if n:
            out = []
            for r, c in zip(RES, hand):
                if c:
                    g = game.copy()
                    g.execute(action, validate_action=False, action_record=ActionRecord(action, r))
                    out.append((g, c / n))
            return out
    g = game.copy()
    g.execute(action, validate_action=False)
    return [(g, 1.0)]
