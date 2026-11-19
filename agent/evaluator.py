"""Position evaluation: features, the value nets, the handcrafted heuristic and the leaf value.

Everything here is a pure function of a catanatron Game plus the colour `me`, and reads only
what `me` can see (the opponent's unplayed development cards enter as a count). That is what
makes it safe to evaluate positions reached from a sampled world: the features are the same on
the sampled copy as on the true game.

The leaf value used by the search is

    leaf(game, me) = heuristic(game, me) + LEAF_K * logit(P_win)

where P_win is an ensemble of two small numpy MLPs over the 76 features below (value_v2r1.npz
and value_v5.npz, trained offline by self-play; this file only runs them), and `heuristic` is a
hand-written score whose weights (WEIGHTS) were selected by an offline tuning run and are frozen.
Two public-information overlays sit on top of the leaf value (see Evaluator): a denial term that
penalises leaving the rival ready to build, and a robber term used only when placing the robber.

Numeric constants are kept exactly as shipped; nothing here is a tunable.
"""
import math

import numpy as np
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import SETTLEMENT, CITY
from catanatron.models.map import number_probability

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
NEIGH = {n: list(STATIC_GRAPH.neighbors(n)) for n in STATIC_GRAPH.nodes}

#: Heuristic weights from the frozen tuning run (PARAMS in the shipped codex/evaluator_swap/baseline.py).
#: `econ` was 0.0 there, which disabled the economic-rate term entirely; it is therefore absent.
WEIGHTS = dict(prod=18.0, expand=0.71404, army=2.49792, hand=0.9)
#: Scale of the net logit added to the heuristic (K in the shipped codex/residual_sweep/runtime.py).
LEAF_K = 8.0
#: Overlay strengths from the same tuning run: denial multiplies public_threat, and the knight /
#: robber continuation uses 6 * denial per card-per-roll of newly blocked income.
DENIAL = 0.35
#: Expected-value bonus per extra card when a monopoly would take 3 or more (root scoring).
MONOPOLY_BONUS = 0.3037
#: Bonus for buying a development card that could complete Largest Army (root scoring).
TACTICAL_ARMY = 1.4


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
_map_cache = {}


def _node_prod(cmap):
    """node -> list of (tile_coordinate, resource_index, probability), cached per map object."""
    key = id(cmap)
    hit = _map_cache.get(key)
    if hit is not None and hit[0] is cmap:
        return hit[1]
    coord_of = {id(t): c for c, t in cmap.land_tiles.items()}
    np_ = {}
    for n, tiles in cmap.adjacent_tiles.items():
        np_[n] = [(coord_of[id(t)], RES.index(t.resource), number_probability(t.number))
                  for t in tiles if t.resource is not None]
    # Bounded: one entry per map seen, and a long run sees one map per game. Without the bound
    # the cache (and every map it pins) would grow for the whole process lifetime.
    if len(_map_cache) >= 8:
        _map_cache.clear()
    _map_cache[key] = (cmap, np_)
    return np_


def _player_block(st, board, nprod, color, idx, robber, own):
    """The per-player half of the feature vector (31 values, plus 6 private ones for `own`)."""
    ps = st.player_state
    k = f"P{idx}_"
    b = st.buildings_by_color[color]
    setts, cities = b[SETTLEMENT], b[CITY]
    prod = [0.0] * 5
    for n, mult in [(n, 1) for n in setts] + [(n, 2) for n in cities]:
        for coord, r, p in nprod[n]:
            if coord != robber:
                prod[r] += p * mult
    total = sum(prod)
    distinct = sum(1 for x in prod if x > 0)
    ports = board.get_player_port_resources(color)
    port = [1.0 if r in ports else 0.0 for r in RES] + [1.0 if None in ports else 0.0]
    hand = [ps[k + r + "_IN_HAND"] for r in RES]
    nhand = sum(hand)
    # Expansion: nodes buildable now, and nodes one road away, with the best production of each.
    comps = board.connected_components[color]
    mine = set().union(*comps) if comps else set()
    buildable = board.board_buildable_ids
    now = [n for n in mine if n in buildable]
    one = set()
    for n in mine:
        for m in NEIGH[n]:
            if m in buildable and m not in mine and board.get_edge_color((n, m)) is None:
                one.add(m)

    def npv(n):
        return sum(p for _, _, p in nprod[n])

    best_now = max((npv(n) for n in now), default=0.0)
    best_one = max((npv(n) for n in one), default=0.0)
    devs = [ps[k + d + "_IN_HAND"] for d in DEVS]
    out = [
        ps[k + "VICTORY_POINTS"],
        len(setts), len(cities), ps[k + "ROADS_AVAILABLE"],
        *prod, total, distinct,
        *port,
        *hand, nhand, 1.0 if nhand > 7 else 0.0,
        ps[k + "PLAYED_KNIGHT"], 1.0 if ps[k + "HAS_ARMY"] else 0.0,
        ps[k + "LONGEST_ROAD_LENGTH"], 1.0 if ps[k + "HAS_ROAD"] else 0.0,
        len(now), best_now, len(one), best_one,
        sum(devs),
    ]
    if own:
        out += devs + [ps[k + "ACTUAL_VICTORY_POINTS"]]
    return out


def feats(game, me):
    """The 76-feature vector from `me`'s perspective: me (37), opponent (31), global (4)."""
    st = game.state
    board = st.board
    nprod = _node_prod(board.map)
    robber = board.robber_coordinate
    i = st.color_to_index[me]
    j = 1 - i
    opp = st.colors[j]
    a = _player_block(st, board, nprod, me, i, robber, True)
    b = _player_block(st, board, nprod, opp, j, robber, False)
    g = [
        len(st.development_listdeck) + 0.0,
        st.num_turns / 100.0,
        1.0 if st.current_color() == me else 0.0,
        1.0 if st.is_initial_build_phase else 0.0,
    ]
    return a + b + g


def feat_names():
    base = ["vp", "n_sett", "n_city", "roads_left"] + [f"prod_{r}" for r in RES] + \
           ["prod_total", "prod_distinct"] + [f"port_{r}" for r in RES] + ["port_3to1"] + \
           [f"hand_{r}" for r in RES] + ["hand_n", "hand_over7", "knights", "has_army",
                                          "road_len", "has_road", "build_now_n", "build_now_best",
                                          "build_one_n", "build_one_best", "devs_n"]
    me = ["me_" + x for x in base] + [f"me_dev_{d}" for d in DEVS] + ["me_actual_vp"]
    op = ["opp_" + x for x in base]
    return me + op + ["deck_left", "turn", "my_turn", "initial_phase"]


NAMES = {name: i for i, name in enumerate(feat_names())}


# ----------------------------------------------------------------------------------------------
# Value nets
# ----------------------------------------------------------------------------------------------
class Value:
    """A numpy MLP (standardise, ReLU layers, sigmoid) loaded from an .npz of mu, sd, W_i, b_i."""

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


class EnsValue:
    """Mean of the nets' logits, returned as a probability."""

    def __init__(self, *paths):
        self.nets = [Value(p) for p in paths]

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = 0.0
        for n in self.nets:
            p = np.clip(n(X), 1e-6, 1 - 1e-6)
            z = z + np.log(p / (1 - p))
        z /= len(self.nets)
        return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------------------
# Handcrafted heuristic
# ----------------------------------------------------------------------------------------------
def heuristic(game, me, weights=WEIGHTS):
    """Hand-written score of a position for `me`, in roughly victory-point units."""
    x = feats(game, me)

    def f(name):
        return x[NAMES[name]]

    vp = f('me_actual_vp')
    if vp >= game.vps_to_win:
        return 1e9
    prod = [f('me_prod_' + r) for r in RES]
    hand = [f('me_hand_' + r) for r in RES]
    rates = [2 if f('me_port_'+r) else 3 if f('me_port_3to1') else 4 for r in RES]
    # Early income compounds; late in the game cashing in points takes priority.
    phase = max(.3, 1. - max(0., vp-4.) * .12)
    production = sum(prod) + .08 * sum(min(1., p/.08) for p in prod)
    enemy = f('opp_prod_total') + .08*f('opp_prod_distinct')
    score = 12*vp - 10*f('opp_vp') + phase*weights['prod']*(production - .8*enemy)
    # Immediate expansion is worth more than merely extending roads. The counts are capped so
    # that many empty low-yield branches cannot dominate a real purchase.
    expansion = (7*f('me_build_now_best') + 3*f('me_build_one_best')
                 + .25*min(3, f('me_build_now_n')) + .08*min(3, f('me_build_one_n')))
    score += phase * weights['expand'] * expansion
    # Goal-aware hand value. Ports enter as sell rates, never as free bonuses.
    goals = []
    if f('me_n_sett') > 0 and f('me_n_city') < 4:
        goals.append(((0,0,0,2,3), 5.5))
    if f('me_build_now_n') and f('me_n_sett') < 5:
        goals.append(((1,1,1,1,0), 5.5))
    elif f('me_build_one_n') and f('me_n_sett') < 5:
        goals.append(((2,2,1,1,0), 4.))
    if f('deck_left'):
        goals.append(((0,0,1,1,1), 1.5+weights['army']*.3))
    ready = 0.
    for cost, value in goals:
        deficit = sum(max(0, c-h) for c,h in zip(cost,hand))
        surplus = sum(max(0, h-c)//k for c,h,k in zip(cost,hand,rates))
        ready = max(ready, value*(.8*math.exp(-.65*deficit)
                                  + .2*math.exp(-.65*max(0, deficit-surplus))))
    score += weights['hand'] * (ready + .12*sum(hand))
    score -= .10*max(0, sum(hand)-7)
    knights = f('me_knights')
    opponent_knights = f('opp_knights')
    target = max(3., opponent_knights+1)
    # Played knights approach the army threshold. Stop valuing endless knights once the award is
    # safely held; the robber's own benefit already shows up in production.
    if not f('me_has_army'):
        score += weights['army'] * min(knights, target-1)
        knight_card = weights['army'] * (1.0 if knights < target else .3)
    else:
        knight_card = .4 + .35*(knights <= opponent_knights+1)
    score += knight_card*f('me_dev_KNIGHT')
    score += 1.3*f('me_dev_ROAD_BUILDING') + 1.5*f('me_dev_YEAR_OF_PLENTY')
    score += 1.8*f('me_dev_MONOPOLY')
    # Actual VP already includes VP cards: never count those twice.
    if not f('me_has_road'):
        score += .25*min(4., f('me_road_len'))
    return score


# ----------------------------------------------------------------------------------------------
# Public-information overlays
# ----------------------------------------------------------------------------------------------
def public_threat(game, me):
    """How ready the rival is to build a point right now, weighted by how close it is to 10."""
    st = game.state; i = 1-st.color_to_index[me]; other = st.colors[i]
    ps = st.player_state; key = 'P%d_' % i
    h = [ps[key+r+'_IN_HAND'] for r in RES]
    buildings = st.buildings_by_color[other]
    ports = st.board.get_player_port_resources(other)
    rates = [2 if r in ports else 3 if None in ports else 4 for r in RES]
    costs = []
    if buildings[SETTLEMENT] and ps[key+'CITIES_AVAILABLE']:
        costs.append((0,0,0,2,3))
    if ps[key+'SETTLEMENTS_AVAILABLE'] and st.board.buildable_node_ids(other):
        costs.append((1,1,1,1,0))
    if (ps[key+'LONGEST_ROAD_LENGTH'] >= 4 and not ps[key+'HAS_ROAD']
            and ps[key+'ROADS_AVAILABLE']):
        costs.append((1,1,0,0,0))
    if not costs:
        return 0.
    readiness = 0.
    for cost in costs:
        short = sum(max(0,c-v) for c,v in zip(cost,h))
        bank = sum(max(0,v-c)//k for c,v,k in zip(cost,h,rates))
        readiness = max(readiness, 1./(1.+max(0,short-bank)))
    # Public VP only: no claims about the identity of their held dev cards.
    urgency = 1.+max(0,ps[key+'VICTORY_POINTS']-5)*.75
    return urgency*readiness


def blocked_need(game, me):
    """Scarcity-weighted income currently under the robber, in cards per roll: positive when the
    robber hurts the opponent, negative when it hurts `me`."""
    st = game.state; board = st.board; np_ = _node_prod(board.map)
    result = 0.
    for color in st.colors:
        i=st.color_to_index[color]; key='P%d_' % i
        ps=st.player_state; b=st.buildings_by_color[color]
        production=[0.]*5
        blocked=[0.]*5
        for kind,mult in ((SETTLEMENT,1),(CITY,2)):
            for n in b[kind]:
                for coord,r,p in np_[n]:
                    production[r] += p*mult
                    if coord == board.robber_coordinate:
                        blocked[r] += p*mult
        score=0.
        for r,res in enumerate(RES):
            h=ps[key+res+'_IN_HAND']
            need = (3 if res=='ORE' else 2 if res=='WHEAT' else 1)
            marginal = .25 + .65*(h < need) + .35*(production[r] < .12)
            score += blocked[r]*marginal
        urgency=1.+max(0,ps[key+'VICTORY_POINTS']-6)*.2
        result += score*urgency*(-1. if color==me else 1.)
    return result


class Evaluator:
    """The three value functions the search uses, sharing one loaded net ensemble."""

    def __init__(self, *net_paths):
        self.net = EnsValue(*net_paths)

    def leaf(self, game, me):
        """heuristic + LEAF_K * logit(P_win); the base leaf without overlays."""
        p = float(self.net([feats(game, me)])[0]); p = min(max(p, 1e-6), 1-1e-6)
        return heuristic(game, me) + LEAF_K*np.log(p/(1-p))

    def value(self, game, me):
        """Leaf minus the denial overlay: a position is worse if the rival is about to score."""
        return self.leaf(game, me) - DENIAL*public_threat(game, me)

    def robber_value(self, game, me, original_block):
        """Value of a position reached by moving the robber, crediting newly blocked income."""
        return self.value(game, me) + DENIAL*6.*(blocked_need(game, me)-original_block)
