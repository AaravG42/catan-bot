"""v3's 76 position features (agents/common/feats.py), computed in batch from the engine's 639-float
observation (engine/obs_np.py layout), so the proven evaluator can score engine states at speed.

    F = efeats(X, my_turn)     X (n, 639) from the seat's view, my_turn (n,) 0/1 -> (n, 76)

Same feature order as feats.feat_names(). Known approximations versus feats.py: "my network" nodes
are my buildings plus nodes touched by my roads (catanatron's component bookkeeping differs in rare
road-cut cases), and road length is the board value rather than the lagging player_state copy.
"""
import numpy as np

from catanatron.models.board import get_edges
from catanatron.models.map import build_map

_m = build_map("BASE")
_coords = list(_m.land_tiles.keys())
_tidx = {id(t): i for i, t in enumerate(_m.land_tiles.values())}
NT, NN = 19, 54
A = np.zeros((NN, NT), np.float32)          # node -> adjacent land tiles
for n, tiles in _m.adjacent_tiles.items():
    for t in tiles:
        if id(t) in _tidx:
            A[n, _tidx[id(t)]] = 1
_edges = list(get_edges(None))
EU = np.array([e[0] for e in _edges])
EV = np.array([e[1] for e in _edges])
NE = len(_edges)
INC = np.zeros((NE, NN), np.float32)        # edge -> its two nodes
INC[np.arange(NE), EU] = 1
INC[np.arange(NE), EV] = 1
ADJ = (INC.T @ INC > 0).astype(np.float32)
np.fill_diagonal(ADJ, 0)
OHU = np.eye(NN, dtype=np.float32)[EU]
OHV = np.eye(NN, dtype=np.float32)[EV]


def _block(prod_node, npv, occ_any, b_me, settle, city, road, road_other, sc, ports, res, devs_n):
    """One player's 31 base features. sc: dict of scalar columns."""
    prod = np.matmul((settle + 2 * city)[:, None, :], prod_node)[:, 0, :]
    total = prod.sum(1, keepdims=True)
    distinct = (prod > 0).sum(1, keepdims=True).astype(np.float32)
    hand_n = res.sum(1, keepdims=True)
    mine = ((settle + city) > 0) | ((road @ INC) > 0)
    buildable = (occ_any == 0) & ((occ_any @ ADJ) == 0)
    now = mine & buildable
    free = (road + road_other) == 0
    cand = buildable & ~mine
    hit = free & ((mine[:, EU] & cand[:, EV]) | (mine[:, EV] & cand[:, EU]))
    # node m is "one road away" if some free edge joins it to my network
    one = ((hit & cand[:, EV]) @ OHV + (hit & cand[:, EU]) @ OHU) > 0
    best_now = np.where(now, npv, 0).max(1, keepdims=True)
    best_one = np.where(one, npv, 0).max(1, keepdims=True)
    cols = [sc["vp"], settle.sum(1, keepdims=True), city.sum(1, keepdims=True), sc["roads_left"],
            prod, total, distinct, ports, res, hand_n, (hand_n > 7).astype(np.float32),
            sc["knights"], sc["army"], sc["road_len"], sc["has_road"],
            now.sum(1, keepdims=True).astype(np.float32), best_now,
            one.sum(1, keepdims=True).astype(np.float32), best_one, devs_n]
    return np.concatenate([c.astype(np.float32) for c in cols], 1)


def efeats(X, my_turn):
    X = np.asarray(X, np.float32)
    n = len(X)
    tiles = X[:, :152].reshape(n, NT, 8)
    tres = tiles[:, :, :5]
    p = tiles[:, :, 6] / 36.0 * (1 - tiles[:, :, 7])         # robber blocks its tile
    prod_node = np.matmul(A, tres * p[:, :, None])                          # (b, 54, 5)
    npv = (tres.sum(2) * tiles[:, :, 6] / 36.0) @ A.T                       # feats' npv ignores robber
    nodes = X[:, 206:422].reshape(n, NN, 4)
    ms, mc, os_, oc = nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2], nodes[:, :, 3]
    occ = ms + mc + os_ + oc
    roads = X[:, 422:566].reshape(n, NE, 2)
    mr, orr = roads[:, :, 0], roads[:, :, 1]
    me = X[:, 566:601]
    op = X[:, 601:624]
    g = X[:, 624:639]
    c = lambda a, i: a[:, i:i + 1]
    # me: res 0-4, dev 5-9 (K YOP MONO RB VP), played 10-13 (K YOP MONO RB), vp 14, avp 15,
    # roads 16, setts 17, cities 18, LR 19, LA 20, road_len 21, rolled 22, played 23, owned 24-27,
    # ports 28-33, discards 34
    sm = dict(vp=c(me, 14), roads_left=c(me, 16), knights=c(me, 10), army=c(me, 20),
              road_len=c(me, 21), has_road=c(me, 19))
    a = _block(prod_node, npv, occ, None, ms, mc, mr, orr, sm, me[:, 28:34], me[:, 0:5],
               me[:, 5:10].sum(1, keepdims=True))
    devs = me[:, [5, 9, 8, 6, 7]]          # feats order: KNIGHT VP ROAD_BUILDING YOP MONOPOLY
    a = np.concatenate([a, devs, c(me, 15)], 1)
    # opp: res 0-4, dev count 5, played 6-9, vp 10, roads 11, setts 12, cities 13, LR 14, LA 15,
    # road_len 16, ports 17-22
    so = dict(vp=c(op, 10), roads_left=c(op, 11), knights=c(op, 6), army=c(op, 15),
              road_len=c(op, 16), has_road=c(op, 14))
    b = _block(prod_node, npv, occ, None, os_, oc, orr, mr, so, op[:, 17:23], op[:, 0:5], c(op, 5))
    # global: bank 0-4, deck 5, num_turns 6, prompt 7-11, rb 12, free 13, my_turn 14
    gl = np.concatenate([c(g, 5), c(g, 6) / 100.0, np.asarray(my_turn, np.float32)[:, None],
                         (g[:, 7:9].sum(1, keepdims=True) > 0).astype(np.float32)], 1)
    return np.concatenate([a, b, gl], 1)
