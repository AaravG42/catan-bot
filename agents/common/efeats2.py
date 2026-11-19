"""Extended position features: the 76 of efeats.py plus 47 that the 64-32 MLP could not derive.

    F = efeats2(X)      X (n, 639) engine observation from the viewing seat -> (n, 123) float32

my_turn is read from the observation itself (field 638). What is new, and why:

  affordability  deficit (cards still missing after the best bank/port trades) and can-build-now for
                 settlement / city / road / dev card, for BOTH players. The opponent's hand is public,
                 so "what can the opponent do next turn" is a feature, not a search.
  race           opponent's expected hidden VP (dev count x unseen VP cards / unseen pile), its
                 estimated total VP, the VP gap, roads still needed for Longest Road and knights for
                 Largest Army on both sides.
  robber         production each side loses to the robber right now, and how concentrated each
                 side's production is on one tile (robber exposure).
  contested      spots one road away for both players: count and best pips.
  pieces, phase  settlements/cities left per side, has_rolled, played a dev this turn, discards
                 pending, prompt one-hot, free roads, bank per resource.
"""
import numpy as np

from efeats import efeats, A, INC, ADJ, EU, EV, OHU, OHV, NT, NN

# cost rows in RES order WOOD BRICK SHEEP WHEAT ORE: settlement, city, road, dev card
COST = np.array([[1, 1, 1, 1, 0], [0, 0, 0, 2, 3], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], np.float32)
N_EXTRA = 47
N_FEATS = 76 + N_EXTRA


def _deficit(res, ports):
    """Cards still missing for each build after greedily trading surplus at the best rate."""
    rate = np.where(ports[:, :5] > 0, 2.0, np.where(ports[:, 5:6] > 0, 3.0, 4.0)).astype(np.float32)
    out = []
    for c in COST:
        missing = np.maximum(c - res, 0).sum(1)
        trades = np.floor(np.maximum(res - c, 0) / rate).sum(1)
        out.append(np.maximum(missing - trades, 0))
    return np.stack(out, 1)


def _spots(occ_any, settle, city, road, road_other):
    mine = ((settle + city) > 0) | ((road @ INC) > 0)
    buildable = (occ_any == 0) & ((occ_any @ ADJ) == 0)
    now = mine & buildable
    free = (road + road_other) == 0
    cand = buildable & ~mine
    hit = free & ((mine[:, EU] & cand[:, EV]) | (mine[:, EV] & cand[:, EU]))
    one = ((hit & cand[:, EV]) @ OHV + (hit & cand[:, EU]) @ OHU) > 0
    return now, one


def extra(X):
    X = np.asarray(X, np.float32)
    n = len(X)
    tiles = X[:, :152].reshape(n, NT, 8)
    tres = tiles[:, :, :5]
    pips = tiles[:, :, 6] / 36.0
    robber = tiles[:, :, 7]
    land = tres.sum(2)                                   # 1 for resource tiles
    nodes = X[:, 206:422].reshape(n, NN, 4)
    ms, mc, os_, oc = nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2], nodes[:, :, 3]
    occ = ms + mc + os_ + oc
    roads = X[:, 422:566].reshape(n, -1, 2)
    mr, orr = roads[:, :, 0], roads[:, :, 1]
    me, op, g = X[:, 566:601], X[:, 601:624], X[:, 624:639]
    c = lambda a, i: a[:, i:i + 1]

    # --- affordability (both players) ---
    d_me = _deficit(me[:, 0:5], me[:, 28:34])
    d_op = _deficit(op[:, 0:5], op[:, 17:23])
    now_me, one_me = _spots(occ, ms, mc, mr, orr)
    now_op, one_op = _spots(occ, os_, oc, orr, mr)
    npv = (land * pips) @ A.T                             # node pip value, robber ignored
    deck = c(g, 5)
    can_me = np.stack([
        (d_me[:, 0] == 0) & (now_me.sum(1) > 0) & (me[:, 17] > 0),
        (d_me[:, 1] == 0) & (ms.sum(1) > 0) & (me[:, 18] > 0),
        (d_me[:, 2] == 0) & (me[:, 16] > 0),
        (d_me[:, 3] == 0) & (deck[:, 0] > 0)], 1).astype(np.float32)
    can_op = np.stack([
        (d_op[:, 0] == 0) & (now_op.sum(1) > 0) & (op[:, 12] > 0),
        (d_op[:, 1] == 0) & (os_.sum(1) > 0) & (op[:, 13] > 0),
        (d_op[:, 2] == 0) & (op[:, 11] > 0),
        (d_op[:, 3] == 0) & (deck[:, 0] > 0)], 1).astype(np.float32)

    # --- race ---
    unseen_vp = np.maximum(5.0 - me[:, 9], 0.0)
    opp_devs = op[:, 5]
    exp_hidden = opp_devs * unseen_vp / np.maximum(deck[:, 0] + opp_devs, 1.0)
    opp_vp_est = op[:, 10] + exp_hidden
    vp_gap = me[:, 15] - opp_vp_est
    my_len, op_len = me[:, 21], op[:, 16]
    me_has_lr, op_has_lr = me[:, 19] > 0, op[:, 14] > 0
    me_lr_need = np.where(me_has_lr, 0.0, np.maximum(np.where(op_has_lr, op_len + 1, 5) - my_len, 0))
    op_lr_need = np.where(op_has_lr, 0.0, np.maximum(np.where(me_has_lr, my_len + 1, 5) - op_len, 0))
    my_k, op_k = me[:, 10], op[:, 6]
    me_has_la, op_has_la = me[:, 20] > 0, op[:, 15] > 0
    me_la_need = np.where(me_has_la, 0.0, np.maximum(np.where(op_has_la, op_k + 1, 3) - my_k, 0))
    op_la_need = np.where(op_has_la, 0.0, np.maximum(np.where(me_has_la, my_k + 1, 3) - op_k, 0))

    # --- robber: production lost now, and concentration on one tile ---
    w_me = np.matmul((ms + 2 * mc)[:, None, :], A)[:, 0, :]        # (n, 19) building weight per tile
    w_op = np.matmul((os_ + 2 * oc)[:, None, :], A)[:, 0, :]
    tp_me = w_me * land * pips
    tp_op = w_op * land * pips
    lost_me = (tp_me * robber).sum(1)
    lost_op = (tp_op * robber).sum(1)
    conc_me = tp_me.max(1) / np.maximum(tp_me.sum(1), 1e-6)
    conc_op = tp_op.max(1) / np.maximum(tp_op.sum(1), 1e-6)

    # --- contested expansion spots ---
    both = one_me & one_op
    cont_n = both.sum(1).astype(np.float32)
    cont_best = np.where(both, npv, 0).max(1)

    cols = [d_me, can_me, d_op, can_op,
            np.stack([exp_hidden, opp_vp_est, vp_gap, me_lr_need, op_lr_need, me_la_need, op_la_need,
                      lost_me, lost_op, conc_me, conc_op, cont_n, cont_best], 1),
            me[:, 17:19], op[:, 12:14],                       # pieces left: settlements, cities
            me[:, 22:24], c(me, 34),                          # has_rolled, played_dev, discards left
            g[:, 7:12], c(g, 13),                             # prompt one-hot, free roads
            g[:, 0:5]]                                        # bank
    out = np.concatenate([np.asarray(x, np.float32) for x in cols], 1)
    assert out.shape[1] == N_EXTRA, out.shape
    return out


def efeats2(X):
    X = np.asarray(X, np.float32)
    return np.concatenate([efeats(X, X[:, 638]), extra(X)], 1)


def extra_names():
    b = ["settle", "city", "road", "dev"]
    return ([f"me_deficit_{x}" for x in b] + [f"me_can_{x}" for x in b] +
            [f"opp_deficit_{x}" for x in b] + [f"opp_can_{x}" for x in b] +
            ["opp_exp_hidden_vp", "opp_vp_est", "vp_gap", "me_lr_need", "opp_lr_need", "me_la_need",
             "opp_la_need", "me_prod_lost_robber", "opp_prod_lost_robber", "me_prod_conc",
             "opp_prod_conc", "contested_one_n", "contested_one_best",
             "me_setts_left", "me_cities_left", "opp_setts_left", "opp_cities_left",
             "me_has_rolled", "me_played_dev", "me_discards_left",
             "prompt_init_settle", "prompt_init_road", "prompt_play", "prompt_discard", "prompt_robber",
             "free_roads"] + [f"bank_{r}" for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]])
