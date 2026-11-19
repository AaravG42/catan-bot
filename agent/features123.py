"""The 123 features read by the setup value net (value_123.npz), in two steps.

    x = encode_game(game, color)        float32[639]  flat observation from `color`'s seat
    F = features123(X)                   X (n, 639)  -> (n, 123) float32

Step 1, encode_game, writes the raw board and both players' public state into a 639-float
vector (layout below). Step 2 derives features from a batch of those vectors with numpy: first
the same 76 position features that evaluator.feats computes from a Game (base76), then 47 more
that a 64-32 MLP cannot easily derive itself (extra47). Going through the flat vector lets a
whole batch of candidate positions be featurised in one vectorised call.

Only what `color` may see is read: the opponent's development cards enter as a count, its hidden
victory points not at all.

Observation layout (raw small integers as float32; the net standardises them):

  [0,152)   19 tiles x 8   resource one-hot (WOOD BRICK SHEEP WHEAT ORE), desert, pips, robber
  [152,206) 9 ports x 6    resource one-hot, 3:1
  [206,422) 54 nodes x 4   my settlement, my city, opp settlement, opp city
  [422,566) 72 edges x 2   my road, opp road
  [566,601) me    res(5) dev(5: KNIGHT YOP MONO RB VP) played(4) vp actual_vp roads_left
                  settlements_left cities_left has_longest_road has_largest_army road_length
                  has_rolled played_dev_this_turn dev_owned_at_turn_start(4) ports(6: 2:1 x5, 3:1)
                  discards_left
  [601,624) opp   res(5) dev_count played(4) vp roads_left settlements_left cities_left
                  has_longest_road has_largest_army road_length ports(6)
  [624,639) bank(5) deck_left num_turns prompt one-hot(5: init settle, init road, play, discard,
                  robber) road_building free_roads to_move (1 if the viewing seat is to move)

Tile, port, node and edge ids are catanatron's (land_tiles order, port ids, node ids, get_edges
order). The base76 step differs from evaluator.feats in two known, harmless ways: "my network" is
my buildings plus the nodes my roads touch, and road length is the board's value rather than the
player_state copy. The net was trained on exactly these features, so they are kept as they are.
"""
import numpy as np

from catanatron.models.board import get_edges
from catanatron.models.enums import ActionPrompt, SETTLEMENT
from catanatron.models.map import build_map

OBS_DIM = 639
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
#: Development-card order of the observation vector (not the order of evaluator.DEVS).
OBS_DEVS = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]
_RES_INDEX = {r: i for i, r in enumerate(RES)}
_PROMPT = {ActionPrompt.BUILD_INITIAL_SETTLEMENT: 0, ActionPrompt.BUILD_INITIAL_ROAD: 1,
           ActionPrompt.PLAY_TURN: 2, ActionPrompt.DISCARD: 3, ActionPrompt.MOVE_ROBBER: 4}
_PIPS = {n: 6 - abs(7 - n) for n in range(2, 13)}

# ----------------------------------------------------------------------------------------------
# Board topology (identical for every BASE map: only the tile contents are shuffled)
# ----------------------------------------------------------------------------------------------
_EDGES = list(get_edges(None))
_EDGE_ID = {}
for _i, _e in enumerate(_EDGES):
    _EDGE_ID[_e] = _i
    _EDGE_ID[(_e[1], _e[0])] = _i

# Any BASE map gives the node -> tile adjacency; tile contents are not used. build_map draws from
# the global `random`, exactly once, at import, as the shipped module did.
_m = build_map("BASE")
_tidx = {id(t): i for i, t in enumerate(_m.land_tiles.values())}
NT, NN, NE = 19, 54, len(_EDGES)
A = np.zeros((NN, NT), np.float32)          # node -> adjacent land tiles
for n, tiles in _m.adjacent_tiles.items():
    for t in tiles:
        if id(t) in _tidx:
            A[n, _tidx[id(t)]] = 1
EU = np.array([e[0] for e in _EDGES])
EV = np.array([e[1] for e in _EDGES])
INC = np.zeros((NE, NN), np.float32)        # edge -> its two nodes
INC[np.arange(NE), EU] = 1
INC[np.arange(NE), EV] = 1
ADJ = (INC.T @ INC > 0).astype(np.float32)  # node adjacency
np.fill_diagonal(ADJ, 0)
OHU = np.eye(NN, dtype=np.float32)[EU]      # edge -> one-hot of its first / second node
OHV = np.eye(NN, dtype=np.float32)[EV]


# ----------------------------------------------------------------------------------------------
# Step 1: Game -> 639-float observation
# ----------------------------------------------------------------------------------------------
_static_cache = {}


def _static(cmap):
    """The map-only part of the vector (tiles without robber, ports) and coord -> tile index,
    cached per CatanMap object and bounded so a long run does not pin every map it has seen."""
    key = id(cmap)
    hit = _static_cache.get(key)
    if hit is not None and hit[0] is cmap:
        return hit[1], hit[2]
    v = np.zeros(206, np.float32)
    coords = []
    for t, (coord, tile) in enumerate(cmap.land_tiles.items()):
        coords.append(coord)
        if tile.resource is None:
            v[t * 8 + 5] = 1
        else:
            v[t * 8 + _RES_INDEX[tile.resource]] = 1
            v[t * 8 + 6] = _PIPS[tile.number]
    for p in sorted(cmap.ports_by_id.values(), key=lambda p: p.id):
        v[152 + p.id * 6 + (5 if p.resource is None else _RES_INDEX[p.resource])] = 1
    coord_index = {c: i for i, c in enumerate(coords)}
    if len(_static_cache) > 64:
        _static_cache.clear()
    _static_cache[key] = (cmap, v, coord_index)
    return v, coord_index


def _ports(board, color):
    f = [0.0] * 6
    for r in board.get_player_port_resources(color):
        f[5 if r is None else _RES_INDEX[r]] = 1.0
    return f


def encode_game(game, color):
    """Observation of `game` from `color`'s seat (see the module docstring for the layout)."""
    st = game.state
    b = st.board
    ps = st.player_state
    me = st.color_to_index[color]
    op = 1 - me
    opp_color = st.colors[op]
    o = np.zeros(OBS_DIM, np.float32)
    static, coord_index = _static(b.map)
    o[:206] = static
    o[coord_index[b.robber_coordinate] * 8 + 7] = 1
    for n, (c, kind) in b.buildings.items():
        o[206 + n * 4 + (0 if c == color else 2) + (0 if kind == SETTLEMENT else 1)] = 1
    seen = set()
    for e, c in b.roads.items():          # catanatron stores each road under both orientations
        i = _EDGE_ID[e]
        if i in seen:
            continue
        seen.add(i)
        o[422 + i * 2 + (0 if c == color else 1)] = 1
    k = f"P{me}_"
    x = [ps[k + r + "_IN_HAND"] for r in RES]
    x += [ps[k + d + "_IN_HAND"] for d in OBS_DEVS]
    x += [ps[k + "PLAYED_" + d] for d in OBS_DEVS[:4]]
    x += [ps[k + "VICTORY_POINTS"], ps[k + "ACTUAL_VICTORY_POINTS"], ps[k + "ROADS_AVAILABLE"],
          ps[k + "SETTLEMENTS_AVAILABLE"], ps[k + "CITIES_AVAILABLE"],
          b.road_color == color, bool(ps[k + "HAS_ARMY"]), b.road_lengths.get(color, 0),
          bool(ps[k + "HAS_ROLLED"]), bool(ps[k + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])]
    x += [bool(ps[k + d + "_OWNED_AT_START"]) for d in OBS_DEVS[:4]]
    x += _ports(b, color)
    x.append(st.discard_counts[me])
    k = f"P{op}_"
    x += [ps[k + r + "_IN_HAND"] for r in RES]
    x.append(sum(ps[k + d + "_IN_HAND"] for d in OBS_DEVS))
    x += [ps[k + "PLAYED_" + d] for d in OBS_DEVS[:4]]
    x += [ps[k + "VICTORY_POINTS"], ps[k + "ROADS_AVAILABLE"], ps[k + "SETTLEMENTS_AVAILABLE"],
          ps[k + "CITIES_AVAILABLE"], b.road_color == opp_color, bool(ps[k + "HAS_ARMY"]),
          b.road_lengths.get(opp_color, 0)]
    x += _ports(b, opp_color)
    x += list(st.resource_freqdeck)
    x += [len(st.development_listdeck), st.num_turns]
    pr = [0] * 5
    pr[_PROMPT[st.current_prompt]] = 1
    x += pr
    x += [bool(st.is_road_building), st.free_roads_available,
          st.current_player_index == me]   # the viewing seat is to move
    o[566:] = x
    return o


# ----------------------------------------------------------------------------------------------
# Step 2a: the 76 base features (same order as evaluator.feat_names)
# ----------------------------------------------------------------------------------------------
def _spots(occ_any, settle, city, road, road_other):
    """Nodes I can build on now, and empty buildable nodes one free edge from my network."""
    mine = ((settle + city) > 0) | ((road @ INC) > 0)
    buildable = (occ_any == 0) & ((occ_any @ ADJ) == 0)
    now = mine & buildable
    free = (road + road_other) == 0
    cand = buildable & ~mine
    hit = free & ((mine[:, EU] & cand[:, EV]) | (mine[:, EV] & cand[:, EU]))
    one = ((hit & cand[:, EV]) @ OHV + (hit & cand[:, EU]) @ OHU) > 0
    return now, one


def _block(prod_node, npv, occ_any, settle, city, road, road_other, sc, ports, res, devs_n):
    """One player's 31 base features. sc: dict of scalar columns."""
    prod = np.matmul((settle + 2 * city)[:, None, :], prod_node)[:, 0, :]
    total = prod.sum(1, keepdims=True)
    distinct = (prod > 0).sum(1, keepdims=True).astype(np.float32)
    hand_n = res.sum(1, keepdims=True)
    now, one = _spots(occ_any, settle, city, road, road_other)
    best_now = np.where(now, npv, 0).max(1, keepdims=True)
    best_one = np.where(one, npv, 0).max(1, keepdims=True)
    cols = [sc["vp"], settle.sum(1, keepdims=True), city.sum(1, keepdims=True), sc["roads_left"],
            prod, total, distinct, ports, res, hand_n, (hand_n > 7).astype(np.float32),
            sc["knights"], sc["army"], sc["road_len"], sc["has_road"],
            now.sum(1, keepdims=True).astype(np.float32), best_now,
            one.sum(1, keepdims=True).astype(np.float32), best_one, devs_n]
    return np.concatenate([c.astype(np.float32) for c in cols], 1)


def base76(X, my_turn):
    """me (37), opponent (31), global (4): the evaluator.feats layout, from observations."""
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
    a = _block(prod_node, npv, occ, ms, mc, mr, orr, sm, me[:, 28:34], me[:, 0:5],
               me[:, 5:10].sum(1, keepdims=True))
    devs = me[:, [5, 9, 8, 6, 7]]          # feats order: KNIGHT VP ROAD_BUILDING YOP MONOPOLY
    a = np.concatenate([a, devs, c(me, 15)], 1)
    # opp: res 0-4, dev count 5, played 6-9, vp 10, roads 11, setts 12, cities 13, LR 14, LA 15,
    # road_len 16, ports 17-22
    so = dict(vp=c(op, 10), roads_left=c(op, 11), knights=c(op, 6), army=c(op, 15),
              road_len=c(op, 16), has_road=c(op, 14))
    b = _block(prod_node, npv, occ, os_, oc, orr, mr, so, op[:, 17:23], op[:, 0:5], c(op, 5))
    # global: bank 0-4, deck 5, num_turns 6, prompt 7-11, rb 12, free 13, my_turn 14
    gl = np.concatenate([c(g, 5), c(g, 6) / 100.0, np.asarray(my_turn, np.float32)[:, None],
                         (g[:, 7:9].sum(1, keepdims=True) > 0).astype(np.float32)], 1)
    return np.concatenate([a, b, gl], 1)


# ----------------------------------------------------------------------------------------------
# Step 2b: the 47 extra features
# ----------------------------------------------------------------------------------------------
# Cost rows in RES order WOOD BRICK SHEEP WHEAT ORE: settlement, city, road, dev card.
COST = np.array([[1, 1, 1, 1, 0], [0, 0, 0, 2, 3], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], np.float32)
N_EXTRA = 47


def _deficit(res, ports):
    """Cards still missing for each build after greedily trading surplus at the best rate."""
    rate = np.where(ports[:, :5] > 0, 2.0, np.where(ports[:, 5:6] > 0, 3.0, 4.0)).astype(np.float32)
    out = []
    for c in COST:
        missing = np.maximum(c - res, 0).sum(1)
        trades = np.floor(np.maximum(res - c, 0) / rate).sum(1)
        out.append(np.maximum(missing - trades, 0))
    return np.stack(out, 1)


def extra47(X):
    """affordability  deficit and can-build-now for settlement / city / road / dev, both players
                       (the opponent's hand is public, so its next-turn options are a feature)
       race           opponent's expected hidden VP, its estimated total, the VP gap, roads still
                       needed for Longest Road and knights for Largest Army, both sides
       robber         production each side loses to the robber now, and how concentrated each
                       side's production is on one tile
       contested      spots one road away for both players: count and best pips
       pieces, phase  settlements/cities left, has_rolled, played dev this turn, discards pending,
                       prompt one-hot, free roads, bank per resource
    """
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


def features123(X):
    """(n, 639) observations -> (n, 123): base76 (my_turn read from the vector) + extra47."""
    X = np.asarray(X, np.float32)
    return np.concatenate([base76(X, X[:, 638]), extra47(X)], 1)
