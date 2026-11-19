"""Fast, symmetric position features for a learned evaluator.

feats(game, me) -> list of floats, from `me`'s perspective. Only uses what `me` can see (the
opponent's dev cards appear as a count, its hidden VPs not at all), so features computed on a
sampled world equal those on the true game.
"""
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import SETTLEMENT, CITY
from catanatron.models.map import number_probability

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
NEIGH = {n: list(STATIC_GRAPH.neighbors(n)) for n in STATIC_GRAPH.nodes}

_map_cache = {}


def _node_prod(cmap):
    """node -> list of (tile_coordinate, resource_index, probability)."""
    key = id(cmap)
    hit = _map_cache.get(key)
    if hit is not None and hit[0] is cmap:
        return hit[1]
    if len(_map_cache) > 8:
        _map_cache.clear()   # the harness unpickles a fresh map every decision: keep the cache bounded
    coord_of = {id(t): c for c, t in cmap.land_tiles.items()}
    np_ = {}
    for n, tiles in cmap.adjacent_tiles.items():
        np_[n] = [(coord_of[id(t)], RES.index(t.resource), number_probability(t.number))
                  for t in tiles if t.resource is not None]
    _map_cache[key] = (cmap, np_)
    return np_


def _player_block(st, board, nprod, color, idx, robber, own):
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
    # expansion: buildable nodes now, and best/sum production of nodes 1 road away
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
    st = game.state
    board = st.board
    nprod = _node_prod(board.map)
    robber = board.robber_coordinate
    i = st.color_to_index[me]
    j = 1 - i
    opp = st.colors[j]
    a = _player_block(st, board, nprod, me, i, robber, True)
    b = _player_block(st, board, nprod, opp, j, robber, False)
    cur = st.current_color() == me
    g = [
        len(st.development_listdeck) + 0.0,
        st.num_turns / 100.0,
        1.0 if cur else 0.0,
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
