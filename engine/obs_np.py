"""Pure-Python/numpy observation encoder: the same vector as engine/catan.c:cat_encode, built from a
catanatron Game (the harness's masked view, a sample_world, or a real game) or an Obs.

No compiled code, so it can ship inside a submission. Reads only what `color` may see: the
opponent's development cards enter as a count, its hidden victory points not at all.

    from engine.obs_np import encode_game, encode_obs, OBS_DIM
    x = encode_obs(obs)                  # float32[OBS_DIM]
    x = encode_game(game, color)

Layout (raw small integers as float32; normalise inside the network):
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
Tile, port, node and edge ids are catanatron's (land_tiles order, port ids, node ids, get_edges order).
"""
import numpy as np

from catanatron.models.board import get_edges
from catanatron.models.enums import ActionPrompt, SETTLEMENT, CITY

OBS_DIM = 639
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEVS = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]
_RI = {r: i for i, r in enumerate(RES)}
_EDGE_ID = {}
for _i, _e in enumerate(get_edges(None)):
    _EDGE_ID[_e] = _i
    _EDGE_ID[(_e[1], _e[0])] = _i
_PROMPT = {ActionPrompt.BUILD_INITIAL_SETTLEMENT: 0, ActionPrompt.BUILD_INITIAL_ROAD: 1,
           ActionPrompt.PLAY_TURN: 2, ActionPrompt.DISCARD: 3, ActionPrompt.MOVE_ROBBER: 4}
_PIPS = {n: 6 - abs(7 - n) for n in range(2, 13)}

# static board part, cached per CatanMap object
_static_cache = {}


def _static(cmap):
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
            v[t * 8 + _RI[tile.resource]] = 1
            v[t * 8 + 6] = _PIPS[tile.number]
    for p in sorted(cmap.ports_by_id.values(), key=lambda p: p.id):
        v[152 + p.id * 6 + (5 if p.resource is None else _RI[p.resource])] = 1
    coord_index = {c: i for i, c in enumerate(coords)}
    if len(_static_cache) > 64:
        _static_cache.clear()
    _static_cache[key] = (cmap, v, coord_index)
    return v, coord_index


def _ports(board, color):
    f = [0.0] * 6
    for r in board.get_player_port_resources(color):
        f[5 if r is None else _RI[r]] = 1.0
    return f


def encode_game(game, color, out=None):
    """Observation of `game` from `color`'s seat (normally the colour to move)."""
    st = game.state
    b = st.board
    ps = st.player_state
    me = st.color_to_index[color]
    op = 1 - me
    opp_color = st.colors[op]
    o = np.zeros(OBS_DIM, np.float32) if out is None else out
    if out is not None:
        o[:] = 0
    static, coord_index = _static(b.map)
    o[:206] = static
    o[coord_index[b.robber_coordinate] * 8 + 7] = 1
    for n, (c, kind) in b.buildings.items():
        o[206 + n * 4 + (0 if c == color else 2) + (0 if kind == SETTLEMENT else 1)] = 1
    seen = set()
    for e, c in b.roads.items():
        i = _EDGE_ID[e]
        if i in seen:
            continue
        seen.add(i)
        o[422 + i * 2 + (0 if c == color else 1)] = 1
    k = f"P{me}_"
    x = [ps[k + r + "_IN_HAND"] for r in RES]
    x += [ps[k + d + "_IN_HAND"] for d in DEVS]
    x += [ps[k + "PLAYED_" + d] for d in DEVS[:4]]
    x += [ps[k + "VICTORY_POINTS"], ps[k + "ACTUAL_VICTORY_POINTS"], ps[k + "ROADS_AVAILABLE"],
          ps[k + "SETTLEMENTS_AVAILABLE"], ps[k + "CITIES_AVAILABLE"],
          b.road_color == color, bool(ps[k + "HAS_ARMY"]), b.road_lengths.get(color, 0),
          bool(ps[k + "HAS_ROLLED"]), bool(ps[k + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])]
    x += [bool(ps[k + d + "_OWNED_AT_START"]) for d in DEVS[:4]]
    x += _ports(b, color)
    x.append(st.discard_counts[me])
    k = f"P{op}_"
    x += [ps[k + r + "_IN_HAND"] for r in RES]
    x.append(sum(ps[k + d + "_IN_HAND"] for d in DEVS))
    x += [ps[k + "PLAYED_" + d] for d in DEVS[:4]]
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


def encode_obs(obs, out=None):
    """Observation for a catan_kit Obs (the agent's view)."""
    return encode_game(obs._game, obs.me, out)
