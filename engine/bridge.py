"""catanatron <-> engine glue: action indices, board loading, canonical state serialisation.

Pure Python; imports catanatron. Used by the parity test and by anything that needs to move a
position between the two engines.
"""
import importlib.util
import os

import catanatron
from catanatron.models.enums import (DEVELOPMENT_CARDS, RESOURCES, SETTLEMENT, CITY, ActionPrompt,
                                     ActionType)
from catanatron.models.player import Color

MAXC = 12
COLOR_CODE = {Color.RED: 0, Color.BLUE: 1}
PROMPT_CODE = {ActionPrompt.BUILD_INITIAL_SETTLEMENT: 0, ActionPrompt.BUILD_INITIAL_ROAD: 1,
               ActionPrompt.PLAY_TURN: 2, ActionPrompt.DISCARD: 3, ActionPrompt.MOVE_ROBBER: 4}
DEV_CODE = {d: i for i, d in enumerate(DEVELOPMENT_CARDS)}   # KNIGHT YOP MONO RB VP
RES_CODE = {r: i for i, r in enumerate(RESOURCES)}            # WOOD BRICK SHEEP WHEAT ORE


def _load_action_space():
    p = os.path.join(os.path.dirname(catanatron.__file__), "gym", "envs", "action_space.py")
    spec = importlib.util.spec_from_file_location("catan_action_space", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ACTIONS_ARRAY = _load_action_space().get_action_array((Color.RED, Color.BLUE), "BASE")
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS_ARRAY)}
assert len(ACTIONS_ARRAY) == 332


def action_index(action):
    """catanatron Action -> index into the 332-action space."""
    return ACTION_INDEX[(action.action_type, action.value)]


def legal_indices(game):
    return sorted(action_index(a) for a in game.playable_actions)


def board_arrays(game):
    """(tile_res, tile_num, port_res, colors, deck, robber) for Engine.load."""
    st = game.state
    m = st.board.map
    tiles = list(m.land_tiles.items())
    tile_res = [-1 if t.resource is None else RES_CODE[t.resource] for _, t in tiles]
    tile_num = [0 if t.number is None else t.number for _, t in tiles]
    ports = sorted(m.ports_by_id.values(), key=lambda p: p.id)
    port_res = [-1 if p.resource is None else RES_CODE[p.resource] for p in ports]
    colors = [COLOR_CODE[c] for c in st.colors]
    deck = [DEV_CODE[d] for d in st.development_listdeck]
    coords = [c for c, _ in tiles]
    robber = coords.index(st.board.robber_coordinate)
    return tile_res, tile_num, port_res, colors, deck, robber


def load_into(engine, game, seed=0):
    """Load a freshly created catanatron game (no moves yet) into an Engine."""
    assert len(game.state.action_records) == 0, "load_into expects a game at its start"
    engine.load(*board_arrays(game), seed=seed)
    return engine


def outcome_of(record):
    """The random outcome catanatron drew for an executed action, in the engine's encoding."""
    t = record.action.action_type
    if t == ActionType.ROLL:
        return record.result[0] + record.result[1]
    if t == ActionType.MOVE_ROBBER:
        return -1 if record.result is None else RES_CODE[record.result]
    return -1


def _mask(nodes):
    m = 0
    for n in nodes:
        m |= 1 << n
    return m


def serialize(game):
    """Canonical int list of a catanatron state; must match engine/catan.c:cat_serialize."""
    st = game.state
    ps = st.player_state
    b = st.board
    colors = st.colors
    o = []
    for i in range(2):
        k = f"P{i}_"
        o += [ps[k + r + "_IN_HAND"] for r in RESOURCES]
        o += [ps[k + d + "_IN_HAND"] for d in DEVELOPMENT_CARDS]
        o += [ps[k + "PLAYED_" + d] for d in DEVELOPMENT_CARDS[:4]]
        o += [int(ps[k + d + "_OWNED_AT_START"]) for d in DEVELOPMENT_CARDS[:4]]
        o += [ps[k + "VICTORY_POINTS"], ps[k + "ACTUAL_VICTORY_POINTS"], ps[k + "ROADS_AVAILABLE"],
              ps[k + "SETTLEMENTS_AVAILABLE"], ps[k + "CITIES_AVAILABLE"], int(ps[k + "HAS_ROAD"]),
              int(ps[k + "HAS_ARMY"]), int(ps[k + "HAS_ROLLED"]),
              int(ps[k + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]), ps[k + "LONGEST_ROAD_LENGTH"]]
    o += list(st.resource_freqdeck)
    deck = [DEV_CODE[d] for d in st.development_listdeck]
    o += [len(deck)] + deck + [-1] * (25 - len(deck))
    seat = {c: i for i, c in enumerate(colors)}
    for n in range(54):
        bld = b.buildings.get(n)
        o.append(0 if bld is None else (1 if bld[1] == SETTLEMENT else 2) + 2 * seat[bld[0]])
    from catanatron.models.board import get_edges
    for e in _EDGES:
        c = b.roads.get(e)
        o.append(0 if c is None else 1 + seat[c])
    coords = list(b.map.land_tiles.keys())
    o.append(coords.index(b.robber_coordinate))
    o += [b.road_lengths.get(colors[0], 0), b.road_lengths.get(colors[1], 0)]
    o += [-1 if b.road_color is None else seat[b.road_color], b.road_length]
    army = -1
    for i in range(2):
        if ps[f"P{i}_HAS_ARMY"]:
            army = i
            break
    o.append(army)
    bm = _mask(b.board_buildable_ids)
    o += [bm & 0x7FFFFFFF, bm >> 31]
    for c in colors:
        comps = b.connected_components[c]
        o.append(len(comps))
        for i in range(MAXC):
            m = _mask(comps[i]) if i < len(comps) else 0
            o += [m & 0x7FFFFFFF, m >> 31]
    o += [st.num_turns, st.current_player_index, st.current_turn_index,
          PROMPT_CODE[st.current_prompt], int(st.is_initial_build_phase), int(st.is_discarding),
          st.discard_counts[0], st.discard_counts[1], int(st.is_moving_knight),
          int(st.is_road_building), st.free_roads_available]
    for c in colors:
        s = st.buildings_by_color[c][SETTLEMENT]
        o.append(s[-1] if st.is_initial_build_phase and s else -1)
    w = game.winning_color()
    o.append(-1 if w is None else seat[w])
    return o


def _edges():
    from catanatron.models.board import get_edges
    return [tuple(sorted(e)) for e in get_edges(None)]


_EDGES = _edges()

SER_FIELDS = (
    [f"P{i}_{f}" for i in range(2) for f in
     [f"res_{r}" for r in RESOURCES] + [f"dev_{d}" for d in DEVELOPMENT_CARDS] +
     [f"played_{d}" for d in DEVELOPMENT_CARDS[:4]] + [f"owned_{d}" for d in DEVELOPMENT_CARDS[:4]] +
     ["vp", "actual_vp", "roads_av", "settle_av", "city_av", "has_road", "has_army", "has_rolled",
      "has_played_dev", "lr_len"]] +
    [f"bank_{r}" for r in RESOURCES] + ["deck_n"] + [f"deck_{i}" for i in range(25)] +
    [f"node_{n}" for n in range(54)] + [f"edge_{e}" for e in range(72)] +
    ["robber", "board_rl0", "board_rl1", "road_color", "road_length", "army_owner",
     "buildable_lo", "buildable_hi"] +
    [f"comp{p}_{x}" for p in range(2) for x in ["n"] + [f"{i}{h}" for i in range(MAXC) for h in "lh"]] +
    ["num_turns", "cur", "turn", "prompt", "is_initial", "is_discarding", "dc0", "dc1",
     "is_moving_knight", "is_road_building", "free_roads", "last_settle0", "last_settle1", "winner"])
