"""Lockstep parity: catanatron vs the C engine (engine/).

Every game: a catanatron Game is created from a seed, its board, seating and dev deck are loaded into
the engine, and both are driven by the same random policy. catanatron draws the random outcomes
(dice, robber steals); the engine is given the same ones. After EVERY action (no forced-move
autoplay) we assert:
  * identical legal-action sets, as indices into catanatron's 332-action space
  * identical full state (engine/bridge.py:serialize vs cat_serialize: hands, dev cards, flags,
    bank, deck order, buildings, roads, robber, road lengths, LR/LA owners, connected components,
    buildable nodes, turn/prompt/discard state, winner)
  * (with --obs) bit-identical observations from the C and numpy encoders, for both seats' views
and at the end that both agree the game is over, with the same winner.

  PYTHONHASHSEED=0 python -m tools.engine_parity --games 1000 --workers 4
"""
import argparse
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

import catan_kit  # noqa: E402,F401
from catanatron.game import Game, TURNS_LIMIT  # noqa: E402
from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402
from catanatron.apply_action import yield_resources  # noqa: E402
from catanatron.models.board import STATIC_GRAPH  # noqa: E402

from engine import Engine  # noqa: E402
from engine import bridge  # noqa: E402

# policy styles: uniform random, or weighted towards building (games that end, contested roads,
# armies, exhausted banks / decks / pieces)
W_BUILD = {ActionType.BUILD_ROAD: 6, ActionType.BUILD_SETTLEMENT: 30, ActionType.BUILD_CITY: 30,
           ActionType.BUY_DEVELOPMENT_CARD: 8, ActionType.PLAY_KNIGHT_CARD: 6,
           ActionType.PLAY_ROAD_BUILDING: 6, ActionType.PLAY_MONOPOLY: 2,
           ActionType.PLAY_YEAR_OF_PLENTY: 2, ActionType.MARITIME_TRADE: 1.5,
           ActionType.END_TURN: 1}
W_ROADS = {**W_BUILD, ActionType.BUILD_ROAD: 40, ActionType.BUILD_SETTLEMENT: 10}
STYLES = ["uniform", "build", "roads", "hoard"]


def choose(rng, actions, style):
    if style == "uniform" or len(actions) == 1:
        return actions[rng.randrange(len(actions))]
    if style == "hoard":   # rarely spend: big hands -> discards, bank shortages
        w = [0.2 if a.action_type in (ActionType.BUILD_ROAD, ActionType.MARITIME_TRADE) else 1.0
             for a in actions]
    else:
        table = W_BUILD if style == "build" else W_ROADS
        w = [table.get(a.action_type, 1.0) for a in actions]
    return rng.choices(actions, weights=w)[0]


def diff_fields(a, b):
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            name = bridge.SER_FIELDS[i] if i < len(bridge.SER_FIELDS) else str(i)
            out.append(f"{name}: catanatron={x} engine={y}")
    if len(a) != len(b):
        out.append(f"length {len(a)} vs {len(b)}")
    return out


AT = ActionType


def _pre(game, action):
    """What coverage needs from the state just before `action`."""
    st = game.state
    ps = st.player_state
    i = st.current_player_index
    k = f"P{i}_"
    pre = dict(road_color=st.board.road_color, bank=list(st.resource_freqdeck),
               deck=len(st.development_listdeck), roads=ps[k + "ROADS_AVAILABLE"],
               rolled=ps[k + "HAS_ROLLED"], army=[ps["P0_HAS_ARMY"], ps["P1_HAS_ARMY"]])
    if action.action_type == AT.BUILD_SETTLEMENT and not st.is_initial_build_phase:
        opp = st.colors[1 - i]
        pre["cut"] = sum(1 for v in STATIC_GRAPH.neighbors(action.value)
                         if st.board.roads.get((action.value, v)) == opp) == 2
    return pre


def _cover(cov, game, action, pre):
    """Tally the rule corners this step exercised (catanatron's view)."""
    def hit(k):
        cov[k] = cov.get(k, 0) + 1
    st = game.state
    t = action.action_type
    b = st.board
    hit("action:" + t.name)
    if t == AT.DISCARD_RESOURCE:
        pass
    elif t == AT.MOVE_ROBBER and action.value[1] is not None:
        hit("robber_steal")
    elif t == AT.PLAY_KNIGHT_CARD and not pre["rolled"]:
        hit("knight_before_roll")
    elif t == AT.BUY_DEVELOPMENT_CARD and pre["deck"] == 1:
        hit("dev_deck_emptied")
    elif t == AT.PLAY_ROAD_BUILDING and pre["roads"] < 2:
        hit("road_building_with_<2_roads_left")
    elif t == AT.PLAY_YEAR_OF_PLENTY and len(action.value) == 1:
        hit("year_of_plenty_single_card")
    elif t == AT.MARITIME_TRADE:
        hit("trade_%d:1" % sum(1 for x in action.value[:4] if x is not None))
    elif t == AT.ROLL:
        n = sum(st.action_records[-1].result)
        if n != 7:
            payout, depleted = yield_resources(b, pre["bank"], n)
            if depleted:
                hit("bank_shortage_on_roll")
    elif t == AT.BUILD_SETTLEMENT and pre.get("cut"):
        hit("settlement_cuts_road")
    if t == AT.BUILD_ROAD and st.is_road_building is False and pre["roads"] == 1:
        hit("last_road_piece_built")
    if b.road_color != pre["road_color"]:
        hit("longest_road_new_owner" if pre["road_color"] is None else "longest_road_transfer")
        if b.road_length < 5:
            hit("longest_road_awarded_below_5")
    army = [st.player_state["P0_HAS_ARMY"], st.player_state["P1_HAS_ARMY"]]
    if army != pre["army"]:
        hit("largest_army_transfer" if any(pre["army"]) else "largest_army_new_owner")
    if any(st.player_state[f"P{i}_{d}_IN_HAND"] and not st.player_state[f"P{i}_{d}_OWNED_AT_START"]
           for i in (st.current_player_index,) for d in ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING")):
        hit("dev_card_blocked_same_turn")
    if game.winning_color() is not None and t == AT.BUY_DEVELOPMENT_CARD:
        hit("win_by_buying_vp_card")


def run_game(seed, check_obs=False, cov=None):
    """Returns (steps, None) or (steps, error message)."""
    random.seed(seed)
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    eng = bridge.load_into(Engine(), game)
    rng = random.Random(seed ^ 0x9E3779B9)
    style = STYLES[seed % len(STYLES)]
    if check_obs:
        from engine import obs_np
    steps = 0
    while True:
        cl = bridge.legal_indices(game)
        el = sorted(int(x) for x in eng.legal())
        if cl != el:
            return steps, (f"seed {seed} step {steps}: legal sets differ\n  only catanatron: "
                           f"{sorted(set(cl) - set(el))}\n  only engine: {sorted(set(el) - set(cl))}")
        cs = bridge.serialize(game)
        es = [int(x) for x in eng.serialize()]
        if cs != es:
            return steps, f"seed {seed} step {steps}: state differs\n  " + "\n  ".join(diff_fields(cs, es))
        if check_obs:
            a = eng.encode()
            b = obs_np.encode_game(game, game.state.current_color())
            if a.tobytes() != b.tobytes():
                bad = np.nonzero(a != b)[0]
                return steps, f"seed {seed} step {steps}: obs differs at {bad[:20]} c={a[bad[:5]]} np={b[bad[:5]]}"
        over_c = game.winning_color() is not None or game.state.num_turns >= TURNS_LIMIT
        if over_c != eng.done():
            return steps, f"seed {seed} step {steps}: done differs (catanatron {over_c})"
        if over_c:
            if cov is not None:
                w = game.winning_color()
                cov["game_end:" + ("turn_limit" if w is None else "win")] = \
                    cov.get("game_end:" + ("turn_limit" if w is None else "win"), 0) + 1
            return steps, None
        action = choose(rng, game.playable_actions, style)
        idx = bridge.action_index(action)
        pre = _pre(game, action) if cov is not None else None
        game.execute(action)
        eng.step(idx, bridge.outcome_of(game.state.action_records[-1]))
        if cov is not None:
            _cover(cov, game, action, pre)
        steps += 1


def worker(args):
    seeds, check_obs = args
    steps, errors, cov = 0, [], {}
    for s in seeds:
        n, err = run_game(s, check_obs, cov)
        steps += n
        if err:
            errors.append(err)
    return len(seeds), steps, errors, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--obs", action="store_true", help="also compare C vs numpy observations")
    ap.add_argument("--max-errors", type=int, default=20)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("warning: run with PYTHONHASHSEED=0 for reproducible catanatron runs", file=sys.stderr)
    seeds = list(range(a.seed0, a.seed0 + a.games))
    chunks = [(seeds[i:i + a.chunk], a.obs) for i in range(0, len(seeds), a.chunk)]
    t0 = time.time()
    games = steps = 0
    errors = []
    cov = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for g, s, errs, c in pool.imap_unordered(worker, chunks):
            for k, v in c.items():
                cov[k] = cov.get(k, 0) + v
            games += g
            steps += s
            errors += errs
            el = time.time() - t0
            print(f"\r{games}/{len(seeds)} games, {steps} steps, {len(errors)} divergent games, "
                  f"{el:.0f}s", end="", file=sys.stderr, flush=True)
            if len(errors) >= a.max_errors:
                pool.terminate()
                break
    print(file=sys.stderr)
    for e in errors[:a.max_errors]:
        print(e)
    print("coverage (events seen, catanatron side):")
    for k in sorted(cov):
        print(f"  {k:40s} {cov[k]}")
    print(f"RESULT games={games} steps={steps} divergent_games={len(errors)} "
          f"seconds={time.time() - t0:.0f}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
