# engine/: fast C Catan, rule-for-rule identical to our catanatron

A C engine for the 2-player, 10 VP, no player-trade Catan we play. It is meant for offline training:
PPO and self-play, value-net data generation and search experiments. It is **not** for submissions,
which must be pure Python. The submission-side piece is `obs_np.py`, the pure-numpy observation
encoder.

```
make -C engine                       # libcatan.{so,dylib} + ./engine/bench   (-O3 -march=native)
PYTHONHASHSEED=0 .venv/bin/python -m tools.engine_parity --games 1000 --workers 4 --obs
PYTHONHASHSEED=0 .venv/bin/python -m tools.engine_obs_test --games 100
PYTHONHASHSEED=0 .venv/bin/python -m tools.engine_bench
```
`import engine` builds the library on first use if it is missing or stale.

## Python API

```python
from engine import VecEnv, Engine, encode_states, N_ACTIONS, OBS_DIM   # 332, 639

env = VecEnv(1024, seed=0, autoplay=True, obs=True, autoreset=True)
# numpy buffers, filled in place by every call (zero copy):
#   env.obs (N,639) f32   env.mask (N,332) bool   env.cur (N,) seat to move
#   env.reward (N,) f32   +1/-1 for the seat that chose the action, when that ended the game
#   env.terminal / env.truncated (N,)  (truncated = 1000-turn cap, no winner)
#   env.winner (N,) seat that won on this step, else -1
steps = env.step(actions)          # int32 (N,) of catanatron action indices; returns engine steps
env.states                         # (N, 432) uint8: raw states, copy/snapshot with numpy

e = Engine(seed=123)               # one game; e.copy() is a 432-byte memcpy
e.legal()                          # int16 action indices
e.step(a, outcome=-1)              # outcome: dice sum for ROLL / resource for a steal; -1 = sample
e.autoplay(); e.done(); e.encode(seat=None); e.serialize()
encode_states(env.states[idx])     # batch-encode arbitrary states (search leaves)

from engine import bridge          # catanatron glue
bridge.load_into(Engine(), game)   # board, seating, dev deck of a fresh catanatron Game
bridge.action_index(action)        # catanatron Action -> index; bridge.ACTIONS_ARRAY is the inverse
from engine.obs_np import encode_game, encode_obs   # pure numpy, same vector as the C encoder
```

**Action contract.** Indices are catanatron.gym's `ACTIONS_ARRAY` with colours (RED, BLUE) on the
BASE map: 332 actions, sorted by `str(action)`. A policy trained here maps to the harness by
`bridge.ACTION_INDEX[(a.action_type, a.value)]` for each entry of `playable_actions`.
`MOVE_ROBBER` indices name the victim's colour, so seat-to-colour matters. Native games randomise
which colour sits first, as catanatron does. The observation is colour-free (me/opponent).

**Forced moves.** With `autoplay=True`, any position with exactly one legal action is played inside
the engine, as the harness does (PS §3.2). Its random outcome is sampled. Parity mode
(`Engine.step`) steps every action.

## Design

- **Topology** (`gen_topology.py` → `topology.h`, generated, committed). Node, edge, tile and port
  ids, land neighbours in `STATIC_GRAPH.neighbors` order, adjacent tiles per node, port node pairs,
  the official spiral order and the full action table are all dumped from catanatron's own map
  code. Edge ids follow `get_edges()`. Tile ids follow `land_tiles` order.
- **State** (`catan.h`, 432 bytes, plain struct, so a copy is a memcpy). It holds node bitboards
  (settlements and cities per seat), 128-bit road bitboards, buildable nodes, each player's list of
  connected components (as node masks, in catanatron's list order), hands, dev cards in hand,
  played and owned-at-start, the bank, the dev deck in order plus a pointer, tiles, numbers, ports,
  the robber, `road_lengths`, LR/LA owners, VPs, and the turn, prompt, discard and road-building
  state. It also carries a per-env xorshift64* RNG.
- **Moves** are generated from bitboards on demand: buildable edges are the union of component
  nodes' incident edges minus all roads, and settlements are component nodes ∩ buildable. It writes
  a list of action indices, which becomes the 332-entry mask. Recomputing is cheaper than
  maintaining the sets incrementally at this size.
- **Production** loops over the ≤2 tiles carrying the number, using popcounts on tile node masks.
  It applies catanatron's shortage rule: if the bank cannot pay every claim on a resource, nobody
  receives that resource.
- **Longest Road.** This is a DFS for the longest trail (distinct edges), never entering enemy
  nodes. It runs only when a road is built, or when a settlement lands on an opponent's road
  junction. The component bookkeeping is replicated exactly, bugs included (below), because it
  decides which nodes count as "connected".
- **RNG / replay.** In training the engine samples dice (2d6), steals (uniform over the victim's
  cards) and the dev-deck shuffle itself. In parity, `cat_load` takes catanatron's board, seating
  and *deck order*, and `cat_step(a, outcome)` takes catanatron's dice sum and stolen card.
- **Board generation** (`cat_new_game`) samples from catanatron's distribution: a uniform
  permutation of the 19 tile resources and the 9 ports, numbers along the official spiral skipping
  the desert, the robber on the desert, a random first seat and a uniformly shuffled deck.

### catanatron behaviours reproduced on purpose (they are the rules we are scored under)

- **Discards** are one card at a time (`DISCARD_RESOURCE`), using `discard_counts`, with the
  lowest seat first. The robber moves after the discards finish.
- **Robber**: you must move it, to one action per tile, with the victim being the opponent if they
  have a building on the tile and ≥1 card. No friendly robber.
- **Dev cards** can be played before or after rolling, one per turn, and only if owned at the start
  of the turn. The owned flags are refreshed only at the player's own END_TURN.
- **Year of Plenty**: if a pair cannot be drawn from the bank, the single-card options appear
  (deduplicated).
- **Road Building** needs ≥1 buildable edge to be played. It ends early when pieces or edges run
  out.
- **Maritime trades**: only the best rate per resource is offered (4:1 / 3:1 / 2:1), and only for
  resources the bank still has.
- The **second settlement** pays one card per adjacent producing tile.
- **Longest Road**: a new owner needs ≥5 and strictly more than the current `road_length`.
  - When a settlement *cuts* an opponent road (2 of their roads at the node), catanatron rebuilds
    that player's components by DFS and resets `road_color, road_length = max(road_lengths)`.
    Ties go to seat 0, which is first in dict order. **This can award Longest Road with fewer than
    5 roads**, and the engine does the same.
  - DFS-rebuilt components include the cutting enemy node, so the cut player can later build roads
    out of it.
  - A settlement at the end of one opponent road removes that node from the opponent's first
    component containing it.
  - `road_lengths` only grows on road builds. `LONGEST_ROAD_LENGTH` in player_state is a copy that
    is refreshed only on non-initial builds.
- **Largest Army**: the first to 3 knights takes it, and it transfers only to a strictly larger
  army.
- **Win check** comes after every action, on *actual* VP (hidden VP cards count at once, so
  buying the 10th point wins immediately), for any seat. The game stops when `num_turns` reaches
  1000. During setup, `num_turns` also advances.
- `is_moving_knight` is sticky (set, never cleared), exactly as in catanatron. It does not affect
  play.

## Parity

`tools/engine_parity.py` creates a catanatron `Game(seed)` and loads it into the engine. A random
policy then drives both, in one of four styles cycled by seed: uniform, build-heavy, road-heavy,
or hoarding (which forces discards and bank shortages). After **every** action it asserts
identical legal-action index sets, an identical full canonical state (`bridge.serialize` ==
`cat_serialize`: 285 ints covering everything above, including component lists and deck order),
the same done flag and winner, and, with `--obs`, bit-identical C and numpy observations.
Injected faults (a wrong dice sum) are caught at the first step they matter.

| run | games | steps | divergent games |
|---|---|---|---|
| laptop, seeds 1–2000 | 2,000 | 1,696,768 | 0 |
| laptop, seeds 1–1000, `--obs` | 1,000 | 862,200 | 0 |
| server, seeds 1,000,000–1,099,999, `--obs` | **100,000** | **85,625,765** | **0** |

Coverage on the 100k run (events counted on the catanatron side):

| event | count |
|---|---|
| bank_shortage_on_roll | 594 |
| dev_card_blocked_same_turn | 1215578 |
| dev_deck_emptied | 10649 |
| game_end:turn_limit | 752 |
| game_end:win | 99248 |
| knight_before_roll | 625232 |
| largest_army_new_owner | 92808 |
| largest_army_transfer | 14790 |
| last_road_piece_built | 89766 |
| longest_road_awarded_below_5 | 73 |
| longest_road_new_owner | 97475 |
| longest_road_transfer | 23269 |
| road_building_with_<2_roads_left | 2792 |
| robber_steal | 1372832 |
| settlement_cuts_road | 3118 |
| trade_2:1 | 938799 |
| trade_3:1 | 2191961 |
| trade_4:1 | 3134915 |
| win_by_buying_vp_card | 27897 |

`tools/engine_obs_test.py` compares the C encoder (both seats) with `obs_np` on the true game,
and `obs_np` on the harness's masked view (`catan_kit.obs.masked_copy`, opponent cards and deck
redrawn) with the C encoding of the true state. Result: 97,009 positions, all bit-identical. This
confirms that no hidden information enters the observation.

## Observation (639 float32, view of the seat to move)

All values are raw small integers. Normalise in the network. That is also why C and numpy agree
bit for bit.

| range | content |
|---|---|
| 0–151 | 19 tiles × 8: resource one-hot (WOOD BRICK SHEEP WHEAT ORE), desert, pips (6−\|7−n\|), robber |
| 152–205 | 9 ports × 6: resource one-hot, 3:1 |
| 206–421 | 54 nodes × 4: my settlement, my city, opp settlement, opp city |
| 422–565 | 72 edges × 2: my road, opp road |
| 566–600 | me: resources(5), dev cards in hand(5: K YOP MONO RB VP), played(4), public VP, actual VP, roads/settlements/cities left, has LR, has LA, road length, has rolled, played dev this turn, dev owned at turn start(4), ports(6), discards left |
| 601–623 | opp: resources(5), dev card **count**, played(4), public VP, roads/settlements/cities left, has LR, has LA, road length, ports(6) |
| 624–638 | bank(5), deck left, num_turns, prompt one-hot(5), road building, free roads left, **to move**: 1 if the viewing seat is the one to move (seat-relative; before 2026-10-17 this was `cur == turn`, which is the same for both seats and hid whose move it was) |

Opponent resources are included because they are public (PS §2). The opponent's dev-card types
and hidden VP are not. `obs_np.encode_obs(obs)` costs about 8 µs in Python.

## Benchmarks (one core)

`tools/engine_bench.py` and `engine/bench`. "Steps" are engine actions, including forced moves.
"Decisions" are `VecEnv.step` rows.

| | M5 laptop | shared server (EPYC 7742, loaded, load avg ~100) |
|---|---|---|
| C uniform-random self-play | **25,900 games/s, 23.7 M steps/s** | 6,900 games/s, 6.3 M steps/s |
| `VecEnv.step` C call, no obs (1024 envs) | 6.5 M decisions/s, 23.0 M steps/s | — |
| `VecEnv.step` C call, with obs | 4.3 M decisions/s, 15.1 M steps/s | 0.83 M decisions/s, 3.0 M steps/s (bench.c, incl. mask scan) |
| `VecEnv.step` end-to-end from Python, random actions via numpy | 1.05–1.15 M decisions/s | — |
| catanatron `copy()+execute()` | 23 µs/step, 43 k steps/s | — |
| catanatron random games | 128 games/s, 114 k steps/s | — |

The engine is about **200× faster than catanatron per step**. End to end, the cost is already
dominated by numpy *sampling random actions* over a (1024, 332) mask, so any policy forward pass
will be the bottleneck. One core supplies over 1M decisions/s, compared with roughly 10–100k/s per
core for an MLP over 639 inputs on CPU. That is why no further profiling or micro-optimisation was
done. If it is ever needed, the candidates are an incremental legal-edge mask, skipping the
memset on unchanged mask rows, and multi-threading `vec_step` over envs (it is embarrassingly
parallel).

## A later GPU port (not built)

What JAX or CUDA would need:
- The state is already flat and fixed-size, with no pointers, so it maps to struct-of-arrays
  tensors directly: 128-bit edge masks become 2×u64 or a 72-bool array, and node masks become u64
  or 54 bools.
- **Branch-free step.** Compute every action type's effect and select by action type (a
  `lax.switch`, or masked updates), with the prompt logic as `where`s.
- **Move generation** as dense tensor ops: buildable edges = `node_edge_incidence @ comp_union`,
  and masks via comparisons of hands against cost tables.
- **Longest Road** is the hard part. The recursive trail DFS needs either a bounded-depth explicit
  stack (≤15 roads) per env, or a precomputed table. It also has to keep catanatron's
  component-list bookkeeping, plow quirks included, to stay identical. A practical option is to
  run LR on the CPU for the rare steps that need it, or a warp-per-env DFS in CUDA.
- RNG via a counter-based generator (threefry / philox) per env. Auto-reset by `where(done,
  fresh_state, state)`.
- `cat_serialize` gives the canonical comparison point, so the GPU port can reuse this parity
  harness unchanged, as a third engine.
