# agent — a readable copy of the shipped final Catan agent

Same decisions as the shipped final submission (`submission/both_roads_v12_cachefix.zip`,
SHA-256 `46d0c266…`). The ~60-file layered `codex/` tree is flattened into
the six modules that are on its execution path. No logic, constant or iteration order was
changed; `check_parity.py` proves it.

## What the agent does

| Situation | Choice | Module |
|---|---|---|
| setup settlement or road, > 1 legal action (any clock) | reply-aware rollout over the 123-feature net | `opening.SetupPlanner.choose` |
| < 30 ms of CPU allowance left | winning build if any, else city > settlement > end turn > roll > knight | `search.emergency` |
| main game | exact one-ply expectation + tactical bonuses + same-turn beam search (depth 3, beam 3, 80 nodes, 6 ms) | `search.search` |

**Setup** (`opening.py`): every legal placement is played on a copy and scored by the setup net
(`value_123.npz`, P(win) from the 123 features of `features123.py`). The best 5 are each
completed to the end of the setup phase, both players choosing greedily by the same net from
their own seat (settlement choices pre-filtered to the 10 highest-pip nodes), and the candidate
whose finished setup is rated best for us is played. This answers all four setup decisions,
settlements and roads.

**Main game** (`search.py`, `evaluator.py`, `outcomes.py`), leaf value:

    leaf   = heuristic(game, me) + 8 * logit(P_win)        P_win = mean-logit ensemble of two numpy MLPs
    value  = leaf - 0.35 * public_threat(game, me)          denial: the rival is about to score
    robber = value + 0.35 * 6 * (blocked_need - blocked_need_before)   only for robber placements

`P_win` comes from `value_v2r1.npz` and `value_v5.npz` (76-feature nets). All three nets were
trained offline and are only run here. The search samples one world (`obs.sample_world` with a
private `random.Random(0)`); the evaluator reads only public features of the opponent.

## Constants and where each number comes from (paths are inside the shipped zip)

| Constant | Value | Origin |
|---|---|---|
| `WEIGHTS` prod / expand / army / hand | 18.0 / 0.71404 / 2.49792 / 0.9 | `PARAMS` in `codex/evaluator_swap/baseline.py` (offline tuning run); `econ = 0.0` there switches the economic term off, so it is omitted |
| `LEAF_K` | 8 | `K` in `codex/residual_sweep/runtime.py` |
| `DENIAL`, `MONOPOLY_BONUS`, `TACTICAL_ARMY` | 0.35, 0.3037, 1.4 | `PARAMS` `denial`, `monopoly`, `tactical_army` (profile overrides) |
| `DEPTH, BEAM, CAP_S, MAX_NODES` | 3, 3, 0.006, 80 | `'army_exact'` preset of `codex/residual_sweep/guarded.py` |
| `HARD_CAP_S, RESERVE_S, EMERGENCY_S` | 0.060, 0.015, 0.030 | `guarded.decide`: `min(.060, max(0, left-.015))`, fallback when `left < .030`; the harness clock is 0.3 s + 12.5 ms per decision, capped at 1.5 s |
| `SEARCH_RESERVE_S, SEARCH_FRACTION` | 0.10, 0.04 | `guarded.decide`: search gets 4 % of the allowance above 100 ms, never more than `CAP_S` |
| `TOP_K, ROLLOUT_WIDTH` | 5, 10 | `OPEN_K`, `OPEN_M` defaults of `codex/v12_open/frozen/agent.py` (env overrides removed) |
| `_map_cache` bound | cleared at 8 maps | `codex/evaluator_swap/frozen/feats.py` (memory only) |
| `_static_cache` bound | cleared above 64 maps | `codex/novel_quick/frozen/obs_np.py` (memory only) |

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are forced to 1 before numpy is
imported, as the shipped `agent.py` does (the harness charges every thread's CPU). There is no
warm-up: the shipped agent builds its v12 opener with `__new__`, which skips v12's warm-up.

## What was dropped, and why that changes nothing

* The EABR heuristic opener (`codex/evaluator_swap/frozen/opening.py`) and its initial-phase
  branch in `guarded.decide`: every setup decision with > 1 action is answered by the v12
  rollout first, and the one-action case returns 0 before that branch.
* The throw-away `Baseline()` agent the shipped `OpeningAgent` constructs and then replaces:
  its only state is its own unused `random.Random(0)`; it touches no global state.
* `root_v` / `continuation_v` hooks (both `None` in the shipped configuration), the
  `if weights['econ']` term (weight 0), env knobs, v12's unused main-game search and robber.
* The per-decision `_npv` cache: node pips are a pure function of the map, now computed once
  per setup decision. Value-net batches are kept exactly as shipped (same rows, same order).

## Files

    agent.py        entry point: class Agent, __init__ (three nets), decide (dispatch)
    opening.py      setup: one-ply scoring, top 5, greedy two-sided setup rollout
    features123.py  639-float observation encoder and the 123 setup-net features
    search.py       root scoring, bonuses, beam search, clock budget, emergency fallback
    evaluator.py    76 features, numpy MLPs, handcrafted heuristic, overlays, Evaluator
    outcomes.py     exact one-step chance outcomes (dice, dev draw, steal)
    value_v2r1.npz, value_v5.npz, value_123.npz   weights, byte-identical to the shipped
                    codex/residual_sweep/nets/value.npz, .../value_v5.npz, codex/v12_open/frozen/value.npz
    check_parity.py decision-for-decision comparison against the shipped zip
    submission/     the shipped zip, byte for byte
    LICENSE, NOTICE.md   unchanged

## Running

Parity against the shipped zip (read from `submission/` and hash-checked; both agents in one process, same
observations, patched clock; see the script's docstring):

    cd <repo>
    PYTHONHASHSEED=0 .venv/bin/python agent/check_parity.py --games 60
    PYTHONHASHSEED=0 .venv/bin/python agent/check_parity.py --games 30 --seed0 100 --cpu-left gates
    PYTHONHASHSEED=0 .venv/bin/python agent/check_parity.py --games 30 --seed0 200 --clock ticking

Official harness:

    PYTHONHASHSEED=0 .venv/bin/python -m catan_kit.evaluate --a agent/agent.py --b examples/f_agent.py --games 40 --seed0 990000
