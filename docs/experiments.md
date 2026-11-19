# Experiment log

The run log we kept during the event and the week after, lightly edited. Every row is a
head-to-head in `tools.arena` unless it says otherwise. Agents that are not in this repository
(v2, v3, v4, v6–v9, the n-series nets, the RL agent) were intermediate steps; their results are
kept here because the conclusions depend on them. The final hybrid agent's ladder is in the
[README](../README.md) and the [report](report.pdf).

Eval seeds: `--seed0 990000` (same as `catan_kit.evaluate`). Training data seeds: 100000–599999.
Server: 2x EPYC 7742 (128 cores / 256 threads). Arena = `tools.arena` (verified
game-identical to `catan_kit.match` on 16/16 games).

## Calibration

| Machine | F mean CPU/game | Ratio to eval box |
|---|---|---|
| EPYC 9R14 (eval, from problem statement) | 158 ms | 1.0 |
| M5 laptop | 78 ms | 0.5 |
| server, 120 workers (1/physical core) | 216 ms | 1.37 |
| server, 240 workers (SMT) | 424 ms | 2.7 |

First decision in a fresh process costs up to ~1 s (catanatron lazy init). Warm up in `__init__`.

## Runs

| # | A | B | games | win rate A | A CPU/game | notes |
|---|---|---|---|---|---|---|
| 1 | F | F | 4000 | 0.499 ±0.016 | 216 ms (120w) | sanity, seat-balanced |
| 2 | SameTurnAlphaBeta d2 pruned, unclocked | F | 960 | 0.379 ±0.031 | 1.2 s (240w) | search on F's value hurts |
| 3 | AlphaBeta d2 pruned, unclocked | F | 960 | 0.372 ±0.031 | 1.6 s | |
| 4 | SameTurnAlphaBeta d2 full, unclocked | F | 960 | 0.501 ±0.032 | 4.8 s | 30x F's CPU for parity |
| 5 | v1: 1-ply + learned V (MLP 64-32, 76 feats, 4.9M pos from F/F-eps games) | F | 2000 | 0.391 ±0.021 | 275 ms | learned V alone is not enough at 1 ply |
| 6 | v2: same-turn beam search (beam 3, depth 3) + same V | F | 2000 | 0.571 ±0.022 | 717 ms | 212 over clock (budget overshoot) |
| 7 | v2 with per-action deadline, cap 60 ms/decision | F | 2000 | **0.616 ±0.021** | 581 ms (F 194) | 33 over (F 2), 100 workers, scale 1.4 |
| 8 | v2r1: v2 search + V retrained on +2.8M positions from v2 self-play / v2-vs-F (expert iteration round 1) | F | 2000 | 0.618 ±0.021 | 628 ms (F 264) | 146 over (concurrent GPU job loading data) |
| 9 | v2r1 | v2 | 2000 | **0.590 ±0.022** | 481 ms (v2 487) | 0 over. Round 1 beats round 0 head-to-head, not vs F |
| 10 | v2r1, strict clock (scale 1.0), 32 quiet workers | F | 640 | **0.650 ±0.037** | 398 ms (F 153 = eval box) | 0 over, max decide 264 ms. Submission candidate |

### Pool (1000 games each, unclocked, eval seeds)

| Agent | F | eps10 | road_rush | dev_rush | city_rush | greedy_vp | worst-half | mean |
|---|---|---|---|---|---|---|---|---|
| v2 | 0.605 | 0.711 | 0.895 | 0.605 | 0.609 | 0.983 | 0.606 | 0.735 |
| v2r1 | 0.670 | 0.750 | 0.938 | 0.628 | 0.649 | 0.988 | **0.649** | 0.770 |

### Value-model size (GPU, 8.3M positions, bs 8192, 30 epochs)

| Hidden | test logloss | acc |
|---|---|---|
| 32 | 0.4546 | 0.768 |
| 64-32 (CPU, bs 4096) | 0.4515 | 0.771 |
| 128-64 | 0.4544 | 0.770 |
| 256-128-64 | 0.4775 | 0.765 (overfits) |

Capacity is not the bottleneck; targets/features/data are.
| 11 | v3: v2r1 + F's robber | F | 2000 | **0.699 ±0.020** | | unclocked; matches the ablation (+5 pts over v2r1) |
| 12 | v4: v3 search + 639-obs net (V0, 4M pos from v3 games, old obs with to-move bug) | F | 400 | 0.018 ±0.013 | | obs field 638 hid whose move it was; fixed later |
| 13 | **v5**: v3 search + 76-feat net retrained with +400k C-engine self-play games (v3 net, 41.5M decisions) | v3 | 2000 | **0.567 ±0.022** | 362 ms | 0 over. More data helps |
| 14 | **v5** | F | 2000 | **0.750 ±0.019** | 405 ms | new best (v3: 0.699) |
| 15 | v6: v3 search + 639-obs net (fixed encoder), trained on the same 400k games as v5 | v3 | 2000 | 0.099 ±0.013 | 369 ms | log-loss 0.434 (better than v5's 0.439) yet loses: raw-obs errors on off-distribution afterstates get picked by the argmax |
| 16 | v7: v5 net retrained with +600k more self-play games (v5 net), ~1M engine games total | v5 | 2000 | 0.521 ±0.022 | 332 ms | not significant: the 64-32 net has plateaued |
| 17 | v7 | F | 2000 | 0.741 ±0.019 | 450 ms | same as v5 |
| 18 | E3: 2.5M more C-engine self-play games (v5 net, C efeats path, 128 workers, 2330 games/s overlapped) | | | | | |
| 19 | n128a: 76-feat 128-64 net on ~1M engine games | v5 | 2000 | 0.513 ±0.022 | 577 ms | vs F 0.759 ±0.019 |
| 20 | n256a: 76-feat 256-256 net, same data | v5 | 2000 | 0.495 ±0.022 | 460 ms | vs F 0.723 ±0.020 (slower leaves, less search) |
| 21 | v5 search grid: beam 5 / depth 3; beam 3 / depth 4; beam 5 / depth 4 / 100 ms cap | v5 | 1000 each | 0.506 / 0.503 / 0.496 | | same-turn search is saturated |
| 22 | v5 with its own value-based robber | v5 | 2000 | 0.490 ±0.022 | | F's robber stays |
| 23 | v9: v5 + exact dice/dev expectations (ActionRecord.result fix) | v5 | 2000 | 0.510 ±0.022 | 333 ms | vs F 0.751 ±0.019 |
| 24 | RL r64-1444 (efeats 76-64-64) in v5 search | F | 2000 | 0.766 ±0.019 | 357 ms | strict clock on a busy server: 0.738 ±0.034, 11/640 over |

| 25 | v5_open: v5 + Codex pair-aware balanced opening | v5 | 2000 | 0.4995 ±0.022 | 294 ms | vs F 0.744 ±0.019 |

Plateau: data, capacity, search width/depth and robber, exact chance expectations and the Codex opening all tie v5 (~75% vs F).

### Decision-type ablation (v2r1 vs F, 2000 games, unclocked, ±2.1)

| Variant | win vs F | reading |
|---|---|---|
| v2r1 | 0.650 | baseline |
| v2r1, F's opening | 0.645 | our opening = F's |
| v2r1, **F's robber** | **0.699** | our robber is worse than F's (-5 pts) |
| v2r1, F's discard | 0.651 | discard irrelevant |
| v2r1, F's pre-roll | 0.637 | ~same |
| v2r1, F's main turn | 0.491 | all our edge is in main-turn play |
| v2r1, random opening | 0.104 | opening is worth ~55 pts |
| v2r1, random robber | 0.362 | robber worth ~29 pts |
| v2r1, random discard | 0.642 | |
| F, our opening | 0.514 | |
| F, our robber | 0.465 | |
| F, our main turn | 0.675 | |

### How v2r1 wins and loses vs F (2000 games)

Wins: we end with 2.3 settlements + 1.8 cities, 1.07 VP cards, 88% longest road, 65% largest army,
5.2 dev cards bought. Losses: 1.8 settlements + 1.1 cities (barely expanded), F has 3.2 + 2.6 and makes
8.4 bank trades. 40% of losses are blowouts (<= 5 VP). 12% of boards lost from both seats.
Settlement available to build: 124 times in 60 games for v2r1 vs 195 for F; chosen 83% vs 86% when
available. Our weakness is reaching buildable spots, not taking them.

Takeaway (2-4): F's hand-tuned lexicographic evaluator is the bottleneck, not search depth.

Value net v0: test logloss 0.4466, accuracy 0.774 (split by game).
Value net r1 (8.3M positions incl. v2 games): test logloss 0.4515, accuracy 0.771 (different test set).

Note: timing noise in big runs is self-inflicted (our own workers sharing SMT siblings and memory
bandwidth). One quiet server core ~= one eval core (F 121 ms/game
on 1 worker vs 158 ms on EPYC 9R14). Clock checks: <= 64 workers. Strength/data runs: up to ~240 workers.

## Feature and opening round (16:00-16:30)

| # | A | B | games | win rate A | notes |
|---|---|---|---|---|---|
| 26 | v11: v5 search + 123-feature net (efeats2: +47 affordability/race/robber/contested/phase/bank features), 12M rows from E1+E2 obs | v5 | 2000 | **0.549 ±0.022** | same rows, 76 feats: logloss 0.4193; 123 feats: 0.4081 |
| 27 | v11 | F | 2000 | **0.7725 ±0.018** | strict clock (480 g, 24 workers): 0.748 ±0.039, 2 over |
| 28 | 123-feat 128-64 net | | | | logloss 0.4094: no gain over 64-32 |
| 29 | v12: v11 + reply-aware opening (greedy setup rollouts, K=5, M=10) | v11 | 2000 | **0.5405 ±0.022** | opening call ~80 ms |
| 30 | v12 | F | 2000 | **0.7975 ±0.018** | |
| 31 | v12r: v12 with the net's own robber (no F rule) | v12 | 2000 | 0.471 ±0.022 | F's robber stays; robber data is all from F's rule (on-policy only) |
| 32 | v12k: opening K=8, M=14 | v12 | 2000 | 0.498 ±0.022 | wider opening search saturated |
| 33 | v12, strict clock (scale 1.0, 24 workers) | F | 480 | **0.815 ±0.035** | 1 over, 523 ms/game. Submission candidate |
| 34 | 123-feat 128-64 net (v11b); 123-feat 64-32 on 24M rows (v11c) | v11 | 2000 each | 0.4935 / 0.501 | capacity and data volume are not the lever |
| 35 | v12g: v12 + exact chance outcomes (ActionRecord.result) + gc.freeze, strict clock | F | 480 | 0.825 ±0.034 | = v12 (0.815); 2 over, max decide 300 ms |
| 36 | v13: v12 with a two-net logit ensemble (second 64-32 net, lr 1e-3, 40 ep, logloss 0.4082) | v12 | 2000 | 0.526 ±0.022 | +2.5% CPU; small, borderline gain |
