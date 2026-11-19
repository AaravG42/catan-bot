# Catan Bot (Team Blanc)

A two-player Catan agent built at Hyde's Catan hackathon at IIT Bombay
(October 2026), where it placed **2nd of 61 submissions**. It wins **83.5% of games against F**,
catanatron's strongest built-in player, under the official clock (300 ms per game plus 12.5 ms per
decision), with **no games over the time limit**. On the event leaderboard it won 85.7% against F
and 70.1% against O, Hyde's own trained bot.

It runs on one CPU core with numpy, without a GPU or a large model: a hand-written evaluation, a
small search over the current turn, and a tiny learned win-probability network.

## How it plays

At every decision the agent:

1. **Scores every legal move exactly.** For each move it enumerates the chance outcomes (dice
   rolls, development-card draws, robber steals) with their probabilities and takes the expected
   value. No sampling.
2. **Searches its own turn.** The three best moves that keep the turn are expanded again, three
   levels deep, within 80 nodes and 6 ms.
3. **Evaluates positions** as a hand-written score (production, expansion, army, hand) plus eight
   times the log-odds of a learned win probability. That probability comes from two 64-32 MLPs
   over 76 board features, averaged in logit space. A denial term penalises positions where the
   opponent is about to score.
4. **Plans the setup separately.** For each initial settlement and road, the top five candidates
   are each played out to the end of the setup phase, with both players choosing greedily by a
   123-feature setup network. The candidate whose finished setup is rated best is played.
5. **Watches the clock.** Running out of time means random moves for the rest of the game, so
   search gets a few percent of the remaining allowance. Below 30 ms, a fixed fallback takes over
   (winning build, else city > settlement > end turn).

Each part's contribution, against F with 95% intervals:

| Agent | Win rate vs F |
|---|---|
| Hand-written evaluation, one move ahead | 71.8% ±2.0 |
| + tactical terms | 76.1% ±1.9 |
| + turn search, tuned weights | 77.7% ±1.8 |
| + learned correction | 81.0% ±2.4 |
| + setup rollouts (event leaderboard, 1,000 games) | 85.7% ±2.2 |
| **Final, official harness, 600 games** | **83.5% ±3.0** |

The details, with every constant and where it came from, are in [`agent/README.md`](agent/README.md)
and the [report](docs/report.pdf).

## A Catan engine in C

To generate training data fast, we rewrote catanatron's rules in C ([`engine/`](engine)). The C
engine makes the same move as catanatron on every one of 100,000 test games (85.6 million moves),
and steps games about 200 times faster per move: 25,900 random games per second on one laptop
core. See [`engine/README.md`](engine/README.md) for the design, the parity tests and benchmarks.

## The learned-value line

Alongside the final agent we pushed a pure "search plus learned value" line as far as it would
go. These agents are in [`agents/`](agents):

| Agent | What it adds | Win rate vs F |
|---|---|---|
| [`v1`](agents/v1) | 76-feature 64-32 MLP, one move ahead, trained on 4.9M positions | 39.1% ±2.1 |
| [`v5`](agents/v5) | Same-turn beam search, F's robber, net retrained on 400k C-engine games | 75.0% ±1.9 |
| [`v11`](agents/v11) | 123 features from the full observation, net trained on 12M engine positions | 77.3% ±1.8 |
| [`v12`](agents/v12) | v11 with reply-aware setup rollouts | 79.8% ±1.8 |

What we learned along the way, in [`docs/experiments.md`](docs/experiments.md):

- **More data and bigger nets stopped helping early.** 29M positions did no better than 12M,
  and 128-64 or 256-256 nets did no better than 64-32.
- **Lower log-loss did not mean stronger play.** A net with better validation loss could lose
  more games, especially a net on the raw 639-float observation, which memorised boards.
- **CPU spikes, not average cost, lose games on the clock.** One submission went over the limit
  in 21 of 1,000 games while averaging only 382 ms. Later, a feature cache that grew by one board
  per decision made Python's garbage-collection pauses grow from 32 ms to 365 ms over a game.
  Capping it took the final agent from 17 games over the limit in 600 to none.

## Quick start

You need Python 3.11 and git. catanatron is pinned to an exact commit; don't upgrade it, because
the action space changed between releases.

```bash
git clone https://github.com/AaravG42/catan-bot.git
cd catan-bot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # on Linux, first: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
make -C engine                    # builds engine/libcatan.so (or .dylib)
```

Play the final agent against F under the official clock:

```bash
PYTHONHASHSEED=0 python -m catan_kit.evaluate --a agent/agent.py --b examples/f_agent.py --games 100 --seed0 990000
```

Run a faster, unclocked head-to-head on several cores:

```bash
PYTHONHASHSEED=0 python -m tools.arena --a agents/v12/agent.py --b examples/f_agent.py --games 200 --workers 8
```

Check that the readable agent makes the same decisions as the shipped submission:

```bash
PYTHONHASHSEED=0 python agent/check_parity.py --games 60
```

## Training your own value net

```bash
# 1. self-play on the C engine
PYTHONHASHSEED=0 python -m tools.engine_selfplay --net agents/v5/value.npz --games 100000 --workers 8 --seed 1 --out data/sp/run1
# 2. build the 123 features
PYTHONHASHSEED=0 python -m tools.build_feats2 --data data/sp/run1 --rows 2000000 --out data/f2
# 3. train the 64-32 MLP
PYTHONHASHSEED=0 python -m tools.train_value --data data/f2_123.npz --out agents/mine/value.npz --hidden 64 32
```

Each script's docstring lists its options.

## Repository layout

    agent/        the final agent: readable, parity-checked copy of the shipped submission
      submission/ the shipped zip, byte for byte
    agents/       learned-value milestones v1, v5, v11, v12, and shared features (common/)
    engine/       C rules engine, Python bindings, feature encoders
    tools/        arena, self-play, feature building, training, parity and benchmarks
    catan_kit/    Hyde's match harness and official evaluator
    examples/     F (catanatron's value-function player) and a random agent
    docs/         report and experiment notes

## Credits

Team Blanc. Built on [catanatron](https://github.com/bcollazo/catanatron) by Bryan Collazo and on
Hyde's [hackathon kit](https://github.com/pbhar-hyde/catan-agent-hackathon). Released under
GPL-3.0-or-later, as both are; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
