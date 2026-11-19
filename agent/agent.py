"""Catan hackathon entry: reply-aware setup rollout + handcrafted/learned leaf value + bounded search.

Decision flow (Agent.decide):
  1. setup settlement or road, >1 legal action -> opening.SetupPlanner.choose (always, whatever
                                                  the clock: it costs a fixed few ms)
  2. one legal action                          -> play it
  3. less than EMERGENCY_S (30 ms) CPU left    -> search.emergency: a winning build if any, else
                                                  a cheap priority order
  4. otherwise                                 -> search.search: exact one-ply expectation +
                                                  tactical bonuses + same-turn beam search

Nothing is tuned at run time; every constant is frozen and documented where it lives.
"""
import os
# Before numpy is imported: BLAS/OpenMP must never start worker threads (the harness charges
# the CPU of every thread, and small matrix products gain nothing from them).
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                     # the harness adds it too; be self-sufficient
    sys.path.insert(0, str(HERE))

from evaluator import Evaluator, Value  # noqa: E402
import opening  # noqa: E402
import search  # noqa: E402


class Agent:
    def __init__(self):
        # Main game: heuristic + 8 * logit of the mean-logit ensemble of the two 76-feature nets.
        self.evaluator = Evaluator(str(HERE / 'value_v2r1.npz'), str(HERE / 'value_v5.npz'))
        # Setup: the 123-feature net, alone.
        self.setup = opening.SetupPlanner(Value(str(HERE / 'value_123.npz')))
        # Private generator for obs.sample_world: never the engine's global `random`. Seeded once
        # per process, so its stream continues across games, as in the shipped agent.
        self.rng = random.Random(0)

    def decide(self, obs, actions):
        game = obs._game
        if (len(actions) > 1 and game.state.is_initial_build_phase
                and actions[0].action_type in opening.SETUP_ACTIONS):
            return self.setup.choose(game, actions, obs.me)
        started = time.process_time()
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        # Whole-decision deadline, keeping RESERVE_S of the allowance for atomic engine and
        # evaluator operations that cannot be pre-empted safely in Python.
        hard_deadline = started + min(search.HARD_CAP_S, max(0., left-search.RESERVE_S))
        fallback = lambda: search.emergency(obs, actions, self.rng, min(hard_deadline, started+.003))
        if len(actions) == 1:
            return 0
        if left < search.EMERGENCY_S:
            return fallback()
        return search.search(obs, actions, self.evaluator, self.rng, started, left,
                             hard_deadline, fallback)
