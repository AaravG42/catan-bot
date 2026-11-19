"""Fast parallel arena: plays two agent files against each other in-process, many workers at once.

Same games as catan_kit.evaluate (same board seeds, seats, dice key scheme, masked views, agent RNG
reseeding), but without one OS process per agent, so it scales to hundreds of cores.

    python -m tools.arena --a agents/v1/agent.py --b examples/f_agent.py --games 2000 --workers 8

The engine and each agent get their own `random` state: before an agent decides, the global
generator is swapped to that agent's stream and swapped back afterwards, exactly as if the agent
lived in its own process. CPU is measured with process_time around decide (Obs construction
included, masking excluded, as in the official harness). --cpu-scale S enforces the official clock
multiplied by S (use ~2.7 on the EPYC 7742 server to emulate the EPYC 9R14 eval machine).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy loads: one BLAS thread per worker process

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
import time

from catan_kit._patch import ensure_hash_seed

_W = {}  # per-worker state


def load_agent_class(path, tag):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f"arena_agent_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


def _make_seat_class():
    from catanatron.models.player import Player
    from catan_kit import match
    from catan_kit.obs import Obs, masked_copy
    from catanatron.models.player import RandomPlayer

    class Seat(Player):
        def __init__(self, color, agent, name, cpu_scale):
            super().__init__(color)
            self.agent, self.name, self.cpu_scale = agent, name, cpu_scale
            self.agent_rng_state = None

        def begin_game(self, seed):
            self.fallback = random.Random(seed)
            self.mask_rng = random.Random(seed ^ 0x5EED)
            ag_seed = int.from_bytes(hashlib.sha256(f"agent:{seed}".encode()).digest()[:8], "big")
            saved = random.getstate()
            random.seed(ag_seed)
            self.agent_rng_state = random.getstate()
            random.setstate(saved)
            self.cpu = 0.0
            self.calls = self.errors = self.random_moves = 0
            self.over = False
            self.max_decide = 0.0
            if hasattr(self.agent, "new_game"):
                self.agent.new_game()

        def allowance(self):
            return min(match.CPU_BASE_S + (self.calls + 1) * match.CPU_PER_DECISION_S,
                       match.CPU_MAX_S) * self.cpu_scale

        def decide(self, game, playable_actions):
            actions = list(playable_actions)
            if len(actions) == 1:
                return actions[0]
            if self.over:
                self.random_moves += 1
                return self.fallback.choice(actions)
            allow = self.allowance()
            if self.cpu_scale and self.cpu > allow:
                self.over = True
                self.random_moves += 1
                return self.fallback.choice(actions)
            view = masked_copy(game, self.color, self.mask_rng)
            view.state.players = [RandomPlayer(c) for c in view.state.colors]
            view.seed = 0
            self.calls += 1
            engine_state = random.getstate()
            random.setstate(self.agent_rng_state)
            t0 = time.process_time()
            try:
                obs = Obs(view, self.color)
                # in this machine's seconds: the scaled allowance buys the same work as on eval
                obs.cpu_left = (allow - self.cpu) if self.cpu_scale else None
                idx = self.agent.decide(obs, actions)
                ok = type(idx) is int or hasattr(idx, "__index__")
                idx = int(idx) if ok else -1
            except Exception:
                import traceback
                if _W.get("verbose"):
                    traceback.print_exc()
                idx = -1
            dt = time.process_time() - t0
            self.agent_rng_state = random.getstate()
            random.setstate(engine_state)
            self.cpu += dt
            self.max_decide = max(self.max_decide, dt)
            if self.cpu_scale and self.cpu > allow:
                self.over = True
            if not 0 <= idx < len(actions):
                self.errors += 1
                self.random_moves += 1
                return self.fallback.choice(actions)
            return actions[idx]

        def record(self):
            return dict(cpu_s=self.cpu, decisions=self.calls, over=self.over,
                        random_moves=self.random_moves, errors=self.errors,
                        max_decide_s=self.max_decide)

    return Seat


def _init_worker(path_a, path_b, cpu_scale_a, cpu_scale_b, vps, key, verbose, record=0.0,
                 record_obs=0.0):
    import catan_kit  # noqa: F401
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from catanatron.models.player import Color
    Seat = _make_seat_class()
    random.seed(20261013)
    A = load_agent_class(path_a, "a")()
    B = load_agent_class(path_b, "b")()
    _W.update(a=Seat(Color.RED, A, "a", cpu_scale_a), b=Seat(Color.BLUE, B, "b", cpu_scale_b),
              vps=vps, key=key, verbose=verbose, record=record, record_obs=record_obs,
              rec_rng=random.Random(os.getpid()))


def _play(job):
    board_seed, gi = job
    from catanatron.game import Game
    from catan_kit import match
    sa, sb, vps, key = _W["a"], _W["b"], _W["vps"], _W["key"]
    swap = gi % 2 == 1
    sa.color = match.COLORS[1] if swap else match.COLORS[0]
    sb.color = match.COLORS[0] if swap else match.COLORS[1]
    order = [sb, sa] if swap else [sa, sb]
    random.seed(board_seed)
    game = Game(order, seed=board_seed, vps_to_win=vps)
    random.seed(match.dice_seed(board_seed, gi // 2, key))
    s = (board_seed * 1_000_003 + gi) & 0xFFFFFFFF
    sa.begin_game(s)
    sb.begin_game(s ^ 0xABCDEF)
    rows = None
    if _W.get("record_obs"):
        import numpy as np
        from engine.obs_np import encode_game
        obs_rows, meta = [], []
        p, rr = _W["record_obs"], _W["rec_rng"]

        class RecObs:
            def before(self, g): pass
            def after(self, g): pass
            def step(self, g, action):
                if rr.random() < p:
                    t = len(g.state.action_records)
                    for k, c in enumerate(g.state.colors):
                        obs_rows.append(encode_game(g, c))
                        meta.append((t, k))
        winner = game.play(accumulators=[RecObs()])
    elif _W.get("record"):
        rows = []
        p, rr = _W["record"], _W["rec_rng"]
        from agents.common.feats import feats

        class Rec:
            def before(self, g): pass
            def after(self, g): pass
            def step(self, g, action):
                if rr.random() < p:
                    t = len(g.state.action_records)
                    for c in g.state.colors:
                        rows.append((c, feats(g, c), t))
        winner = game.play(accumulators=[Rec()])
    else:
        winner = game.play()
    w = None if winner is None else ("a" if winner == sa.color else "b")
    out = dict(game=gi, board_seed=board_seed, a_first=not swap, winner=w,
               turns=game.state.num_turns, a=sa.record(), b=sb.record())
    out["opening"] = _opening(game, sa.color)
    out["a"].update(_end_stats(game, sa.color))
    out["b"].update(_end_stats(game, sb.color))
    if _W.get("record_obs") and obs_rows:
        import numpy as np
        win_k = -1 if winner is None else game.state.colors.index(winner)
        m = np.asarray(meta, np.int32)
        out["obs"] = (np.asarray(obs_rows).astype(np.int16), m[:, 0], m[:, 1].astype(np.int8),
                      (m[:, 1] == win_k).astype(np.int8))
    if rows is not None:
        out["rows"] = [(f, 1.0 if winner == c else 0.0, t, k) for c, f, t in rows
                       for k in [game.state.colors.index(c)]]
    return out


class _Sharder:
    """Collects (obs int16, ply, seat, won) per game and writes data_out/shard_KKKK.npz files."""

    def __init__(self, out_dir, rows):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.rows, self.buf, self.n, self.k, self.total = out_dir, rows, [], 0, 0, 0

    def add(self, obs, game):
        self.buf.append(obs + (game,))
        self.n += len(obs[0])
        if self.n >= self.rows:
            self.flush()

    def flush(self):
        import numpy as np
        if not self.buf:
            return
        X = np.concatenate([b[0] for b in self.buf])
        np.savez(os.path.join(self.dir, f"shard_{self.k:04d}.npz"), X=X,
                 t=np.concatenate([b[1] for b in self.buf]),
                 k=np.concatenate([b[2] for b in self.buf]),
                 y=np.concatenate([b[3] for b in self.buf]),
                 g=np.concatenate([np.full(len(b[1]), b[4], np.int32) for b in self.buf]))
        self.total += len(X)
        self.k += 1
        self.buf, self.n = [], 0


def _opening(game, a_color):
    """Initial placements in order: [is_a, settlement_node, road_edge] x 4."""
    from catanatron.models.enums import ActionType as AT
    out, pend = [], None
    for r in game.state.action_records[:16]:
        t = r.action.action_type
        if t == AT.BUILD_SETTLEMENT:
            pend = [r.action.color == a_color, r.action.value]
        elif t == AT.BUILD_ROAD and pend is not None:
            out.append(pend + [list(r.action.value)])
            pend = None
        if len(out) == 4:
            break
    return out


def _end_stats(game, color):
    """How this player stood at the end: points and where they came from."""
    from catanatron.models.enums import SETTLEMENT, CITY
    st = game.state
    k = f"P{st.color_to_index[color]}_"
    ps = st.player_state
    b = st.buildings_by_color[color]
    return dict(vp=ps[k + "ACTUAL_VICTORY_POINTS"], pub_vp=ps[k + "VICTORY_POINTS"],
                setts=len(b[SETTLEMENT]), cities=len(b[CITY]), vp_cards=ps[k + "VICTORY_POINT_IN_HAND"],
                road=bool(ps[k + "HAS_ROAD"]), army=bool(ps[k + "HAS_ARMY"]),
                knights=ps[k + "PLAYED_KNIGHT"], road_len=ps[k + "LONGEST_ROAD_LENGTH"],
                devs_bought=sum(1 for r in st.action_records if r.action.color == color
                                and r.action.action_type.name == "BUY_DEVELOPMENT_CARD"),
                trades=sum(1 for r in st.action_records if r.action.color == color
                           and r.action.action_type.name == "MARITIME_TRADE"))


def run(path_a, path_b, games, seed0=990000, workers=None, cpu_scale_a=0.0, cpu_scale_b=0.0,
        vps=10, key=None, out=None, verbose=False, quiet=False, record=0.0, data_out=None,
        record_obs=0.0, shard_rows=500_000):
    from catan_kit import match
    key = key or match.DEFAULT_KEY
    workers = workers or os.cpu_count()
    jobs = [(seed0 + i // 2, i) for i in range(games)]
    t0 = time.time()
    ctx = mp.get_context("spawn")  # fork charges copy-on-write faults to agents
    with ctx.Pool(min(workers, games), initializer=_init_worker,
                  initargs=(path_a, path_b, cpu_scale_a, cpu_scale_b, vps, key, verbose, record,
                            record_obs)) as pool:
        recs, step, wins = [], max(1, games // 20), 0
        shard = _Sharder(data_out, shard_rows) if record_obs else None
        for r in pool.imap_unordered(_play, jobs, chunksize=1):
            if shard is not None and "obs" in r:
                shard.add(r.pop("obs"), r["game"])
            recs.append(r)
            wins += r["winner"] == "a"
            if not quiet and len(recs) % step == 0:
                print(f"  [{len(recs)}/{games}] A win {wins/len(recs):.3f}  {time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
    if record_obs:
        shard.flush()
        if not quiet:
            print(f"saved {shard.total} positions in {shard.k} shards under {data_out}")
    recs.sort(key=lambda r: r["game"])
    if record:
        import numpy as np
        X, y, g, t, k = [], [], [], [], []
        for r in recs:
            for f, lab, ply, seat in r.pop("rows"):
                X.append(f); y.append(lab); g.append(r["game"]); t.append(ply); k.append(seat)
        np.savez_compressed(data_out, X=np.asarray(X, np.float32), y=np.asarray(y, np.float32),
                            g=np.asarray(g, np.int32), t=np.asarray(t, np.int32),
                            k=np.asarray(k, np.int8))
        if not quiet:
            print(f"saved {len(y)} positions to {data_out}")
    res = summarise(recs)
    res["wall_s"] = time.time() - t0
    if out:
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    if not quiet:
        print(f"A = {path_a}\nB = {path_b}")
        print(f"A win rate   {res['win_rate']:.4f} +-{res['ci95']:.4f} ({res['wins']}/{res['games']}, "
              f"no winner {res['no_winner']})")
        for s in ("a", "b"):
            print(f"{s.upper()} cpu/game mean {res[s+'_mean_cpu']*1000:7.1f} ms  worst "
                  f"{res[s+'_worst_cpu']*1000:7.1f} ms  max decide {res[s+'_max_decide']*1000:6.1f} ms  "
                  f"over {res[s+'_over']}  errors {res[s+'_errors']}  decisions/game "
                  f"{res[s+'_decisions']/res['games']:.1f}")
        print(f"wall {res['wall_s']:.1f} s")
    return res, recs


def summarise(recs):
    n = len(recs)
    wins = sum(r["winner"] == "a" for r in recs)
    rate = wins / n
    res = dict(games=n, wins=wins, win_rate=rate, ci95=1.96 * (rate * (1 - rate) / n) ** 0.5,
               no_winner=sum(r["winner"] is None for r in recs))
    for s in ("a", "b"):
        cpu = [r[s]["cpu_s"] for r in recs]
        res[s + "_mean_cpu"] = sum(cpu) / n
        res[s + "_worst_cpu"] = max(cpu)
        res[s + "_max_decide"] = max(r[s]["max_decide_s"] for r in recs)
        res[s + "_over"] = sum(r[s]["over"] for r in recs)
        res[s + "_errors"] = sum(r[s]["errors"] for r in recs)
        res[s + "_decisions"] = sum(r[s]["decisions"] for r in recs)
    return res


def main():
    ensure_hash_seed()
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=990000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--scale-a", type=float, default=0.0,
                   help="enforce the clock on A scaled by this factor (0 = no clock)")
    p.add_argument("--scale-b", type=float, default=0.0)
    p.add_argument("--out", default=None, help="write per-game records as jsonl")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--record", type=float, default=0.0, help="record each ply with this probability")
    p.add_argument("--data-out", default=None)
    p.add_argument("--record-obs", type=float, default=0.0,
                   help="record the 639-dim engine observation of both seats with this probability")
    a = p.parse_args()
    run(a.a, a.b, a.games, a.seed0, a.workers, a.scale_a, a.scale_b, out=a.out, verbose=a.verbose,
        record=a.record, data_out=a.data_out, record_obs=a.record_obs)


if __name__ == "__main__":
    main()
