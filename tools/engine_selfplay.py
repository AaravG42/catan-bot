"""Self-play data at engine speed: a batched search player over an ObsNet value, on the C engine.

    python -m tools.engine_selfplay --net agents/v5/value.npz --games 1000000 --workers 64 \
        --seed 1500000 --out data/sp/g1

Each worker steps N games at once (VecEnv). At every decision, per game:
  1. re-deal what the mover cannot see (opponent dev cards, deck order), as the harness does
  2. expand all legal actions with exact chance outcomes (engine/search.c)
  3. score children with the value net from the mover's view (terminal: 1 win / 0 loss)
  4. same-turn beam: the top BEAM deterministic actions that keep the turn are expanded again
     (DEPTH levels) and take their best continuation's value
  5. pick argmax, with exploration: opening settlements/roads sampled from the top K, and an
     epsilon-random move otherwise
The mover's observation is recorded with probability --record and labelled with the game's winner.
Shards (int16 X, int8 y, int32 g game id, int16 t decision number, int8 seat) go to --out.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy loads: one BLAS thread per worker process

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents", "common"))


class TorchNet:
    """The same MLP on a GPU (shared by many worker processes)."""

    def __init__(self, path, device):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        d = np.load(path)
        n = int(d["n"])
        mu, sd = d["mu"].astype(np.float32), d["sd"].astype(np.float32)
        W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        b[0] = b[0] - (mu / sd) @ W[0]
        W[0] = W[0] / sd[:, None]
        self.W = [torch.tensor(w, device=device) for w in W]
        self.b = [torch.tensor(x, device=device) for x in b]
        self.torch, self.device = torch, device

    def value(self, X):
        t = self.torch
        with t.no_grad():
            h = t.from_numpy(np.ascontiguousarray(X, np.float32)).to(self.device, non_blocking=True)
            for i, (W, b) in enumerate(zip(self.W, self.b)):
                h = h @ W + b
                if i < len(self.W) - 1:
                    h = t.relu(h)
            return t.sigmoid(h[:, 0]).cpu().numpy()


NO_OBS = os.environ.get("SELFPLAY_NO_OBS") == "1"   # skip the raw 639-obs copy in shards


def _worker(args):
    (wid, net_path, n_env, games, seed, out_dir, record, beam, depth, eps, open_k, shard_rows,
     temp, device, py_feats) = args
    os.environ["OMP_NUM_THREADS"] = "1"
    from engine.cengine import VecEnv, Expander, redeal, state_info, P_INIT_SETTLE, P_INIT_ROAD, lib

    def lib_encode_seat(states, seats, out):
        states = np.ascontiguousarray(states)
        seats = np.ascontiguousarray(seats, np.int8)
        lib.vec_encode_seat(states.ctypes.data, len(states), seats.ctypes.data, out.ctypes.data)
    from obsnet import ObsNet
    from efeats import efeats
    from core import Value
    f76 = net_path is not None and "W0" in np.load(net_path) and np.load(net_path)["W0"].shape[0] == 76
    if f76 and not py_feats:
        from engine.cfeats import Value76, efeats_obs as efeats
        net = Value76(net_path)      # features + MLP in C, fused with the encoder
    elif f76:
        v76 = Value(net_path)

        class _N:
            def value(self, X, mt):
                return v76(efeats(X, mt))
        net = _N()
    else:
        net = ObsNet(net_path) if net_path else None
        if net is not None and device != "cpu":
            net = TorchNet(net_path, device)
    rng = np.random.default_rng(seed)
    env = VecEnv(n_env, seed=seed)
    exps = [Expander(1 << 15) for _ in range(depth + 1)]   # one buffer set per search level
    NA = 332
    game_id = np.arange(n_env, dtype=np.int64) + wid * 10_000_000
    next_game = n_env
    dec = np.zeros(n_env, np.int32)
    rows_X, rows_g, rows_t, rows_k = [], [], [], []
    pending = {}            # game id -> list of (row index, seat)
    labels = {}             # game id -> winner seat (-1 none)
    done_games = 0
    shard_k = 0
    buf_X, buf_y, buf_g, buf_t, buf_k, buf_O = [], [], [], [], [], []
    stats = dict(decisions=0, children=0, t_search=0.0)

    def values(exp, m, seats, info):
        v = np.empty(m, np.float32)
        cur, _, dn, w = info
        live = dn == 0
        v[~live] = (w[~live] == seats[~live]).astype(np.float32)
        if live.any():
            idx = np.flatnonzero(live)
            if f76 and not py_feats:
                v[idx] = net.states(exp.out, idx, seats[idx])
                return v
            X = exp.encode(m, seats)[idx]
            if net is None:
                v[idx] = rng.random(len(idx))
            elif f76:
                v[idx] = net.value(X, (cur[idx] == seats[idx]).astype(np.float32))
            else:
                v[idx] = net.value(X)
        return v

    def qvals(exp, states, seats, lvl):
        """q (n, 332) for the given parent states from seats' views."""
        n = len(states)
        m, nact = exp.expand(states)
        q = np.full((n, NA), -np.inf, np.float32)
        if m == 0:
            return q
        par = exp.parent[:m].copy()
        act = exp.act[:m].astype(np.int64)
        prob = exp.prob[:m].copy()
        cs = seats[par]
        info = state_info(exp.out[:m])
        v = values(exp, m, cs, info)
        flat = par.astype(np.int64) * NA + act
        acc = np.zeros(n * NA, np.float32)
        np.add.at(acc, flat, prob * v)
        seen = np.zeros(n * NA, bool)
        seen[flat] = True
        qf = q.reshape(-1)
        qf[seen] = acc[seen]
        stats["children"] += m
        if lvl < depth and beam > 0:
            # deterministic children (one outcome) that keep the turn with the same seat
            cnt = np.bincount(flat, minlength=n * NA)
            det = (cnt[flat] == 1) & (info[0] == cs) & (info[2] == 0) & (act != END_IDX)
            cand = np.flatnonzero(det)
            if len(cand):
                # top `beam` per parent by q
                order = cand[np.lexsort((-qf[flat[cand]], par[cand]))]
                p_o = par[order]
                first = np.r_[True, p_o[1:] != p_o[:-1]]
                rank = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
                pick = order[rank < beam]
                if len(pick):
                    sub_states = exp.out[pick].copy()
                    sub_q = qvals(exps[lvl + 1], sub_states, cs[pick], lvl + 1)
                    best = sub_q.max(1)
                    ok = np.isfinite(best)
                    tgt = flat[pick][ok]
                    qf[tgt] = np.maximum(qf[tgt], best[ok])
        return q

    from engine.bridge import ACTIONS_ARRAY
    global END_IDX
    END_IDX = next(i for i, a in enumerate(ACTIONS_ARRAY) if a[0].name == "END_TURN")

    def flush(final=False):
        nonlocal shard_k, buf_X, buf_y, buf_g, buf_t, buf_k, buf_O
        if not buf_X or (not final and sum(len(x) for x in buf_X) < shard_rows):
            return
        np.savez(os.path.join(out_dir, f"shard_w{wid:03d}_{shard_k:04d}.npz"),
                 X=np.concatenate(buf_X), y=np.concatenate(buf_y), g=np.concatenate(buf_g),
                 t=np.concatenate(buf_t), k=np.concatenate(buf_k),
                 **({} if NO_OBS else dict(O=np.concatenate(buf_O))))
        shard_k += 1
        buf_X, buf_y, buf_g, buf_t, buf_k, buf_O = [], [], [], [], [], []

    store_X, store_O, store_meta = [], [], []
    t0 = time.time()
    wins_seat0 = 0
    while done_games < games:
        cur = env.cur.copy()
        states = env.states.copy()
        redeal(states, cur)
        ts = time.time()
        q = qvals(exps[0], states, cur, 0)
        stats["t_search"] += time.time() - ts
        q[~env.mask] = -np.inf
        a = q.argmax(1)
        # exploration
        _, prompt, _, _ = state_info(env.states)
        opening = (prompt == P_INIT_SETTLE) | (prompt == P_INIT_ROAD)
        if open_k > 1 and opening.any():
            for i in np.flatnonzero(opening):
                legal = np.flatnonzero(env.mask[i])
                k = min(open_k if prompt[i] == P_INIT_SETTLE else 2, len(legal))
                top = legal[np.argsort(-q[i, legal])[:k]]
                a[i] = rng.choice(top)
        if temp > 0:
            noisy = (~opening) & (rng.random(n_env) < 1.0)
            if noisy.any():
                z = q[noisy] / temp
                z -= z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True)
                a[noisy] = (p.cumsum(1) > rng.random((noisy.sum(), 1))).argmax(1)
        if eps > 0:
            r = (~opening) & (rng.random(n_env) < eps)
            for i in np.flatnonzero(r):
                a[i] = rng.choice(np.flatnonzero(env.mask[i]))
        # record before stepping
        rec = np.flatnonzero(rng.random(n_env) < record)
        if len(rec):
            both = np.repeat(rec, 2)
            seats = np.tile(np.array([0, 1], np.int8), len(rec))
            ob = np.zeros((len(both), 639), np.float32)
            lib_encode_seat(env.states[both], seats, ob)
            fx = efeats(ob, (cur[both] == seats).astype(np.float32))
            ob16 = ob.astype(np.int16)
            for j, i in enumerate(both):
                pending.setdefault(int(game_id[i]), []).append(len(store_X))
                store_X.append(fx[j])
                if not NO_OBS:
                    store_O.append(ob16[j])
                store_meta.append((int(game_id[i]), int(dec[i]), int(seats[j])))
        env.step(a.astype(np.int32))
        stats["decisions"] += n_env
        dec += 1
        fin = np.flatnonzero(env.terminal | env.truncated)
        for i in fin:
            gid = int(game_id[i])
            w = int(env.winner[i]) if env.terminal[i] else -1
            wins_seat0 += w == 0
            rows = pending.pop(gid, [])
            if rows:
                Xg = np.stack([store_X[r] for r in rows]).astype(np.float32)
                if not NO_OBS:
                    buf_O.append(np.stack([store_O[r] for r in rows]))
                meta = np.array([store_meta[r] for r in rows])
                buf_X.append(Xg)
                buf_y.append((meta[:, 2] == w).astype(np.int8))
                buf_g.append(meta[:, 0].astype(np.int32))
                buf_t.append(meta[:, 1].astype(np.int16))
                buf_k.append(meta[:, 2].astype(np.int8))
            game_id[i] = wid * 10_000_000 + next_game
            next_game += 1
            dec[i] = 0
            done_games += 1
            if wid == 0 and done_games % 200 == 0:
                el = time.time() - t0
                print(f"  [w0] {done_games} games {el:.0f}s  {stats['decisions']/el:.0f} dec/s  "
                      f"{stats['children']/max(1, stats['decisions']):.0f} children/dec", flush=True)
        if not pending and len(store_X) > 0:
            store_X, store_O, store_meta = [], [], []
        elif len(store_X) > 400_000:
            # compact: keep only rows of unfinished games
            keep = sorted(r for rs in pending.values() for r in rs)
            remap = {r: j for j, r in enumerate(keep)}
            store_X = [store_X[r] for r in keep]
            store_O = [] if NO_OBS else [store_O[r] for r in keep]
            store_meta = [store_meta[r] for r in keep]
            pending = {g: [remap[r] for r in rs] for g, rs in pending.items()}
        flush()
    flush(final=True)
    el = time.time() - t0
    return dict(wid=wid, games=done_games, seconds=el, decisions=stats["decisions"],
                children=stats["children"], t_search=stats["t_search"], wins_seat0=wins_seat0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="ObsNet npz (none = random values)")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", type=int, default=64, help="games stepped together per worker")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", type=float, default=0.25)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--temp", type=float, default=0.0, help="softmax temperature over q (0 = argmax)")
    ap.add_argument("--open-k", type=int, default=6)
    ap.add_argument("--shard-rows", type=int, default=100_000)
    ap.add_argument("--device", default="cpu", help="cuda: run the value net on the GPU")
    ap.add_argument("--py-feats", action="store_true",
                    help="76-feature nets: numpy efeats + Value (old path) instead of engine/feats.c")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = (a.games + a.workers - 1) // a.workers
    jobs = [(w, a.net, a.envs, per, a.seed * 1000 + w, a.out, a.record, a.beam, a.depth, a.eps,
             a.open_k, a.shard_rows, a.temp, a.device, a.py_feats) for w in range(a.workers)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        res = []
        for r in pool.imap_unordered(_worker, jobs):
            res.append(r)
            g = sum(x["games"] for x in res)
            print(f"  worker {r['wid']} done: {r['games']} games in {r['seconds']:.0f}s "
                  f"({r['decisions']/r['seconds']:.0f} dec/s, {r['children']/max(1,r['decisions']):.0f} children/dec); "
                  f"total {g} games, {time.time()-t0:.0f}s", flush=True)
    g = sum(x["games"] for x in res)
    print(f"done: {g} games, {sum(x['decisions'] for x in res)} decisions, "
          f"seat-0 win {sum(x['wins_seat0'] for x in res)/g:.3f}, wall {time.time()-t0:.0f}s, "
          f"{g/(time.time()-t0):.0f} games/s", flush=True)


if __name__ == "__main__":
    main()
