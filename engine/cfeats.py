"""v3's 76 features and the small value MLP in C (engine/feats.c). Same numbers as
agents/common/efeats.py + core.Value, ~20-50x faster, and no large numpy temporaries (so it scales
across many worker processes instead of saturating memory bandwidth).

    from engine.cfeats import efeats_obs, efeats_states, Value76
    F = efeats_obs(X, my_turn)            # X (n, 639) float32, my_turn (n,) -> (n, 76)
    F = efeats_states(states, seats)      # raw engine states, seat view; my_turn = cur == seat
    v = Value76("agents/v5/value.npz")
    p = v.states(states, idx, seats)      # fused encode -> features -> MLP for states[idx]
    p = v(F)                              # MLP on precomputed features

    python -m engine.cfeats               # parity test vs efeats.py / core.Value
"""
import ctypes
import os
import sys

import numpy as np

from .cengine import lib, OBS_DIM, STATE_SIZE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "agents", "common") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "agents", "common"))

NF = 76
_vp, _i = ctypes.c_void_p, ctypes.c_int
lib.feat_init.argtypes = [_vp, _vp, _vp, _i]
lib.vec_efeats.argtypes = [_vp, _i, _vp, _vp]
lib.vec_efeats_st.argtypes = [_vp, _i, _vp, _vp]
lib.mlp_set.argtypes = [_i, _vp, _vp, _vp]
lib.mlp_set.restype = _i
lib.vec_value76.argtypes = [_vp, _i, _vp, _vp, _vp]
lib.vec_mlp.argtypes = [_vp, _i, _vp]

_ready = False


def _init():
    global _ready
    if _ready:
        return
    import efeats as E    # catanatron's BASE topology, exactly what efeats.py uses
    nt = np.full((E.NN, 3), -1, np.int8)
    for n in range(E.NN):
        t = np.flatnonzero(E.A[n])
        nt[n, :len(t)] = t
    eu = np.ascontiguousarray(E.EU, np.int16)
    ev = np.ascontiguousarray(E.EV, np.int16)
    lib.feat_init(nt.ctypes.data, eu.ctypes.data, ev.ctypes.data, len(eu))
    _ready = True


def efeats_obs(X, my_turn):
    _init()
    X = np.ascontiguousarray(X, np.float32)
    mt = np.ascontiguousarray(my_turn, np.float32)
    out = np.empty((len(X), NF), np.float32)
    lib.vec_efeats(X.ctypes.data, len(X), mt.ctypes.data, out.ctypes.data)
    return out


def efeats_states(states, seats):
    _init()
    states = np.ascontiguousarray(states, np.uint8)
    seats = np.ascontiguousarray(seats, np.int8)
    out = np.empty((len(states), NF), np.float32)
    lib.vec_efeats_st(states.ctypes.data, len(states), seats.ctypes.data, out.ctypes.data)
    return out


class Value76:
    """A 76-input ReLU MLP value net (train_value.py npz) evaluated in C. One per process: the C
    side holds a single set of weights, so constructing another replaces it."""

    def __init__(self, path):
        _init()
        d = np.load(path)
        n = int(d["n"])
        mu, sd = d["mu"].astype(np.float64), d["sd"].astype(np.float64)
        W = [d[f"W{i}"].astype(np.float64) for i in range(n)]
        b = [d[f"b{i}"].astype(np.float64) for i in range(n)]
        b[0] = b[0] - (mu / sd) @ W[0]
        W[0] = W[0] / sd[:, None]
        dims = np.array([W[0].shape[0]] + [w.shape[1] for w in W], np.int32)
        Wf = np.concatenate([w.astype(np.float32).ravel() for w in W])
        bf = np.concatenate([x.astype(np.float32).ravel() for x in b])
        r = lib.mlp_set(n, dims.ctypes.data, Wf.ctypes.data, bf.ctypes.data)
        if r != 0:
            raise ValueError(f"mlp_set failed ({r}) for dims {dims.tolist()}")

    def __call__(self, F):
        F = np.ascontiguousarray(F, np.float32)
        out = np.empty(len(F), np.float32)
        lib.vec_mlp(F.ctypes.data, len(F), out.ctypes.data)
        return out

    def states(self, states, idx, seats):
        assert states.dtype == np.uint8 and states.flags.c_contiguous and states.shape[1] == STATE_SIZE
        idx = np.ascontiguousarray(idx, np.int32)
        seats = np.ascontiguousarray(seats, np.int8)
        out = np.empty(len(idx), np.float32)
        lib.vec_value76(states.ctypes.data, len(idx), idx.ctypes.data, seats.ctypes.data,
                        out.ctypes.data)
        return out


def _parity(n_states=200_000, net="agents/v5/value.npz"):
    import time
    from efeats import efeats
    from core import Value
    from .cengine import VecEnv, Expander, state_info
    rng = np.random.default_rng(0)
    env = VecEnv(256, seed=7)
    exp = Expander(1 << 16)
    got = []
    while sum(len(s) for s in got) < n_states:      # search children of random-play states
        m, _ = exp.expand(env.states)
        k = rng.choice(m, min(m, 2000), replace=False)
        got.append(exp.out[k].copy())
        a = np.array([rng.choice(np.flatnonzero(r)) for r in env.mask], np.int32)
        env.step(a)
    S = np.concatenate(got)[:n_states]
    seats = rng.integers(0, 2, len(S)).astype(np.int8)
    cur = state_info(S)[0]
    X = np.zeros((len(S), OBS_DIM), np.float32)
    lib.vec_encode_seat(S.ctypes.data, len(S), seats.ctypes.data, X.ctypes.data)
    mt = (cur == seats).astype(np.float32)
    t = time.time(); Fp = efeats(X, mt); tp = time.time() - t
    t = time.time(); Fc = efeats_obs(X, mt); tc = time.time() - t
    t = time.time(); Fs = efeats_states(S, seats); ts = time.time() - t
    d1, d2 = np.abs(Fp - Fc).max(), np.abs(Fp - Fs).max()
    print(f"{len(S)} states   efeats max|diff| obs {d1:.2e}  states {d2:.2e}   "
          f"numpy {tp/len(S)*1e6:.2f} us/row, C obs {tc/len(S)*1e6:.2f}, C states {ts/len(S)*1e6:.2f}")
    vp = Value(os.path.join(_ROOT, net))
    vc = Value76(os.path.join(_ROOT, net))
    t = time.time(); Pp = vp(Fp); tp = time.time() - t
    idx = np.arange(len(S), dtype=np.int32)
    t = time.time(); Pc = vc.states(S, idx, seats); tc = time.time() - t
    dv = np.abs(Pp - Pc).max()
    print(f"value max|diff| {dv:.2e}   numpy feats+mlp {(tp/len(S)*1e6) + 0:.2f}+ us/row, "
          f"C fused {tc/len(S)*1e6:.2f} us/row")
    assert d1 < 1e-4 and d2 < 1e-4 and dv < 1e-4, "parity failed"
    print("parity OK")


if __name__ == "__main__":
    _parity()
