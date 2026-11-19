"""Python bindings (ctypes) for the C Catan engine. See engine/README.md.

    from engine import Engine, VecEnv, N_ACTIONS, OBS_DIM   (or engine.cengine)

Action indices are catanatron's ACTIONS_ARRAY indices (332 actions, colours RED/BLUE, BASE map).
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBNAME = "libcatan.dylib" if sys.platform == "darwin" else "libcatan.so"
_LIBPATH = os.path.join(_DIR, _LIBNAME)

N_ACTIONS = 332
OBS_DIM = 639
MAX_LEGAL = 128
SER_MAX = 512
TURNS_LIMIT = 1000

F_AUTOPLAY, F_OBS, F_AUTORESET = 1, 2, 4

# prompts
P_INIT_SETTLE, P_INIT_ROAD, P_PLAY, P_DISCARD, P_ROBBER = range(5)


def build(force=False):
    srcs = [os.path.join(_DIR, f) for f in ("catan.c", "vec.c", "search.c", "feats.c", "catan.h", "topology.h")]
    if force or not os.path.exists(_LIBPATH) or \
            os.path.getmtime(_LIBPATH) < max(os.path.getmtime(s) for s in srcs):
        subprocess.check_call(["make", "-s", _LIBNAME], cwd=_DIR)


build()
lib = ctypes.CDLL(_LIBPATH)
_vp = ctypes.c_void_p
_i = ctypes.c_int
_u64 = ctypes.c_uint64
lib.cat_init_tables.restype = None
lib.cat_state_size.restype = _i
lib.cat_new_game.argtypes = [_vp, _u64]
lib.cat_load.argtypes = [_vp, _vp, _vp, _vp, _vp, _vp, _i, _i, _u64]
lib.cat_legal.argtypes = [_vp, _vp]
lib.cat_legal.restype = _i
lib.cat_step.argtypes = [_vp, _i, _i]
lib.cat_done.argtypes = [_vp]
lib.cat_done.restype = _i
lib.cat_autoplay.argtypes = [_vp, _vp, _vp]
lib.cat_autoplay.restype = _i
lib.cat_encode.argtypes = [_vp, _vp]
lib.cat_encode_seat.argtypes = [_vp, _i, _vp]
lib.cat_serialize.argtypes = [_vp, _vp]
lib.cat_serialize.restype = _i
lib.cat_play_random.argtypes = [_vp, _i]
lib.cat_play_random.restype = _i
lib.vec_reset.argtypes = [_vp, _i, _u64, _vp, _vp, _vp, _i]
lib.vec_step.argtypes = [_vp, _i, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _i]
lib.vec_step.restype = ctypes.c_long
lib.vec_random_games.argtypes = [_vp, _i, ctypes.c_long, _u64, _i]
lib.vec_random_games.restype = ctypes.c_long
lib.vec_encode.argtypes = [_vp, _i, _vp]
lib.vec_expand.argtypes = [_vp, _i, _vp, _vp, _vp, _vp, _vp, ctypes.c_long]
lib.vec_expand.restype = ctypes.c_long
lib.vec_encode_seat.argtypes = [_vp, _i, _vp, _vp]
lib.vec_info.argtypes = [_vp, _i, _vp, _vp, _vp, _vp]
lib.vec_redeal.argtypes = [_vp, _i, _vp]
lib.cat_init_tables()
STATE_SIZE = lib.cat_state_size()


def _ptr(a):
    return a.ctypes.data if a is not None else None


class Engine:
    """One game state. Copying is a memcpy of STATE_SIZE bytes (`copy()`)."""

    def __init__(self, seed=None, buf=None):
        self.buf = np.zeros(STATE_SIZE, np.uint8) if buf is None else buf
        self._legal = np.zeros(MAX_LEGAL, np.int16)
        if seed is not None:
            self.new_game(seed)

    @property
    def ptr(self):
        return self.buf.ctypes.data

    def new_game(self, seed):
        lib.cat_new_game(self.ptr, seed & 0xFFFFFFFFFFFFFFFF)
        return self

    def load(self, tile_res, tile_num, port_res, colors, deck, robber, seed=0):
        tr = np.asarray(tile_res, np.int8)
        tn = np.asarray(tile_num, np.int8)
        pr = np.asarray(port_res, np.int8)
        co = np.asarray(colors, np.int8)
        dk = np.asarray(list(deck) + [0], np.int8)
        lib.cat_load(self.ptr, _ptr(tr), _ptr(tn), _ptr(pr), _ptr(co), _ptr(dk), len(deck),
                     robber, seed)
        return self

    def legal(self):
        n = lib.cat_legal(self.ptr, self._legal.ctypes.data)
        return self._legal[:n].copy()

    def step(self, action, outcome=-1):
        lib.cat_step(self.ptr, int(action), int(outcome))

    def autoplay(self):
        n = ctypes.c_int()
        return lib.cat_autoplay(self.ptr, self._legal.ctypes.data, ctypes.byref(n))

    def done(self):
        return bool(lib.cat_done(self.ptr))

    def encode(self, out=None, seat=None):
        """Observation of the seat to move (or of `seat`)."""
        out = np.zeros(OBS_DIM, np.float32) if out is None else out
        if seat is None:
            lib.cat_encode(self.ptr, out.ctypes.data)
        else:
            lib.cat_encode_seat(self.ptr, int(seat), out.ctypes.data)
        return out

    def serialize(self):
        out = np.zeros(SER_MAX, np.int32)
        n = lib.cat_serialize(self.ptr, out.ctypes.data)
        return out[:n]

    def copy(self):
        return Engine(buf=self.buf.copy())

    def play_random(self):
        return lib.cat_play_random(self.ptr, 0)

    # a few fields, read through the canonical serialisation
    def info(self):
        s = self.serialize()
        return dict(num_turns=int(s[-14]), cur=int(s[-13]), turn=int(s[-12]), prompt=int(s[-11]),
                    winner=int(s[-1]))


class VecEnv:
    """N auto-resetting games stepped in one C call. Buffers are numpy arrays filled in place:

        obs (N, OBS_DIM) float32    observation of the seat to move (zeros if obs=False)
        mask (N, N_ACTIONS) bool    legal actions of the seat to move
        reward (N,) float32         +1/-1 for the seat that chose the action, when it ended the game
        terminal, truncated (N,)    game ended with a winner / at the 1000-turn cap
        cur (N,) int8               seat to move (0/1)
        winner (N,) int8            seat that won on this step, else -1

    Forced moves (exactly one legal action) are played inside the engine when autoplay=True, as the
    harness does. With auto-reset, an env that finished starts a new game immediately and the
    buffers already describe the new game's first decision.
    """

    def __init__(self, n, seed=0, autoplay=True, obs=True, autoreset=True):
        self.n = n
        self.flags = (F_AUTOPLAY if autoplay else 0) | (F_OBS if obs else 0) | \
            (F_AUTORESET if autoreset else 0)
        self.states = np.zeros((n, STATE_SIZE), np.uint8)
        self.obs = np.zeros((n, OBS_DIM), np.float32)
        self._mask = np.zeros((n, N_ACTIONS), np.uint8)
        self.mask = self._mask.view(np.bool_)
        self.reward = np.zeros(n, np.float32)
        self.terminal = np.zeros(n, np.uint8)
        self.truncated = np.zeros(n, np.uint8)
        self.cur = np.zeros(n, np.int8)
        self.winner = np.full(n, -1, np.int8)
        self.reset(seed)

    def reset(self, seed=0):
        lib.vec_reset(self.states.ctypes.data, self.n, seed, self.obs.ctypes.data,
                      self._mask.ctypes.data, self.cur.ctypes.data, self.flags)
        return self.obs, self.mask

    def step(self, actions):
        a = np.ascontiguousarray(actions, dtype=np.int32)
        return lib.vec_step(self.states.ctypes.data, self.n, a.ctypes.data, self.obs.ctypes.data,
                            self._mask.ctypes.data, self.reward.ctypes.data,
                            self.terminal.ctypes.data, self.truncated.ctypes.data,
                            self.cur.ctypes.data, self.winner.ctypes.data, self.flags)

    def engine(self, i):
        """An Engine view (shared memory) on env i."""
        return Engine(buf=self.states[i])


def encode_states(states):
    """Encode an (M, STATE_SIZE) uint8 array of states (e.g. search leaves) -> (M, OBS_DIM)."""
    states = np.ascontiguousarray(states, np.uint8)
    out = np.zeros((len(states), OBS_DIM), np.float32)
    lib.vec_encode(states.ctypes.data, len(states), out.ctypes.data)
    return out


class Expander:
    """Batched move expansion for search (engine/search.c). Buffers are reused across calls."""

    def __init__(self, max_out=1 << 16):
        self.max_out = max_out
        self.out = np.zeros((max_out, STATE_SIZE), np.uint8)
        self.prob = np.zeros(max_out, np.float32)
        self.act = np.zeros(max_out, np.int16)
        self.parent = np.zeros(max_out, np.int32)
        self.obs = np.zeros((max_out, OBS_DIM), np.float32)

    def expand(self, states):
        """states (n, STATE_SIZE) uint8 -> m, nact. Children are self.out[:m] etc."""
        states = np.ascontiguousarray(states, np.uint8)
        nact = np.zeros(len(states), np.int16)
        m = lib.vec_expand(states.ctypes.data, len(states), self.out.ctypes.data,
                           self.prob.ctypes.data, self.act.ctypes.data, self.parent.ctypes.data,
                           nact.ctypes.data, self.max_out)
        return int(m), nact

    def encode(self, m, seats):
        """Encode children [0, m) from seats (m,) int8 views into self.obs[:m]."""
        seats = np.ascontiguousarray(seats, np.int8)
        lib.vec_encode_seat(self.out.ctypes.data, m, seats.ctypes.data, self.obs.ctypes.data)
        return self.obs[:m]


def state_info(states):
    """(cur, prompt, done, winner) int8 arrays for an (n, STATE_SIZE) array."""
    states = np.ascontiguousarray(states, np.uint8)
    n = len(states)
    cur, prompt, done, winner = (np.zeros(n, np.int8) for _ in range(4))
    lib.vec_info(states.ctypes.data, n, cur.ctypes.data, prompt.ctypes.data, done.ctypes.data,
                 winner.ctypes.data)
    return cur, prompt, done, winner


def redeal(states, me):
    """In place: re-deal the opponent's unplayed dev cards and the deck, from seat `me`'s view."""
    me = np.ascontiguousarray(me, np.int8)
    lib.vec_redeal(states.ctypes.data, len(states), me.ctypes.data)
