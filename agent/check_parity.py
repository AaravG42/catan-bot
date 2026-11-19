"""Decision-for-decision parity between the shipped final submission and this cleaned-up package.

Both agents are loaded into ONE process (PYTHONHASHSEED must be 0, as in the harness) and asked
to decide on the very same observation at every decision with more than one legal action, in
self-play games driven exactly the way catan_kit.match builds observations (masked copy, no
opponent object, no board seed). The original's choice is played; any disagreement is reported
with the game, turn and action list and makes the exit status non-zero.

The clock: the beam search is bounded by CPU time as well as by node count, so with a real
clock two identical implementations can legitimately disagree whenever the 6 ms cap cuts the
search at a different node. `--clock frozen` (default) makes time.process_time() return a
constant for both agents, so every deadline is far away and the search is bounded by nodes
only; decisions then depend on logic alone. `--clock ticking` advances the fake clock by a fixed
amount per call instead, which exercises the time cut-offs deterministically and checks that
both implementations consult the clock at the same points. `--clock real` is informational.

--cpu-left sets obs.cpu_left for every decision (a generous 1.0 by default); `gates` cycles
through 0.02 / 0.05 / 0.2 to exercise the < 30 ms emergency path and short search budgets.

The original is read from submission/both_roads_v12_cachefix.zip next to this script and its
SHA-256 checked, unless --original-zip names another copy.

    PYTHONHASHSEED=0 .venv/bin/python agent/check_parity.py --games 60
    PYTHONHASHSEED=0 .venv/bin/python agent/check_parity.py --games 30 \
        --seed0 100 --cpu-left gates
    PYTHONHASHSEED=0 .venv/bin/python agent/check_parity.py --games 30 \
        --seed0 200 --clock ticking
"""
import argparse
import hashlib
import importlib.util
import itertools
import os
import random
import sys
import tempfile
import time
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"    # as both agents do, before numpy is first imported
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SHIPPED_ZIP = os.path.join(HERE, "submission", "both_roads_v12_cachefix.zip")
ZIP_SHA256 = "46d0c26632b9f5640f427aad215e8f1e0d7354eafddc27c766917a799adf46e7"

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("run with PYTHONHASHSEED=0 (the harness does)")

# Python puts this script's directory first on sys.path; the original must not find our
# `opening`/`outcomes` there, so drop it until the original has loaded.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, REPO)
import catan_kit  # noqa: F401,E402  deterministic trade-order patch, as in the harness
from catan_kit.obs import Obs, masked_copy  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Color, Player, RandomPlayer  # noqa: E402


def original_zip_bytes(zip_path):
    if zip_path:
        with open(zip_path, "rb") as f:
            data = f.read()
    else:
        with open(SHIPPED_ZIP, "rb") as f:
            data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ZIP_SHA256:
        sys.exit(f"original zip sha256 {digest} != expected {ZIP_SHA256}")
    return data


def load_original(zip_path):
    """Unzip the shipped submission into a fresh directory and import its Agent the way the
    harness does (its own directory first on sys.path). All its modules live under the `codex`
    package, so nothing collides with our flat module names; its sys.path additions are
    removed afterwards so that our imports cannot resolve into its tree."""
    root = tempfile.mkdtemp(prefix="final_orig_")
    zpath = os.path.join(root, "original.zip")
    with open(zpath, "wb") as f:
        f.write(original_zip_bytes(zip_path))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    path = os.path.join(root, "agent.py")
    before = set(sys.modules)
    sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("original_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clash = {"opening", "outcomes", "search", "evaluator", "features123", "agent"}
    assert not clash & (set(sys.modules) - before), clash & set(sys.modules)
    sys.path = [p for p in sys.path if not p.startswith(root)]
    return module.Agent


def load_clean():
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location("clean_agent", os.path.join(HERE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


class FakeClock:
    def __init__(self, step):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class PairSeat(Player):
    """Plays the original's choice, after checking the clean agent agrees."""

    def __init__(self, color, original, clean, cpu_left, stats):
        super().__init__(color)
        self.original, self.clean, self.cpu_left, self.stats = original, clean, cpu_left, stats

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]
        view = masked_copy(game, self.color, self.stats["mask_rng"])
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        view.seed = 0
        obs = Obs(view, self.color)
        obs.cpu_left = self.cpu_left(self.stats["n"])
        # Both agents must see the same global random state (the engine draws from it).
        state = random.getstate()
        a = self.original.decide(obs, actions)
        random.setstate(state)
        b = self.clean.decide(obs, actions)
        self.stats["n"] += 1
        phase = "setup" if game.state.is_initial_build_phase else "main"
        self.stats["by_phase"][phase] = self.stats["by_phase"].get(phase, 0) + 1
        if a != b:
            self.stats["mismatches"].append(dict(
                game=self.stats["game"], phase=phase, turn=game.state.num_turns,
                color=self.color.value,
                cpu_left=obs.cpu_left, original=a, clean=b,
                actions=[f"{x.action_type.name}:{x.value}" for x in actions]))
            print(f"MISMATCH game={self.stats['game']} turn={game.state.num_turns} "
                  f"{self.color.value} cpu_left={obs.cpu_left} original={a} clean={b} "
                  f"({actions[a].action_type.name}:{actions[a].value} vs "
                  f"{actions[b].action_type.name}:{actions[b].value})", flush=True)
        return actions[a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-zip", default=None, help="local copy; default: read from git")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--cpu-left", default="1.0", help="a number, or 'gates' for 0.02/0.05/0.2")
    ap.add_argument("--clock", choices=["frozen", "ticking", "real"], default="frozen")
    ap.add_argument("--tick", type=float, default=50e-6, help="seconds per call for --clock ticking")
    args = ap.parse_args()

    Original = load_original(args.original_zip)
    Clean = load_clean()
    t0 = time.monotonic()
    original, clean = Original(), Clean()
    print(f"loaded both agents in {time.monotonic()-t0:.1f}s", flush=True)

    if args.cpu_left == "gates":
        cycle = [0.02, 0.05, 0.2]
        cpu_left = lambda n: cycle[n % 3]
    else:
        value = float(args.cpu_left)
        cpu_left = lambda n: value

    real_clock = time.process_time
    if args.clock == "frozen":
        time.process_time = lambda: 0.0
    elif args.clock == "ticking":
        time.process_time = FakeClock(args.tick)

    stats = dict(n=0, mismatches=[], by_phase={}, game=None, mask_rng=None)
    wall0 = time.monotonic()
    for k in range(args.games):
        seed = args.seed0 + k
        stats["game"] = seed
        stats["mask_rng"] = random.Random(seed ^ 0x5EED)
        seats = [PairSeat(Color.RED, original, clean, cpu_left, stats),
                 PairSeat(Color.BLUE, original, clean, cpu_left, stats)]
        random.seed(seed)
        game = Game(seats, seed=seed)
        winner = game.play()
        print(f"game {seed}: winner={getattr(winner, 'value', None)} turns={game.state.num_turns} "
              f"decisions so far={stats['n']} mismatches={len(stats['mismatches'])} "
              f"[{time.monotonic()-wall0:.0f}s]", flush=True)
    time.process_time = real_clock

    print(f"\nclock={args.clock} cpu_left={args.cpu_left} games={args.games} seeds "
          f"{args.seed0}..{args.seed0+args.games-1}")
    print(f"decisions compared: {stats['n']} {stats['by_phase']}")
    by_phase = {p: sum(m["phase"] == p for m in stats["mismatches"]) for p in ("setup", "main")}
    print(f"mismatches: {len(stats['mismatches'])} {by_phase}")
    for m in stats["mismatches"]:
        print("  ", m)
    sys.exit(1 if stats["mismatches"] else 0)


if __name__ == "__main__":
    main()
