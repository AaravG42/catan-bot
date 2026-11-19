"""Summarise how games were won and lost, from an arena --out jsonl.

    python -m tools.analyze logs/abl_v2r1_base.jsonl
"""
import json
import sys
from statistics import mean


def pct(x):
    return f"{100*x:5.1f}%"


def block(title, recs, side):
    o = "b" if side == "a" else "a"
    if not recs:
        return
    print(f"\n{title}  (n={len(recs)})")
    print(f"  {'':22s} {'A':>8s} {'B':>8s}")
    for k, f in [("final VP", lambda r, s: r[s]["vp"]), ("settlements", lambda r, s: r[s]["setts"]),
                 ("cities", lambda r, s: r[s]["cities"]), ("VP cards", lambda r, s: r[s]["vp_cards"]),
                 ("has longest road", lambda r, s: r[s]["road"]), ("has largest army", lambda r, s: r[s]["army"]),
                 ("knights played", lambda r, s: r[s]["knights"]), ("dev cards bought", lambda r, s: r[s]["devs_bought"]),
                 ("bank trades", lambda r, s: r[s]["trades"]), ("road length", lambda r, s: r[s]["road_len"])]:
        print(f"  {k:22s} {mean(f(r, 'a') for r in recs):8.2f} {mean(f(r, 'b') for r in recs):8.2f}")
    print(f"  turns {mean(r['turns'] for r in recs):.1f}")


def main(path):
    recs = [json.loads(l) for l in open(path)]
    n = len(recs)
    wins = [r for r in recs if r["winner"] == "a"]
    losses = [r for r in recs if r["winner"] == "b"]
    print(f"{path}: A wins {len(wins)}/{n} = {pct(len(wins)/n)}")
    first = [r for r in recs if r["a_first"]]
    second = [r for r in recs if not r["a_first"]]
    print(f"  A first:  {pct(sum(r['winner']=='a' for r in first)/len(first))}   "
          f"A second: {pct(sum(r['winner']=='a' for r in second)/len(second))}")
    # paired boards: both seats on the same board and dice
    by = {}
    for r in recs:
        by.setdefault(r["board_seed"], []).append(r["winner"] == "a")
    pairs = [v for v in by.values() if len(v) == 2]
    print(f"  boards won both seats {pct(sum(all(v) for v in pairs)/len(pairs))}, "
          f"split {pct(sum(any(v) and not all(v) for v in pairs)/len(pairs))}, "
          f"lost both {pct(sum(not any(v) for v in pairs)/len(pairs))}")
    for lo, hi in [(0, 60), (60, 80), (80, 100), (100, 130), (130, 10**9)]:
        g = [r for r in recs if lo <= r["turns"] < hi]
        if g:
            print(f"  turns {lo:3d}-{hi if hi < 10**9 else 'inf':>3}: {len(g):5d} games  A win {pct(sum(r['winner']=='a' for r in g)/len(g))}")
    block("A WINS", wins, "a")
    block("A LOSSES", losses, "a")
    close = [r for r in losses if r["a"]["vp"] >= 8]
    print(f"\n  losses where A had >= 8 VP: {pct(len(close)/max(1,len(losses)))}   "
          f"<= 5 VP: {pct(sum(r['a']['vp'] <= 5 for r in losses)/max(1,len(losses)))}")


if __name__ == "__main__":
    main(sys.argv[1])
