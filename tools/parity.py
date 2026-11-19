"""Check that tools.arena plays exactly the games catan_kit.match plays."""
import sys
from catan_kit._patch import ensure_hash_seed

if __name__ == "__main__":
    ensure_hash_seed()
    from catan_kit import match
    from tools import arena
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seeds = [990000 + i // 2 for i in range(n)]
    ref, _ = match.play_chunk(a, b, seeds, list(range(n)))
    _, got = arena.run(a, b, n, workers=4, quiet=True)
    same = sum((r["winner"], r["turns"]) == (g["winner"], g["turns"]) for r, g in zip(ref, got))
    print(f"identical games: {same}/{n}")
    for r, g in zip(ref, got):
        if (r["winner"], r["turns"]) != (g["winner"], g["turns"]):
            print("diff", r["game"], r["winner"], r["turns"], g["winner"], g["turns"])
