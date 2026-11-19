"""Build training sets for the extended features from engine self-play shards that carry the raw
639-float observation (tools.engine_selfplay without SELFPLAY_NO_OBS).

    python -m tools.build_feats2 --data data/sp/E1 data/sp/E2 --rows 12000000 --out data/f2

Writes <out>_123.npz (X = efeats2, 123 features) and <out>_76.npz (X = the stored 76 features of the
same rows) with y and g, in tools.train_value's format, so both nets train on identical rows.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents", "common"))
from efeats2 import efeats2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=12_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = []
    for d in a.data:
        files += sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    X2, X76, Y, G = [], [], [], []
    n, t0, off = 0, time.time(), 0
    for i, f in enumerate(files):
        z = np.load(f)
        if "O" not in z:
            continue
        O = z["O"].astype(np.float32)
        X2.append(efeats2(O))
        X76.append(z["X"].astype(np.float32))
        Y.append(z["y"].astype(np.float32))
        G.append(z["g"].astype(np.int64) + off)
        n += len(O)
        if i % 10 == 0:
            print(f"{i}/{len(files)} shards, {n} rows, {time.time()-t0:.0f}s", flush=True)
        if n >= a.rows:
            break
    off = 0
    Gs = []
    for g in G:                       # game ids are unique per worker already; keep them distinct per shard dir
        Gs.append(g)
    X2, X76, Y, G = np.concatenate(X2), np.concatenate(X76), np.concatenate(Y), np.concatenate(Gs)
    np.savez(a.out + "_123.npz", X=X2, y=Y, g=G)
    np.savez(a.out + "_76.npz", X=X76, y=Y, g=G)
    print(f"saved {len(Y)} rows: {a.out}_123.npz, {a.out}_76.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
