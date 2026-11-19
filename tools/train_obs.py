"""Train a value MLP on 639-dim observation shards (tools.arena --record-obs / tools.engine_selfplay).

    python -m tools.train_obs --data data/boot/* --out agents/obsnet/value.npz --hidden 256 256

Shards: X int16 (n, 639), y int8 (won), g int32 (game id). Splits by game id per shard directory
(10% test). Normalises with a sample mean/std. Writes agents/common/obsnet.py format.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch


CONCAT = False


def load(dirs, max_rows=None):
    Xs, ys, test = [], [], []
    for di, d in enumerate(dirs):
        for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            z = np.load(f)
            if CONCAT:
                Xs.append(np.concatenate([z["X"].astype(np.float32), z["O"].astype(np.float32)], 1))
            else:
                Xs.append(z["O"] if "O" in z else z["X"])
            ys.append(z["y"].astype(np.float32))
            g = z["g"].astype(np.int64) + di * 10_000_019
            test.append((g * 2654435761 % 2**32) % 10 == 0)
    X, y, test = np.concatenate(Xs), np.concatenate(ys), np.concatenate(test)
    if max_rows and len(X) > max_rows:
        idx = np.random.default_rng(0).choice(len(X), max_rows, replace=False)
        X, y, test = X[idx], y[idx], test[idx]
    return X, y, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="shard directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--init", default=None, help="warm-start from an obsnet npz with the same shape")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--concat", action="store_true", help="input = 76 engineered features ++ 639 obs")
    a = ap.parse_args()
    global CONCAT
    CONCAT = a.concat
    torch.manual_seed(0)
    t0 = time.time()
    X, y, test = load(a.data, a.max_rows)
    perm = np.random.default_rng(2).permutation(len(X))
    X, y, test = X[perm], y[perm], test[perm]
    print(f"loaded {len(X)} rows ({test.mean():.3f} test) in {time.time()-t0:.0f}s", flush=True)
    samp = X[np.random.default_rng(1).choice(len(X), min(len(X), 500_000), replace=False)].astype(np.float32)
    mu, sd = samp.mean(0), samp.std(0) + 1e-3
    dev = a.device
    Xtr = torch.from_numpy(X[~test]); ytr = torch.from_numpy(y[~test])
    Xte = torch.from_numpy(X[test]).to(dev); yte = torch.from_numpy(y[test]).to(dev)
    if dev == "cuda" and Xtr.numel() * Xtr.element_size() < 18e9:
        Xtr, ytr = Xtr.to(dev), ytr.to(dev)       # int16 on the GPU: 2 bytes per feature
    elif dev == "cuda":
        Xtr, ytr = Xtr.pin_memory(), ytr.pin_memory()   # too big: stream batches from host
    mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
    layers, prev = [], X.shape[1]
    for h in a.hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()] + ([torch.nn.Dropout(a.dropout)] if a.dropout else [])
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    net = torch.nn.Sequential(*layers).to(dev)
    if a.init:
        z = np.load(a.init)
        lin = [m for m in net if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for i, m in enumerate(lin):
                m.weight.copy_(torch.tensor(z[f"W{i}"].T)); m.bias.copy_(torch.tensor(z[f"b{i}"]))
        mu, sd = z["mu"], z["sd"]
        mu_t, sd_t = torch.tensor(mu, device=dev, dtype=torch.float32), torch.tensor(sd, device=dev, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(ytr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()

    def prep(xb):
        return (xb.to(dev, non_blocking=True).float() - mu_t) / sd_t

    best, best_state = 1e9, None
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(ytr), device=Xtr.device)
        if Xtr.device.type == "cpu":
            # contiguous random blocks keep host->GPU copies fast
            nb = (len(ytr) + a.bs - 1) // a.bs
            perm = torch.randperm(nb)
        for k in range(0, len(ytr) if Xtr.device.type != "cpu" else len(perm) * a.bs, a.bs):
            if Xtr.device.type == "cpu":
                j = int(perm[k // a.bs]) * a.bs
                idx = slice(j, j + a.bs)
            else:
                idx = perm[k:k + a.bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(prep(Xtr[idx])).squeeze(1), ytr[idx].to(dev))
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            tot, acc = 0.0, 0.0
            for k in range(0, len(yte), 65536):
                lv = net(prep(Xte[k:k + 65536])).squeeze(1)
                tot += lossf(lv, yte[k:k + 65536]).item() * len(lv)
                acc += ((lv > 0).float() == yte[k:k + 65536]).sum().item()
        print(f"epoch {ep+1:3d}  test logloss {tot/len(yte):.4f}  acc {acc/len(yte):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if tot / len(yte) < best:
            best = tot / len(yte)
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"best test logloss {best:.4f}", flush=True)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    out = dict(mu=np.asarray(mu, np.float64), sd=np.asarray(sd, np.float64), n=len(lin), policy=0)
    for i, m in enumerate(lin):
        out[f"W{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        out[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
