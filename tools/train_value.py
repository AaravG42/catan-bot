"""Fit a win-probability model V(position, me) on recorded positions.

    python -m tools.train_value --data data/ff.npz data/fe10.npz --out agents/v1/value.npz --hidden 64 32

Splits by game id (no position of a test game is trained on), standardises features, trains a small
MLP with BCE (hidden=() gives logistic regression), and writes plain numpy weights so the agent needs
no torch at play time.
"""
import argparse
import time

import numpy as np
import torch


def load(paths):
    Xs, ys, gs = [], [], []
    off = 0
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"]); ys.append(d["y"]); gs.append(d["g"].astype(np.int64) + off)
        off += int(d["g"].max()) + 1
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def build(nin, hidden):
    layers, prev = [], nin
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 32])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    X, y, g = load(a.data)
    test = (g * 2654435761 % 2**32) % 10 == 0
    mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
    Xn = (X - mu) / sd
    dev = a.device
    Xt, yt = torch.tensor(Xn[~test], device=dev), torch.tensor(y[~test], device=dev)
    Xv, yv = torch.tensor(Xn[test], device=dev), torch.tensor(y[test], device=dev)
    print(f"train {len(yt)}  test {len(yv)}  features {X.shape[1]}  device {dev}")
    net = build(X.shape[1], a.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(yt), device=dev)
        for k in range(0, len(yt), a.bs):
            idx = perm[k:k + a.bs]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            lv = net(Xv).squeeze(1)
            vl = lossf(lv, yv).item()
            acc = ((lv > 0).float() == yv).float().mean().item()
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep+1:3d}  test logloss {vl:.4f}  acc {acc:.4f}  ({time.time()-t0:.0f}s)")
    Ws = {}
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        Ws[f"W{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float64)
        Ws[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float64)
    np.savez(a.out, mu=mu.astype(np.float64), sd=sd.astype(np.float64), n=len(lin), **Ws)
    print("saved", a.out)


if __name__ == "__main__":
    main()
