"""Numpy MLP over the engine's 639-float observation (engine/obs_np.py layout).

File format (npz): mu (639), sd (639), n (layers), W0..W{n-1} (in, out), b0..b{n-1}; ReLU between
layers; the last layer's first output is the value logit. Optional `policy` > 0: outputs 1..332 are
policy logits. Used both by engine self-play and inside submissions (no torch at play time).
"""
import numpy as np


class ObsNet:
    def __init__(self, path, dtype=np.float32):
        d = np.load(path)
        n = int(d["n"])
        sd = d["sd"].astype(dtype)
        mu = d["mu"].astype(dtype)
        W = [d[f"W{i}"].astype(dtype) for i in range(n)]
        b = [d[f"b{i}"].astype(dtype) for i in range(n)]
        # fold the input normalisation into the first layer: ((x - mu) / sd) @ W0 = x @ W0' + b0'
        W0 = W[0] / sd[:, None]
        b[0] = b[0] - (mu / sd) @ W[0]
        W[0] = W0
        self.W, self.b, self.dtype = W, b, dtype
        self.has_policy = int(d["policy"]) if "policy" in d else 0

    def forward(self, X):
        h = np.asarray(X, dtype=self.dtype)
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < last:
                np.maximum(h, 0, out=h)
        return h

    def value(self, X):
        """P(win) for the seat whose view X is."""
        z = self.forward(X)[:, 0]
        return 1.0 / (1.0 + np.exp(-z))
