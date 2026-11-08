"""Fit pokerlib's preflop strength model: linear regression of Monte Carlo
equity (vs a random hand) on preflop_features. Prints the weights to paste
into pokerlib.PREFLOP_W."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pokerlib as pl  # noqa: E402

HANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 600

rng = np.random.default_rng(7)
t0 = time.time()
holes = np.array([np.sort(rng.permutation(36)[:5]) for _ in range(HANDS)])
eq = np.empty(HANDS)
for i, h in enumerate(holes):
    eq[i] = pl.simulate(rng, tuple(int(c) for c in h), (), TRIALS)["result"].mean()
print(f"simulated {HANDS} hands in {time.time() - t0:.1f}s")

X = pl.preflop_features(holes)
w, *_ = np.linalg.lstsq(X, eq, rcond=None)
pred = X @ w
r2 = 1 - ((eq - pred) ** 2).sum() / ((eq - eq.mean()) ** 2).sum()
print(f"R^2 = {r2:.3f}   eq range {eq.min():.3f}..{eq.max():.3f}")
print("PREFLOP_W = np.array([" + ", ".join(f"{x:.6f}" for x in w) + "])")
