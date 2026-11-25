"""95% bootstrap CIs over the 10 generator subsets, from results_long.csv.

Resamples subsets (not videos: the 110 runs have no per-video scores).
Prints mAP CIs, P(mAP < 88), and CIs on the pre-training gain.
"""
import numpy as np
import pandas as pd

B, SEED, ACCEPT = 10_000, 0, 88.0

t = pd.read_csv("results_long.csv").groupby(["subset", "condition"]).AP.mean().unstack()
assert t.shape == (10, 5), t.shape
idx = np.random.default_rng(SEED).integers(0, len(t), (B, len(t)))


def ci(values):
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return values.mean(), lo, hi, (boot < ACCEPT).mean()


print(f"{'condition':<16}{'mAP':>7}{'CI low':>8}{'CI high':>8}{'P(<88)':>8}")
for c in t:
    m, lo, hi, p = ci(t[c].values)
    print(f"{c:<16}{m:7.2f}{lo:8.2f}{hi:8.2f}{p:8.3f}")

print("\npre-training gain (pre-trained - random)")
for pre, rnd in [("XCLIP-16", "XCLIP-16-rand"), ("ResNet-18", "ResNet-18-rand")]:
    m, lo, hi, _ = ci((t[pre] - t[rnd]).values)
    print(f"{pre:<16}{m:7.2f}{lo:8.2f}{hi:8.2f}")
