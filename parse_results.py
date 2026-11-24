"""Collect D3 ablation results from logs/*.txt into tables.

Usage (from the repo folder):  python parse_results.py
Outputs:
  results_long.csv   one row per run (condition, seed, subset, AP)
  results_table.csv  AP per subset (rows) x condition (columns), seeds averaged, plus mAP row
  results_summary.csv  mAP per condition: mean, std and per-seed values
and prints the summary plus the accept / reject / partial decision.
"""
import re
from pathlib import Path

import pandas as pd

LOG_DIR = Path("logs")
SUBSETS = ["Crafter", "Gen2", "HotShot", "Lavie", "ModelScope",
           "MoonValley", "MorphStudio", "Show_1", "Sora", "WildScrape"]
CONDITIONS = ["XCLIP-16", "ResNet-18", "ResNet-18-rand", "XCLIP-16-rand", "RandProj"]
# "-f8" = control run with every video scored on its first 8 frames (eval.py --num-frames 8)
CONDITIONS += [c + "-f8" for c in CONDITIONS]
ACCEPT, REJECT = 88.0, 70.0

rows, missing = [], []
for f in sorted(LOG_DIR.glob("*.txt")):
    name = f.stem
    sub = next((s for s in SUBSETS if name.endswith("_" + s)), None)
    if sub is None:
        continue
    prefix = name[: -len(sub) - 1]
    m = re.match(r"^(.*)_s(\d+)$", prefix)
    enc, seed = (m.group(1), int(m.group(2))) if m else (prefix, None)

    ap = re.search(r"AP Score:\s*([0-9.]+)", f.read_text(errors="ignore"))
    if ap is None:
        missing.append(f.name)
        continue
    rows.append({"condition": enc, "seed": seed, "subset": sub,
                 "AP": float(ap.group(1)) * 100})

df = pd.DataFrame(rows)
df.to_csv("results_long.csv", index=False)

# Per-subset AP, averaged over seeds for the random conditions
table = (df.groupby(["subset", "condition"])["AP"].mean()
           .unstack("condition")
           .reindex(index=SUBSETS, columns=[c for c in CONDITIONS if c in df.condition.unique()]))
table.loc["mAP"] = table.mean()
table.round(2).to_csv("results_table.csv")

# mAP per (condition, seed), then mean / std over seeds
per_seed = df.groupby(["condition", "seed"], dropna=False)["AP"].agg(["mean", "count"])
summary = []
for cond in CONDITIONS:
    if cond not in df.condition.values:
        continue
    s = per_seed.loc[cond]
    complete = s[s["count"] == len(SUBSETS)]["mean"]
    summary.append({
        "condition": cond,
        "mAP_mean": complete.mean(),
        "mAP_std": complete.std(ddof=1) if len(complete) > 1 else 0.0,
        "n_seeds_complete": len(complete),
        "per_seed_mAP": "; ".join(f"{v:.2f}" for v in complete.values),
    })
summary = pd.DataFrame(summary)
summary.round(2).to_csv("results_summary.csv", index=False)

pd.set_option("display.width", 200)
print("\nAP (%) per subset, seeds averaged:\n")
print(table.round(2).to_string())
print("\nmAP per condition:\n")
print(summary.round(2).to_string(index=False))

# Decision rule (fixed in advance): conditions 3 and 4
m = summary.set_index("condition")["mAP_mean"]
c3, c4 = m.get("ResNet-18-rand"), m.get("XCLIP-16-rand")
if c3 is not None and c4 is not None:
    if c3 > ACCEPT and c4 > ACCEPT:
        verdict = "ACCEPTED (both random encoders > 88 mAP)"
    elif c3 < REJECT and c4 < REJECT:
        verdict = "REJECTED (both random encoders < 70 mAP)"
    else:
        verdict = "PARTIAL DEPENDENCE"
    print(f"\nHypothesis: {verdict}  [ResNet-18-rand {c3:.2f}, XCLIP-16-rand {c4:.2f}]")
if "RandProj" in m:
    print(f"RandProj (read separately): {m['RandProj']:.2f} mAP")

expected = 2 * len(SUBSETS) + 3 * 3 * len(SUBSETS) + 5 * len(SUBSETS)  # paper protocol + f8 (seed 0)
print(f"\nRuns parsed: {len(df)} / {expected}")
if missing:
    print("Logs without an AP score (failed or unfinished):")
    for name in missing:
        print("  ", name)