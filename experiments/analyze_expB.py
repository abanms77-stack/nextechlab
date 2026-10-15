"""Phase B analysis - does profiler-guided pruning improve accuracy?

Compares accuracy across FULL / NONE / RANDOM / SIM conditions and tests
SIM > RANDOM. Reports paired differences and a bootstrap CI on the gain.

Usage:
    python experiments/analyze_expB.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    df = pd.read_csv(RESULTS / "expB_intervention.csv")
    acc = df.groupby("condition", as_index=False).agg(
        n=("correct", "size"), acc=("correct", "mean"),
        mean_p=("prob_correct", "mean"))
    print("=== intervention conditions (L=1024, 8 decoys, prune 4) ===")
    print(acc.to_string(index=False))

    piv = df.pivot_table(index="sample", columns="condition",
                         values="correct")
    print("\npaired per-sample accuracy:")
    print(piv.to_string())

    for a, b in [("SIM", "RANDOM"), ("SIM", "FULL"), ("RANDOM", "FULL"),
                 ("NONE", "FULL"), ("SIM", "NONE")]:
        if a in piv and b in piv:
            diff = (piv[a] - piv[b]).to_numpy()
            lo, hi = boot_ci(diff)
            print(f"  {a:<6} - {b:<6} = {diff.mean():+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]")

    order = ["FULL", "RANDOM", "SIM", "NONE"]
    means = [acc.set_index("condition")["acc"].get(c, np.nan) for c in order]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar(order, means, color=["#999", "#b5651d", "#1f4e79", "#2e7d32"])
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Intervention: pruning by measured interference\n"
                 "(SIM = remove highest-similarity decoys)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_expB_intervention.png", dpi=120)
    print("saved figure_expB_intervention.png")


if __name__ == "__main__":
    main()