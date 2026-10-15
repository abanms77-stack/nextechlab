"""Phase 10 - cross-domain validation (analyzer + figure).

Reads results/phase10_transfer.csv and results/phase10_domainB_factorial.csv
and produces a predicted-vs-observed figure for the domain-B cells (reference
wiki curve vs new general-English curve, with frozen-form predictions), and
prints the preregistered answers. All numbers are copied from the evaluator,
not recomputed.

Usage:
  python experiments/phase10_analyze.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    tr = pd.read_csv(RESULTS / "phase10_transfer.csv")
    db = pd.read_csv(RESULTS / "phase10_domainB_factorial.csv")

    print("=== Phase 10 transfer metrics (zero-refit) ===")
    print(tr.to_string(index=False))

    # domain-B observed cells + reference (wiki) curve for context
    cell = (db.groupby("n_distractors", as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))
    ref = pd.read_csv(RESULTS / "expB_dsweep_cells.csv").sort_values("n_distractors")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    # Left: domain-B observed vs frozen-form predictions
    ax = axes[0]
    ax.plot(ref["n_distractors"], ref["acc"], "o--", color="0.6",
            label="reference (wiki) observed", zorder=4)
    ax.plot(cell["n_distractors"], cell["acc"], "ks-", label="domain-B observed",
            linewidth=2, zorder=5)
    ax.set_xlabel("D (competing passages)")
    ax.set_ylabel("P(correct)")
    ax.set_title("Phase 10: domain transfer (new filler domain)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Right: MAE by form across targets (grouped)
    ax = axes[1]
    piv = tr.pivot_table(index="form", columns="target", values="MAE")
    order = [c for c in ["domainB", "crosslen_Smedium", "crosslen_Shigh",
                         "crosslen_Slow"] if c in piv.columns]
    piv = piv.reindex(["log", "linear", "min_sat", "frozen_law"])[order]
    piv.plot(kind="bar", ax=ax, rot=20)
    ax.set_ylabel("MAE (lower = better)")
    ax.set_title("Zero-refit MAE by form and target")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS / "figure_phase10_domain.png", dpi=120)
    print(f"\nsaved {RESULTS / 'figure_phase10_domain.png'}")

    # preregistered answers
    log_row = tr[(tr.target == "domainB") & (tr.form == "log")].iloc[0]
    law_row = tr[(tr.target == "domainB") & (tr.form == "frozen_law")].iloc[0]
    print("\n=== preregistered answers ===")
    print(f"Q1 domain-B Spearman (log) = {log_row['Spearman']:.3f} "
          f"(threshold 0.8)")
    print(f"Q2 domain-B MAE (log) = {log_row['MAE']:.3f} (threshold 0.15)")
    print(f"Q3 does log beat frozen law on domain-B MAE? "
          f"{'YES' if log_row['MAE'] < law_row['MAE'] else 'NO'} "
          f"({log_row['MAE']:.3f} vs {law_row['MAE']:.3f})")
    print(f"Q4 monotonicity violations (domainB): "
          f"{dict(zip(tr[tr.target=='domainB']['form'], tr[tr.target=='domainB']['mono_viol']))}")


if __name__ == "__main__":
    main()
