"""Phase 8 - D functional-form study (analyzer + figure).

Reads results/phase8_forms.csv and results/phase8_loo.csv and produces a
predicted-vs-observed figure for the held-out D curve, and prints the
preregistered answers. All numbers are copied from the fitter, not recomputed.

Usage:
    python experiments/phase8_analyze.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    forms = pd.read_csv(RESULTS / "phase8_forms.csv")
    loo = pd.read_csv(RESULTS / "phase8_loo.csv")
    print("=== Phase 8 functional forms (held-out, leave-one-D-out) ===")
    print(forms.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    obs = loo[loo.form == "log"].sort_values("D")
    ax.plot(obs["D"], obs["observed"], "ko-", label="observed", zorder=5)
    for form in ["linear", "min_sat", "exp_sat", "log", "isotonic"]:
        s = loo[loo.form == form].sort_values("D")
        ax.plot(s["D"], s["pred"], "--", alpha=0.8, label=f"{form} (held-out)")
    ax.set_xlabel("D (competing passages)")
    ax.set_ylabel("P(correct)")
    ax.set_title("Phase 8: f(D) held-out predictions vs observed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_phase8_forms.png", dpi=120)
    print(f"\nsaved {RESULTS / 'figure_phase8_forms.png'}")


if __name__ == "__main__":
    main()
