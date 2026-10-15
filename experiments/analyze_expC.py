"""Phase C analysis - robustness of accuracy to position and filler source."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    df = pd.read_csv(RESULTS / "expC_robustness.csv")
    g = (df.groupby(["sweep", "position"], as_index=False)
         .agg(n=("correct", "size"), acc=("correct", "mean"),
              mean_p=("prob_correct", "mean")))
    print("=== robustness: position x filler source (L=1024, D=4, medium) ===")
    print(g.to_string(index=False))
    g.to_csv(RESULTS / "summary_expC.csv", index=False)

    for sw, sub in g.groupby("sweep"):
        rng = sub["acc"].max() - sub["acc"].min()
        print(f"  {sw:<10} acc range across positions = {rng:.2f}")
    prose = g[g.sweep == "pos_prose"]["acc"].mean()
    bank = g[g.sweep == "pos_bank"]["acc"].mean()
    print(f"\n  mean acc PROSE={prose:.2f}  BANK={bank:.2f}  "
          f"(filler source delta = {prose-bank:+.2f})")


if __name__ == "__main__":
    main()