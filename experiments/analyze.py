"""Analyse ContextLab result CSVs: summaries + figures.

Usage:
    python experiments/analyze.py position
    python experiments/analyze.py length
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(pattern):
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def analyze_position():
    df = load("exp1_position_d4.csv")
    summ = (
        df.groupby(["position", "n_distractors"], as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             p_correct=("prob_correct", "mean"))
        .sort_values("position")
    )
    print("=== position sweep (d=4) ===")
    print(summ.to_string(index=False))
    summ.to_csv(RESULTS / "summary_position.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(summ["position"], summ["accuracy"], "o-", label="d=4 (interference)")
    d0_path = RESULTS / "exp1_position_d0.csv"
    if d0_path.exists():
        d0 = (pd.read_csv(d0_path)
              .groupby("position", as_index=False)["correct"].mean())
        plt.plot(d0["position"], d0["correct"], "s--", label="d=0 (none)")
    plt.xlabel("needle position (fraction of context)")
    plt.ylabel("accuracy")
    plt.title("Retrieval accuracy vs needle position (L=512)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_position.png", dpi=120)
    print("saved figure_position.png")


def analyze_length():
    df = pd.concat(
        [load("exp2_length_d*.csv")], ignore_index=True
    )
    summ = (
        df.groupby(["length", "n_distractors"], as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             p_correct=("prob_correct", "mean"))
        .sort_values(["n_distractors", "length"])
    )
    print("=== length sweep ===")
    print(summ.to_string(index=False))
    summ.to_csv(RESULTS / "summary_length.csv", index=False)

    plt.figure(figsize=(6, 4))
    for d, grp in summ.groupby("n_distractors"):
        plt.plot(grp["length"], grp["accuracy"], "o-", label=f"d={d}")
    plt.xscale("log", base=2)
    plt.xlabel("context length (tokens)")
    plt.ylabel("accuracy")
    plt.title("Retrieval accuracy vs context length")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_length.png", dpi=120)
    print("saved figure_length.png")


def analyze_interference():
    df = pd.read_csv(RESULTS / "pilot_difficulty.csv")
    summ = (
        df.groupby("n_distractors", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             p_correct=("prob_correct", "mean"))
        .sort_values("n_distractors")
    )
    print("=== interference scan (L=512, pos=0.5) ===")
    print(summ.to_string(index=False))
    summ.to_csv(RESULTS / "summary_interference.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summ["n_distractors"], summ["accuracy"], "o-", label="accuracy")
    ax.plot(summ["n_distractors"], summ["p_correct"], "s--", label="p_correct")
    ax.set_xlabel("number of decoy passages (d)")
    ax.set_ylabel("score")
    ax.set_title("Interference collapses retrieval (L=512, pos=0.5)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_interference.png", dpi=120)
    print("saved figure_interference.png")


def analyze_exp4():
    df = pd.read_csv(RESULTS / "exp4_length_decoys.csv")
    summ = (
        df.groupby(["length", "n_distractors", "similarity"], as_index=False)
        .agg(n=("correct", "size"),
             interference_load=("interference_load", "first"),
             accuracy=("correct", "mean"), p_correct=("prob_correct", "mean"))
    )
    summ.to_csv(RESULTS / "summary_exp4.csv", index=False)
    piv = summ.pivot(index="length", columns="n_distractors", values="accuracy")
    print("=== Exp4 accuracy: length x decoys (real prose) ===")
    print(piv.to_string())

    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("decoys (D)")
    ax.set_ylabel("context length (L)")
    ax.set_title("Accuracy: length x interference (real prose)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_exp4_heatmap.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for L, g in summ.groupby("length"):
        g = g.sort_values("interference_load")
        ax.plot(g["interference_load"], g["accuracy"], "o-", label=f"L={L}")
    ax.set_xlabel("interference load  I = D x w(S)")
    ax.set_ylabel("accuracy")
    ax.set_title("Collapse test: accuracy vs interference load")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_exp4_collapse.png", dpi=120)
    print("saved exp4 figures")


def analyze_exp5():
    df = pd.read_csv(RESULTS / "exp5_similarity.csv")
    summ = (
        df.groupby(["similarity", "n_distractors"], as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             p_correct=("prob_correct", "mean"))
    )
    order = ["low", "medium", "high"]
    summ["similarity"] = pd.Categorical(summ["similarity"], order, ordered=True)
    summ = summ.sort_values("similarity")
    print("=== Exp5 similarity sweep (L=1024, D=4) ===")
    print(summ.to_string(index=False))
    summ.to_csv(RESULTS / "summary_exp5.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(range(len(summ)), summ["accuracy"], color="#1f4e79")
    ax.set_xticks(range(len(summ)))
    ax.set_xticklabels(summ["similarity"])
    ax.set_xlabel("decoy similarity S")
    ax.set_ylabel("accuracy")
    ax.set_title("Effect of decoy similarity (L=1024, D=4)")
    ax.set_ylim(0, 1.08)
    for i, v in enumerate(summ["accuracy"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_exp5_similarity.png", dpi=120)
    print("saved exp5 figure")


def analyze_collapse():
    frames = []
    for f in ["exp4_length_decoys.csv", "exp4b_L4096.csv", "exp5_similarity.csv"]:
        p = RESULTS / f
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    w = {"low": 0.1, "medium": 0.5, "high": 1.0}
    g = (df.groupby(["length", "n_distractors", "similarity"], as_index=False)
         .agg(n=("correct", "size"), accuracy=("correct", "mean"),
              p_correct=("prob_correct", "mean")))
    g["I"] = g["n_distractors"] * g["similarity"].map(w) / g["length"] * 1000.0
    g = g.sort_values("I")
    print("=== Collapse test: accuracy vs interference density I ===")
    print(g.to_string(index=False))
    g.to_csv(RESULTS / "summary_collapse.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for L, gL in g.groupby("length"):
        ax.scatter(gL["I"], gL["accuracy"], s=55, label=f"L={L}")
    ax.set_xlabel("interference density  I = D x w(S) / L  (x1000)")
    ax.set_ylabel("accuracy")
    ax.set_title("Collapse test: accuracy vs interference density")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(title="context length")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_collapse.png", dpi=120)
    print("saved figure_collapse.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "position"
    if what == "position":
        analyze_position()
    elif what == "length":
        analyze_length()
    elif what == "interference":
        analyze_interference()
    elif what == "exp4":
        analyze_exp4()
    elif what == "exp5":
        analyze_exp5()
    elif what == "collapse":
        analyze_collapse()
    else:
        raise SystemExit("unknown analysis stage")
