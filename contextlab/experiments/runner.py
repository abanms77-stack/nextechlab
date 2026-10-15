"""Experiment runner.

Runs a model over a grid of (context length, needle position) cells with N
samples each, scoring needle retrieval as a K-way classification over the
candidate vocabulary. Results are appended to a CSV *incrementally* so long
runs can be monitored and never lose progress.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_COLUMNS = [
    "sweep", "length", "position", "n_distractors", "similarity",
    "interference_load", "sample", "seed", "answer_word", "correct",
    "prob_correct", "argmax_id", "answer_id", "context_tokens", "seconds",
]


def _open_writer(out_csv: Optional[str]):
    if out_csv is None:
        return None, None
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    is_new = (not os.path.exists(out_csv)) or os.path.getsize(out_csv) == 0
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=_COLUMNS)
    if is_new:
        w.writeheader()
        f.flush()
    return w, f


def run_grid(
    model,
    generator,
    cells: Sequence[Tuple[int, float]],
    n_samples: int = 8,
    out_csv: Optional[str] = None,
    sweep: str = "grid",
    n_distractors: int = 0,
    similarity: str = "medium",
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Run every (length, position) cell with `n_samples` samples.

    Resumable: any (length, position, D, S, sample) already present in
    `out_csv` is skipped, so an interrupted run can simply be re-launched.
    """
    done = set()
    if out_csv and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        try:
            ex = pd.read_csv(out_csv)
            for _, r in ex.iterrows():
                done.add((str(r["sweep"]), int(r["length"]),
                          round(float(r["position"]), 4),
                          int(r["n_distractors"]), str(r["similarity"]),
                          int(r["sample"])))
        except Exception:
            done = set()

    writer, fh = _open_writer(out_csv)
    rows: List[Dict] = []
    t_start = time.time()
    try:
        for length, pos in cells:
            cell_t0 = time.time()
            cell_rows: List[Dict] = []
            for k in range(n_samples):
                key = (str(sweep), int(length), round(pos, 4),
                       int(n_distractors), str(similarity), k)
                if key in done:
                    continue
                seed = int(round(pos * 1000)) * 1000 + k + length
                sample = generator.generate(
                    length=length, position=pos, seed=seed,
                    n_distractors=n_distractors, similarity=similarity,
                )
                t0 = time.time()
                correct, prob, argmax, _ = model.predict(
                    sample.input_ids, sample.candidate_ids,
                    sample.answer_token_id,
                )
                dt = time.time() - t0
                row = {
                    "sweep": sweep,
                    "length": length,
                    "position": round(pos, 4),
                    "n_distractors": n_distractors,
                    "similarity": similarity,
                    "interference_load": sample.meta["interference_load"],
                    "sample": k,
                    "seed": seed,
                    "answer_word": sample.answer_word,
                    "correct": correct,
                    "prob_correct": round(prob, 6),
                    "argmax_id": argmax,
                    "answer_id": sample.answer_token_id,
                    "context_tokens": sample.context_tokens,
                    "seconds": round(dt, 3),
                }
                rows.append(row)
                cell_rows.append(row)
                if writer is not None:
                    writer.writerow(row)
                    fh.flush()
            if cell_rows:
                acc = sum(r["correct"] for r in cell_rows) / len(cell_rows)
                log(
                    f"[{sweep}] L={length:>6} pos={pos:<5} "
                    f"d={n_distractors:<3} s={similarity:<6} acc={acc:.2f} "
                    f"cell={time.time()-cell_t0:.1f}s "
                    f"elapsed={time.time()-t_start:.0f}s"
                )
    finally:
        if fh is not None:
            fh.close()
    return pd.DataFrame(rows)


def run_position_sweep(
    model,
    generator,
    length: int,
    positions: Sequence[float],
    n_samples: int = 8,
    out_csv: Optional[str] = None,
    n_distractors: int = 0,
    similarity: str = "medium",
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    cells = [(length, p) for p in positions]
    return run_grid(
        model, generator, cells, n_samples=n_samples, out_csv=out_csv,
        sweep="position", n_distractors=n_distractors, similarity=similarity,
        log=log,
    )


def run_length_sweep(
    model,
    generator,
    lengths: Sequence[int],
    position: float = 0.5,
    n_samples: int = 8,
    out_csv: Optional[str] = None,
    n_distractors: int = 0,
    similarity: str = "medium",
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    cells = [(int(L), position) for L in lengths]
    return run_grid(
        model, generator, cells, n_samples=n_samples, out_csv=out_csv,
        sweep="length", n_distractors=n_distractors, similarity=similarity,
        log=log,
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Group a results frame into per-cell accuracy."""
    g = df.groupby(
        ["sweep", "length", "position", "n_distractors", "similarity"],
        as_index=False,
    ).agg(
        n=("correct", "size"),
        interference_load=("interference_load", "first"),
        accuracy=("correct", "mean"),
        mean_prob_correct=("prob_correct", "mean"),
    )
    return g.sort_values(
        ["sweep", "similarity", "n_distractors", "length", "position"]
    ).reset_index(drop=True)
