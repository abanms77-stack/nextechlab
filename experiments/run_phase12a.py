"""Phase 12A - long-context screening runner (the only model run of 12A).

Runs retrieval at L in {2048,4096,8192} x D in {0,4,8}, S=medium,
position=0.5, n=5/cell (45 samples) using the SAME wiki-prose filler as the
reference. Length is token-exact, so filler amount / query / evidence / decoys
are matched and L is the only variable across the L axis.

ROBUSTNESS (this box has known multi-minute stalls, and 8192 cells are slow):
  * BATCHED by length: L=2048 -> L=4096 -> L=8192, each batch logged with
    start/end + row count. If a later batch dies, earlier batches are safe.
  * RESUMABLE: re-running skips any (sweep, L, pos, D, S, sample) already in
    the CSV - it never overwrites finished rows.
  * APPEND-ONLY: every completed sample is flushed to disk immediately.
  * VERIFIED: at the end, asserts the expected 45 rows (or reports a short
    count + how to resume) instead of failing silently.

Output:
  results/phase12a_factorial.csv   (append-only, resumable)
  results/log_phase12a.txt         (batch start/end + timing)

Usage:
  python experiments/run_phase12a.py [model] [--limit=N]
"""

import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else "Qwen/Qwen2.5-0.5B"
LIMIT = None
for _a in sys.argv[1:]:
    if _a.startswith("--limit="):
        LIMIT = int(_a.split("=", 1)[1])

FILLER_PATH = "data/wiki_train.txt"  # same domain as reference
OUT = "results/phase12a_factorial.csv"
LOG = "results/log_phase12a.txt"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
LENGTHS = [2048, 4096, 8192]
D_GRID = [0, 4, 8]
POSITION = 0.5
SIMILARITY = "medium"
N_SAMPLES = 5
SWEEP = "phase12a_screen"
EXPECTED = len(LENGTHS) * len(D_GRID) * N_SAMPLES  # 45


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _done_rows() -> int:
    if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
        return 0
    import pandas as pd
    try:
        return len(pd.read_csv(OUT))
    except Exception:
        return 0


def main():
    t0 = time.time()
    log(f"START model={MODEL} expected={EXPECTED} existing={_done_rows()}")
    model = LocalHFModel(MODEL, threads=8, max_context=8192)
    cand_ids = [
        model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in CANDIDATES
    ]
    filler = ProseFiller(model.tokenizer, FILLER_PATH, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    log(f"model={MODEL} filler={filler.kind} size={filler.size} "
        f"path={FILLER_PATH} max_context={model.max_context}")

    lengths = LENGTHS[:LIMIT] if LIMIT else LENGTHS
    for L in lengths:
        b_t0 = time.time()
        cells = [(L, POSITION)]
        for d in D_GRID:
            run_grid(model, gen, cells,
                     n_samples=N_SAMPLES, sweep=SWEEP, n_distractors=d,
                     similarity=SIMILARITY, out_csv=OUT)
        log(f"BATCH L={L} done in {time.time()-b_t0:.0f}s "
            f"(cumulative rows={_done_rows()}/{EXPECTED})")

    rows = _done_rows()
    elapsed = time.time() - t0
    if rows >= EXPECTED:
        log(f"ALL DONE rows={rows}/{EXPECTED} in {elapsed:.0f}s "
            f"(verified)")
    else:
        log(f"INCOMPLETE rows={rows}/{EXPECTED} in {elapsed:.0f}s - "
            f"re-run to resume (skips completed samples)")
        sys.exit(2)


if __name__ == "__main__":
    main()
