"""Phase 12B - long-context penalty CONFIRMATION runner (the only model run).

INDEPENDENT replication of Phase 12A: same L in {2048,4096,8192} x D in {0,4,8},
S=medium, position=0.5, wiki filler, but n=20/cell (180 samples) drawn from a
FRESH, DISJOINT sample-index range so no sample overlaps 12A's n=5.

  true sample index  t = k + K_OFFSET  in [5, 25)      (12A used t in [0,5))
  seed              = 500000 + t + L   = 500000 + k + 5 + L   (pos 0.5)

This runner does NOT call run_grid, because run_grid hardcodes
seed = 500000 + k + L (k = 0..n-1), which for k in [0,5) would reproduce 12A's
exact contexts (NOT independent) and would record a seed column identical to
12A's. Instead we drive the SAME row schema with our own loop so that:
  * the CSV `seed` column holds the TRUE independent seed (500000 + t + L),
    verified disjoint from 12A's 500000 + (0..4) + L and unique within 12B;
  * the CSV `sample` column holds the TRUE index t in [5,25), so the
    independence is directly auditable in the CSV;
  * resume-dedup keys on (sweep, L, pos, D, S, t) - consistent, no rewrite.
answer_word = rng.choice(candidates) depends only on the seed (not D) and length
is token-exact, so 12B draws are independent samples from the SAME population as
12A (a genuine replication, not a new design).

ROBUSTNESS (identical to 12A): batched by length, APPEND-only flush per sample,
RESUMABLE (re-run skips any (sweep, L, pos, D, S, t) already in the CSV), and an
end-of-run VERIFIED count instead of failing silently.

Output:
  results/phase12b_factorial.csv   (append-only, resumable, separate from 12A)
  results/log_phase12b.txt         (batch start/end + timing)

Usage:
  python experiments/run_phase12b.py [model] [--limit=N]
"""

import csv
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

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
OUT = "results/phase12b_factorial.csv"
LOG = "results/log_phase12b.txt"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
LENGTHS = [2048, 4096, 8192]
D_GRID = [0, 4, 8]
POSITION = 0.5
SIMILARITY = "medium"
N_SAMPLES = 20          # independent samples / cell (12A was 5)
K_OFFSET = 5            # true sample index t = k + 5  -> [5,25), disjoint vs 12A
SWEEP = "phase12b_confirm"
EXPECTED = len(LENGTHS) * len(D_GRID) * N_SAMPLES  # 180
_COLUMNS = [
    "sweep", "length", "position", "n_distractors", "similarity",
    "interference_load", "sample", "seed", "answer_word", "correct",
    "prob_correct", "argmax_id", "answer_id", "context_tokens", "seconds",
]


def _true_t(k: int) -> int:
    return k + K_OFFSET


def _seed(t: int, length: int) -> int:
    """seed = round(pos*1000)*1000 + t + length, exactly as runner does (pos 0.5)."""
    return int(round(POSITION * 1000)) * 1000 + t + length


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _done_keys() -> set:
    """(L, pos, D, S, true_t) already present in the CSV (resume dedup)."""
    if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
        return set()
    import pandas as pd
    try:
        ex = pd.read_csv(OUT)
    except Exception:
        return set()
    done = set()
    for _, r in ex.iterrows():
        if str(r["sweep"]) != SWEEP:
            continue
        done.add((int(r["length"]), round(float(r["position"]), 4),
                  int(r["n_distractors"]), str(r["similarity"]),
                  int(r["sample"])))
    return done


def _done_rows() -> int:
    if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
        return 0
    import pandas as pd
    try:
        return len(pd.read_csv(OUT))
    except Exception:
        return 0


def _open_writer():
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    is_new = (not os.path.exists(OUT)) or os.path.getsize(OUT) == 0
    f = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=_COLUMNS)
    if is_new:
        w.writeheader()
        f.flush()
    return w, f


def _run_cell(model, gen, L, d, writer, fh, done):
    """Run one (L, D) cell: N_SAMPLES independent samples t in [5,25)."""
    cell_t0 = time.time()
    acc_rows = []
    for k in range(N_SAMPLES):
        t = _true_t(k)                      # true sample index in [5, 25)
        key = (L, round(POSITION, 4), d, SIMILARITY, t)
        if key in done:
            continue
        seed = _seed(t, L)                  # 500000 + t + L  (disjoint vs 12A)
        sample = gen.generate(
            length=L, position=POSITION, seed=seed,
            n_distractors=d, similarity=SIMILARITY,
        )
        t0 = time.time()
        correct, prob, argmax, _ = model.predict(
            sample.input_ids, sample.candidate_ids, sample.answer_token_id,
        )
        dt = time.time() - t0
        row = {
            "sweep": SWEEP, "length": L, "position": round(POSITION, 4),
            "n_distractors": d, "similarity": SIMILARITY,
            "interference_load": sample.meta["interference_load"],
            "sample": t, "seed": seed, "answer_word": sample.answer_word,
            "correct": correct, "prob_correct": round(prob, 6),
            "argmax_id": argmax, "answer_id": sample.answer_token_id,
            "context_tokens": sample.context_tokens, "seconds": round(dt, 3),
        }
        writer.writerow(row)
        fh.flush()
        done.add(key)
        acc_rows.append(row)
    if acc_rows:
        acc = sum(r["correct"] for r in acc_rows) / len(acc_rows)
        log(f"[{SWEEP}] L={L:>6} pos={POSITION:<5} d={d:<3} "
            f"s={SIMILARITY:<6} acc={acc:.2f} "
            f"cell={time.time()-cell_t0:.1f}s "
            f"(t=({_true_t(0)},{_true_t(N_SAMPLES)}))")


def main():
    t0 = time.time()
    log(f"START model={MODEL} expected={EXPECTED} existing={_done_rows()} "
        f"seed_offset=+{K_OFFSET} (true t=[{K_OFFSET},{K_OFFSET + N_SAMPLES}))")
    model = LocalHFModel(MODEL, threads=8, max_context=8192)
    cand_ids = [
        model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in CANDIDATES
    ]
    filler = ProseFiller(model.tokenizer, FILLER_PATH, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    log(f"model={MODEL} filler={filler.kind} size={filler.size} "
        f"path={FILLER_PATH} max_context={model.max_context}")

    done = _done_keys()
    writer, fh = _open_writer()
    try:
        lengths = LENGTHS[:LIMIT] if LIMIT else LENGTHS
        for L in lengths:
            b_t0 = time.time()
            for d in D_GRID:
                _run_cell(model, gen, L, d, writer, fh, done)
            log(f"BATCH L={L} done in {time.time()-b_t0:.0f}s "
                f"(cumulative rows={_done_rows()}/{EXPECTED})")
    finally:
        fh.close()

    rows = _done_rows()
    elapsed = time.time() - t0
    if rows >= EXPECTED:
        log(f"ALL DONE rows={rows}/{EXPECTED} in {elapsed:.0f}s (verified)")
    else:
        log(f"INCOMPLETE rows={rows}/{EXPECTED} in {elapsed:.0f}s - "
            f"re-run to resume (skips completed samples)")
        sys.exit(2)


if __name__ == "__main__":
    main()
