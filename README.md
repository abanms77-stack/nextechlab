# ContextLab — measuring the *effective* context of LLMs

Research code for the **Effective Context Capacity (ECC)** programme.

> Thesis: a model's *advertised* context window is not the context it can
> actually use. We build tools to **measure**, then **predict**, then **move**
> that boundary.

## Why this exists

Nominal context (e.g. "1,000,000 tokens") overstates real usable context.
Prior work (Lost in the Middle, RULER, BABILong, "context rot") shows
degradation, but there is no *predictive, task-specific, pre-inference* account
of how much context a model can actually use. That is the gap this project
targets.

## Layout

```
contextlab/
  tasks/generator.py      # controllable token-level task generator
  models/local_hf.py      # CPU-friendly HF adapter (fp32 + sdpa)
  metrics/ecc.py          # ECC, Effective Context Mass, curve fits
  experiments/runner.py   # grid runner, writes CSV incrementally
  cli.py                  # `python -m contextlab.cli ...`
experiments/              # runnable experiment scripts
results/                  # CSV outputs (generated)
tests/                    # unit tests (no model required)
```

## Install

```
pip install -r requirements.txt
```

## Usage

Position sweep (reproduce "lost in the middle"), 512-token contexts, 4 decoy
passages:

```
python -m contextlab.cli position --model Qwen/Qwen2.5-0.5B \
    --length 512 --positions 0.05,0.2,0.35,0.5,0.65,0.8,0.95 \
    --samples 8 --distractors 4 --out results/position.csv
```

Length sweep + metric report:

```
python -m contextlab.cli length --model Qwen/Qwen2.5-0.5B \
    --lengths 256,512,1024,2048 --position 0.5 --samples 6 \
    --distractors 4 --out results/length.csv
```

## Phase 1B — real prose (WikiText) + similarity

`data/wiki_train.txt` is real WikiText prose used as filler (removes the
repetition confound). Fetch it once with `datasets` (see `run_exp4.py`), then:

```
python experiments/run_exp4.py main          # interference x length, real prose
python experiments/run_exp4.py rest          # L=4096 + similarity sweep
python experiments/analyze.py exp4           # heatmap + collapse
python experiments/analyze.py exp5           # similarity figure
python experiments/analyze.py collapse       # interference-density collapse test
```

## Phase 1B-2 — high-n factorial + fitted law

```
python experiments/run_exp6.py     # n=20 factorial (L x D x S), resumable
python experiments/fit_exp6.py     # fit P=f(D,S,L) + OOD tests -> figure_exp6_fit.png
```

## Phase 1B-3 — cross-model transfer (pre-registered)

```
python experiments/freeze_law.py     # freeze Qwen law + predict Llama cells FIRST
python experiments/run_exp7.py       # run the same factorial on Llama-3.2-1B
python experiments/transfer_eval.py  # frozen-law vs Llama -> figure_transfer.png
```

## Phase 1B-5/6 — third model (zero refit) + D-saturation sweep

```
# 1B-5: freeze once more, predict gpt2-medium cells FIRST, then run
python experiments/expA3_freeze_predict.py   # PRE-register gpt2-medium predictions
python experiments/expA3_run.py              # run identical factorial on gpt2-medium
python experiments/expA3_eval.py             # zero-refit eval -> expA3_gpt2medium_transfer_pairs.csv
python experiments/cross_model_validate.py   # both targets: zero-refit + LOO calibration

# 1B-6: characterize the interference count term (D_SATURATION was unjustified)
python experiments/run_expB.py               # D in {0,1,2,3,4,5,6,8,10,12,16}, Qwen
python experiments/analyze_expB_saturation.py  # saturation curve + best-fit k
```

## Phase 1B-4 — continuous similarity + third model

```
python experiments/exp8_make_prediction.py   # PRE-register similarity metrics
python experiments/exp8_run.py               # D=1 sweep (below detection floor)
python experiments/exp8_run_d4.py            # D=4 sweep  (Qwen, L=1024)
python experiments/exp8_run_d4.py gpt2-medium 768   # third model
python experiments/exp8_analyze.py qwen
python experiments/exp8_analyze.py gpt2-medium
python experiments/exp8_final.py             # cross-model figure
```

## Phase A-D — final checkpoints

```
python experiments/run_expA.py          # cross-task (counting) validation
python experiments/analyze_taskA.py
python experiments/expB_intervention.py # intervention: prune by similarity
python experiments/analyze_expB.py
python experiments/run_expC.py          # robustness: position x filler
python experiments/analyze_expC.py
python experiments/analyze_final.py     # bootstrap CIs + calibration -> final_stats.csv
```

## Build the PDF report

```
python experiments/analyze.py position      # -> results/figure_position.png
python experiments/analyze.py length         # -> results/figure_length.png
python experiments/analyze.py interference   # -> results/figure_interference.png
python experiments/build_pdf.py              # -> ContextLab_Report.pdf
```

## Phase 7 — real RAG intervention benchmark

End-to-end: MiniLM + FAISS retrieval over the local wiki → Qwen2.5-0.5B reader
→ K-way scoring over a frozen 12-word candidate set. 40 pre-registered queries.

```
python experiments/rag_preregister.py    # FREEZE queries + policy (sha256) FIRST
python experiments/run_rag_bench.py      # 3 conditions x 40 queries -> rag_bench.csv
python experiments/analyze_rag_bench.py  # paired deltas + bootstrap CI
# diagnostics:
python experiments/rag_diagnose_floor.py # reader floor (clean vs competing)
python experiments/rag_diagnose_ctx.py   # gold retrieval rank / context dump
```

**Measured result (honest):** reordering by profiler score moves accuracy
0.075 → 0.175 vs a matched random control (0.075), but the paired 95% CI lower
bound is 0.000 (not significant), and **no tokens are saved** (`n_removed = 0`
— the corpus is contradiction-rich, which the policy deliberately protects).
See REPORT.md "Phase 7" for the data-integrity bug that was found and fixed
(duplicate injected strings flooding the index) before any number was reported.

## Phase 7B — redundancy intervention (prune arm + ORACLE)

Phase 7 could only reorder (contradiction-rich → `removed=0`). Phase 7B adds the
**prune arm**: 40 preregistered queries × 4 context types (clean / redundant /
contradictory / mixed) × 4 conditions (BASELINE / RANDOM / CONTEXTLAB / **ORACLE**).

```
python experiments/rag_preregister_7b.py    # FREEZE queries + contexts (sha256) FIRST
python experiments/run_rag_bench_7b.py      # 4x4x40 -> rag7b_bench.csv
python experiments/analyze_rag_bench_7b.py  # preregistered C1-C5 + Pareto figure
python experiments/rag7b_sanity.py          # engine prunes redundant, protects conflict
```

**Measured (honest):** on **redundant** context ContextLab holds accuracy at
**1.000** while cutting context tokens **61.5 → 41.8 (−32%, SIG)** — the Pareto
win. BUT the matched RANDOM control also hits 1.000, so this shows *pruning
works*, not yet that the engine's *choice* beats alternatives. On
**contradictory** context the reorder arm *hurt* (0.350 → 0.250) and ORACLE
(drop conflicts) did better (0.525) — a real product-policy tension. See
REPORT.md "Phase 7B".

## Phase 7C — choice attribution (better than random?)

Only ONE passage yields the answer (the rest are graded/redundant), so RANDOM
removal risks deleting the gold while ContextLab prunes only redundant dups.

```
python experiments/rag_preregister_7c.py    # FREEZE 7C queries (sha256) FIRST
python experiments/run_rag_bench_7c.py      # 4 conditions -> rag7c_bench.csv
python experiments/analyze_rag_bench_7c.py  # preregistered A1-A5 + Pareto fig
python experiments/rag7c_sanity.py          # role retention / gold survival
```

**Measured (SIG):** CONTEXTLAB accuracy **0.800** vs RANDOM **0.550**
(+0.250, CI [+0.05,+0.43]) and vs BASELINE **0.500** (+0.300) at −38 tokens.
ContextLab keeps the gold **100%**; RANDOM deletes it 22.5% of the time
(accuracy collapses to 0.222 when it does). **This is the attribution result:
ContextLab makes *better* pruning decisions, not just "pruning works".**

## Phase 7D — conflict-resolution policies

```
python experiments/rag_preregister_7d.py    # FREEZE policies (sha256) FIRST
python experiments/run_rag_bench_7d.py      # 4 policies -> rag7d_bench.csv
python experiments/analyze_rag_bench_7d.py  # preregistered P1-P4
```

**Measured (SIG):** DROP_WEAKER accuracy **1.000** vs PROTECT **0.562**
(+0.438, CI [+0.34,+0.55]) with −30 tokens; on contradictory alone
0.350 → **1.000**. SURFACE (reorder only) does nothing (ns) ⇒ **resolution**, not
reordering, is what matters. Caveat: the tie-break (highest-token) selects the
gold on this construction, so it proves *"resolving beats protecting"*, not that
the proxy is a general gold-detector.

## Phase 7E — decoupled conflict resolution (the 7D confound removed)

7D's 100% was a confound: conflicts shared one template (equal token counts) and
the gold sat at index 0. 7E decouples **specificity from correctness** (verified:
P(correct)=0.250 by specificity, max dev 0.068 ≤ 0.12) and **randomizes position**.

```
python experiments/rag_preregister_7e.py    # FREEZE + anti-leakage check (sha256)
python experiments/run_rag_bench_7e.py      # 6 policies -> rag7e_bench.csv
python experiments/analyze_rag_bench_7e.py  # preregistered E1-E6
```

**Measured (honest negative):** every resolution heuristic collapses to **chance**
— DROP_WEAKER 0.300, DROP_MORE_SPECIFIC 0.250, DROP_LESS_SPECIFIC 0.275, RANDOM
0.275 vs base rate 0.25 (E5 DROP_WEAKER−RANDOM = +0.025, ns). Only ORACLE
(0.700 headroom, SIG) closes the gap. The 7D "win" did not survive the confound
fix ⇒ **conflict *detection* is solved; conflict *resolution without an oracle*
is an open problem.**

## Phase 8 — the functional form of the count term f(D)

`D_SATURATION = 5` was never empirically recovered. Phase 8 asks which form
`f(D)` in `P = σ(a + b·f(D))` best predicts accuracy on **held-out D** (protocol
frozen before fitting, sha256 `f1d548ac…`; leave-one-D-out on the D-sweep).

```
python experiments/phase8_forms.py           # FREEZE protocol (sha256)
python experiments/phase8_fit.py             # fit forms + held-out metrics
python experiments/phase8_analyze.py         # table + figure
```

**Measured:** **`log(1+D)` is the best form** — held-out MAE **0.056** and best
calibration (ECE 0.034) with only **2 parameters**, and it beats the frozen
7-feature law (MAE 0.094). `min(1,D/k)` extrapolates best (err 0.037 at D=16);
`1−e^(−D/k)` is worst and the only form with a monotonicity violation. Learned
isotonic (10 params) buys nothing over `log`. ⇒ The count term should be
`log(1+D)`, replacing the unsupported `D_SATURATION = 5`.

## Phase 9 — cross-task transfer (zero-refit)

Does the Phase-8 count structure survive a different task without refitting?
Protocol frozen before target (sha256 `4b8ca8da…`): fit on the retrieval D-sweep,
**freeze (a,b,k)**, evaluate on the **counting** task family.

```
python experiments/phase9_transfer.py --freeze   # freeze reference params
python experiments/phase9_transfer.py            # zero-refit transfer metrics
```

**Measured (outcome B, with a C signal):** on the L=1024 slice every frozen form
ranks the counting-task D-effect correctly (**Spearman 0.949**) — the *form*
transfers — but calibration is poor (ECE 0.18–0.24), i.e. the *coefficients* are
task-dependent; pooling the L=256 slice (where accuracy *rises* with D) inverts
the correlation to **ρ=−0.60** (condition-dependent). ⇒ supports
`R = F(task, f(D), S, L)` (task-conditioned reliability), not one universal law.

## Phase 10 — cross-domain validation (zero-refit)

Does the retrieval law survive a change of **context domain** (different filler
text) without refitting? Protocol frozen before any new cell ran (sha256
`604a8a96…`): new D-sweep cells with filler swapped from wiki prose to
**general English** (`NeelNanda/pile-10k`, URL-stripped, token-matched to the
reference within 0.82%), plus a cross-length zero-refit test on existing Exp6
data.

```
python experiments/phase10_freeze.py --build   # FREEZE protocol + build filler
python experiments/run_phase10.py              # new-domain cells (only model run)
python experiments/phase10_transfer.py         # zero-refit transfer metrics
python experiments/phase10_analyze.py          # table + figure
```

**Measured (outcome A):** on the new-domain cells (observed 1.00 → 0.50 → 0.20
across D) every frozen form ranks perfectly (**Spearman 1.000**), and the 2-param
**`log(1+D)` is best calibrated (MAE 0.137)** beating the 7-feature law (0.197).
Cross-length transfer ranks perfectly too but the *frozen law* calibrates better
where its S/L terms are active (MAE 0.0745 at S=high). ⇒ domain is another
condition that changes **coefficients, not functional form**; calibration depends
on which axis is held fixed. Verdicts preliminary (n=10/cell, 3 D points).

## Phase 11 — position effects (zero-refit)

Does the evidence **position** P (fraction of context before the needle)
explain residual error after D, S, L are already known? Protocol frozen
before any cell ran (sha256 `1bd50644…`): P ∈ {0.05,0.15,0.30,0.50,0.70,
0.85,0.95}, D ∈ {0,4}, L=256, S=medium, n=10/cell (140 trials), filler held
fixed. **Isolation is exact — every cell is 267 context tokens**, so length
and filler amount are constant and only position varies. Position coefficient
fit on {0.05,0.30,0.70,0.95}; evaluated **zero-refit** on held-out
{0.15,0.50,0.85}.

```bash
python experiments/phase11_freeze.py   # freeze protocol (sha256) + write json
python experiments/run_phase11.py      # 140 cells (only model run)
python experiments/phase11_transfer.py # zero-refit metrics + prereg verdict
python experiments/phase11_analyze.py  # figure + preregistered answers
```

**Measured (outcome B — modest calibration signal):** held-out test MAE
**0.049 → 0.027** and ECE **0.049 → 0.017** when the position term is added
(ΔR² **+0.128** on the fit-split, 0.841 → 0.968). But rank barely moves —
Spearman **0.949 → 0.906** — because D=0 is flat (1.00 across all P) and D=4
moves only 0.70 → 0.80. **The MAE gain is 0.021, below the preregistered A
threshold of ≥ 0.030**, so the verdict is **B**: position provides modest but
measurable *calibration* information, not enough to promote to a core law
term. ⇒ position is a **candidate calibration feature**; core law stays
`R = F(task, domain, log(1+D), S, L, …)` for now. Exploratory `exp1_position`
(L=512) showed a stronger mid-context dip but is **non-preregistered** and
does not feed the verdict. Verdicts preliminary (n=10/cell, 7 P, 2 D).

## Phase 12A — long-context scaling (zero-refit screening)

Before 12A, long-context scaling (how much of a window is usable) was a hole in
the evidence — Phases 8–11 all ran at L ≤ 1024. Protocol frozen before any cell
ran (sha256 `d30146e3…`): **L ∈ {2048,4096,8192} × D ∈ {0,4,8}, S=medium,
position=0.5, n=5/cell = 45 samples**, wiki filler fixed, length token-exact so
**L is the only variable across the length axis**. Short-L baselines reuse exp6
(n=20, L ∈ {256,1024}) + exp4 — zero new baseline runs.

```bash
python experiments/phase12a_freeze.py   # freeze protocol (sha256) + json
python experiments/run_phase12a.py      # 45 cells (batched, resumable)
python experiments/phase12a_eval.py     # apply frozen rule + model A/B/C
```

**Measured:** run **45/45 in 4582s, zero failures**. **D=0 = 1.00 at every
length** (2048 → 8192) — length alone is harmless. D=4: **0.80 → 0.40 → 0.00**;
D=8: 0.40 → 0.40 → 0.40. Aggregates over D∈{4,8} vs the L=256 baseline (0.575):
**harm4096 = +0.175**, **harm8192 = +0.375**, **deltaInt = −0.400**. Model
comparison: **M_A → M_B (adding L) is the major predictive jump** (log-loss
0.486 → 0.449, Brier 0.0475 → 0.0264); **M_C (+ L×D) adds only a little**
(0.4466). Frozen rule → **Stage A = INTERESTING** (triggers 12B) and **Stage B
= CONFIRMED at the preregistered screening threshold** (n=5 is screening, not
independent confirmation). **Long-context degradation is conditional on
competing evidence — not length alone** ⇒ triggers preregistered **Phase 12B**
(n=10–20, identical frozen spec). Preliminary pending 12B.

## Phase 12B — long-context confirmation (independent n=20)

12A was *screening* evidence (n=5). 12B repeats the **identical** experiment on
**fresh, independent samples** at n=20/cell, applying the **unchanged** frozen rule.
Protocol frozen before any cell ran (sha256 `7336f02c…`, byte-verified identical
thresholds/models to 12A): **L ∈ {2048,4096,8192} × D ∈ {0,4,8}, S=medium,
position=0.5, n=20/cell = 180 samples**, true sample index **t ∈ [5,25)** —
disjoint from 12A's [0,5) (min seed 502053), so **no seed collides** and the
confirmation set is genuinely independent. Same exp6 L=256 baseline (0.575).

```bash
python experiments/phase12b_freeze.py   # freeze protocol (sha256) + json
python experiments/run_phase12b.py      # 180 samples (batched, resumable, dedup)
python experiments/phase12b_eval.py     # apply UNCHANGED frozen rule + model A/B/C
```

**Measured:** run **180/180 in 11238s (~187 min)**, verified (resumed once after an
interrupt, no recomputation/seed drift). **D=0 = 1.00 at every length** on the fresh
sample. Aggregates over D∈{4,8} vs baseline 0.575: **harm8192 = +0.275**,
**deltaInt = −0.150** (harm4096 = −0.075, mid-length noise, not in the rule). Model
comparison: **M_A → M_B (adding L) improves again** (log-loss 0.4774 → 0.4723); **M_C
(+ L×D) adds nothing** (0.4720). Frozen rule with **unchanged thresholds** → **Stage A
= INTERESTING** and **Stage B = CONFIRMED** (0.275 ≥ 0.20 **and** −0.150 ≤ −0.05) on
fresh n=20. **Competition-dependent long-context penalty CONFIRMED** ⇒ long-context
arm of the evidence chain rests on a preregistered independent confirmation, not a
single screening pass. Caveat: S=medium, one model; cross-model/S robustness → Phase 13.

## Methodology (and honesty rules)

* **Token-exact tasks.** Prompts are assembled at the token level; context
  length and needle position are exact, not approximate.
* **Deterministic scoring.** The answer is a single token from a fixed
  12-word candidate set, so accuracy is a clean K-way classification.
  (Instruct-mode free generation is deliberately avoided on a *base* model.)
* **No hallucinated results.** Every number in the report comes from a CSV in
  `results/` produced by this code. Anything not measured is labelled a
  hypothesis.
* **Known limitations** are documented in the report, not hidden.

## Environment note (measured)

This machine is CPU-only (i5-12450H, 16 GB RAM, no GPU). The HF model loads in
bfloat16 by default; this CPU has no native BF16, so we force **float32 +
sdpa**, which is ~5–100× faster. Even so, per-forward time is a few seconds and
occasionally stalls for minutes (environmental), which caps experiment size.
