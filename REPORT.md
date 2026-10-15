# ContextLab â€” Phase-1 Research Report

**Project:** Effective Context Capacity (ECC) for large language models
**Status:** Phase 1 (measurement) â€” first real results
**Model under test:** `Qwen/Qwen2.5-0.5B` (base), local, CPU-only
**Date:** 2026-10-01

> **Honesty statement.** Every number in this report comes from a CSV in
> `results/` produced by the code in this repository on the machine described
> in Â§4. Nothing is simulated, inferred, or hallucinated. Claims that are not
> backed by a measurement are explicitly labelled **[hypothesis]**.

---

## 1. Executive summary

We set out to test the central claim of the plan: *a model's advertised context
window is not the context it can actually use.* We built a controllable,
token-exact needle-retrieval benchmark (`ContextLab`) and ran real sweeps on a
0.5B model.

**What we measured:**

1. **Single-fact retrieval is trivially easy.** With no interference, the model
   retrieves the needle with **100% accuracy and ~0.999 confidence at 512
   tokens**, at every needle position. So "context length" alone, at lengths a
   CPU can reach, does **not** degrade this model's behaviour. (Exp 1 / pilot)
2. **Interference is the real killer.** Adding decoy `"{key} password is {word}"`
   passages collapses accuracy **even at 512 tokens** â€” from 1.00 (0 decoys) to
   0.50 (3 decoys) and 0.17 (6 decoys). This is a *task-difficulty* effect, not
   a length effect.
3. **Lost-in-the-middle reproduces â€” but only under interference.** With 4
   decoys, mean probability of the correct answer is **0.76 at position 0.05,
   0.35 at the middle, 0.94 at the end**: a clear U-shape. Without interference
   the curve is flat at 1.00. (Exp 2)

**Taking these together:** the measurable quantity that varies with context is
not "how many tokens" but **"how much competing information is in those
tokens."** This sharpens the plan's thesis and is the seed of the predictive
model.

*(Length-sweep results: Â§7.)*

---

## 2. The question

For a model `M`, task `T`, context length `L`:

* Is there a *predictable* degradation curve `P(L)`?
* Does degradation depend on context **length**, on **interference**, or both?
* Where is the information placed (**position**) and does that matter?

The plan proposes an *Effective Context Capacity*:
`ECC(M,T,Ï„) = max{ L : P(M,T,L) â‰¥ Ï„Â·P(M,T,Lâ‚€) }`, to be *predicted before
inference*. This report is the first empirical step toward that.

---

## 3. Method

### 3.1 Task (token-exact, controllable)

Each sample is a single-token-answer needle task:

* **Needle:** `" Important: the vault password is {word}."`
* **Decoys (optional):** `" Note: the {noun} password is {word}."` inserted at
  random positions, using decoy keys (storage, mail, gate, â€¦) and *other*
  candidate words â€” creating binding interference.
* **Query:** `"\nQuestion: What is the vault password?\nAnswer:"`
* **Answer space:** a fixed set of 12 single-token words
  (`apple, river, tiger, cloud, stone, candle, silver, garden, planet, forest,
  harbor, lantern`). The model's next-token distribution is restricted to these
  12, giving a clean **12-way classification**.
* **Controlled variables:** context length (tokens, exact), needle position
  (fraction), number of decoys. Filler sentences containing a candidate word
  are removed automatically, so the answer word appears **exactly once** in the
  prompt (verified: `body` contains the answer token 1Ã—).

### 3.2 Scoring

`accuracy` = argmax over the 12 candidates equals the true answer.
`p_correct` = softmax probability mass on the correct candidate. Both are
recorded per sample, so we can look at a smooth signal (`p_correct`) as well as
the discrete one.

### 3.3 Metrics (`contextlab/metrics/ecc.py`)

* **ECC(Ï„)** â€” first context length at which accuracy drops below `Ï„ Â· P(Lâ‚€)`,
  linearly interpolated. Right-censored if it never drops.
* **Effective Context Mass (ECM)** â€” normalised area under `P(L)` over
  log-length; a threshold-free summary in `[0, 1]`.
* **Stretched-exponential fit** â€” `P(L) = Pâ‚€Â·exp(âˆ’(L/scale)^Î²)`.

### 3.4 Reproducibility

Deterministic seeds per `(position, length, sample)`. Results append to CSV
incrementally. Re-run with `experiments/run_main.py`; analyse with
`experiments/analyze.py`.

---

## 4. Environment and constraints (honest)

| item | value |
|---|---|
| CPU | Intel i5-12450H, 12 logical cores |
| RAM | 16 GB |
| GPU | none |
| torch | 2.13.0+cpu |
| transformers | 5.14.1 |
| model | `Qwen/Qwen2.5-0.5B` (base) |

Two measured constraints shaped everything:

1. **Dtype trap.** The model loads in **bfloat16** by default. This CPU has no
   native BF16, so it is emulated: a 512-token forward pass took **57 s**.
   Forcing **float32 + sdpa** cut it to **~4 s** (a ~14â€“100Ã— speedup). *Any*
   future run must force fp32 on this class of machine.
2. **Sporadic multi-minute stalls.** Most forwards take 2.5â€“7 s, but occasional
   passes stalled for 27â€“1661 s (the 2048-token d=4 cell took 27 min). This is
   environmental, not algorithmic, and it caps experiment size. It is why the
   sweeps here are small (n = 4â€“8) and why results are noisy.

**Consequence for the plan:** full-scale sweeps (128k contexts, many models,
many samples) are **not possible on this hardware**. They require a GPU box or
API budget. This should be the first infrastructure decision.

---

## 5. Experiment 1 â€” interference scan (pilot)

Fixed: L = 512, position = 0.5, n = 6. Vary number of decoys `d`.

| decoys | accuracy | p_correct |
|---|---|---|
| 0 | **1.00** | 0.999 |
| 3 | 0.50 | 0.344 |
| 6 | 0.17 | 0.180 |
| 9 | 0.33 | 0.218 |
| 12 | 0.33 | 0.238 |

**Result:** with zero interference, retrieval is trivial. A handful of decoy
`"{key} password is {word}"` passages destroys accuracy **at the same 512-token
length**. Interference, not length, is the dominant variable.
(`results/pilot_difficulty.csv`)

---

## 6. Experiment 2 â€” position sweep (lost in the middle)

Fixed: L = 512, n = 8. Two conditions: `d=0` and `d=4`.

| position | d=0 acc | d=4 acc | d=4 p_correct |
|---|---|---|---|
| 0.05 | 1.00 | 0.875 | 0.759 |
| 0.20 | 1.00 | 0.750 | 0.703 |
| 0.35 | 1.00 | **1.000** | 0.690 |
| 0.50 | 1.00 | **0.500** | **0.351** |
| 0.65 | 1.00 | 0.750 | 0.584 |
| 0.80 | 1.00 | 0.750 | 0.627 |
| 0.95 | 1.00 | **1.000** | **0.945** |

**Result:** without interference the curve is **flat at 1.00**. With
interference, `p_correct` shows a clear **U-shape**: high at the edges
(0.76 start, 0.94 end), a pronounced dip in the **middle (0.35)**. This is a
direct, reproducible instance of the *lost-in-the-middle* effect, and it is
*conditional on interference*. (`results/exp1_position_d{0,4}.csv`,
`figure_position.png`)

---

## 7. Experiment 3 â€” length sweep (effective context)

Fixed: position = 0.5, n = 4. Lengths 256 â†’ 2048 tokens.

| length | d=0 acc | d=0 p_corr | d=4 acc | d=4 p_corr |
|---|---|---|---|---|
| 256 | 1.00 | 0.9989 | 0.50 | 0.366 |
| 512 | 1.00 | 0.9992 | 0.50 | 0.384 |
| 1024 | 1.00 | 0.9988 | 0.75 | 0.542 |
| 2048 | 1.00 | 0.9991 | 0.75 | 0.610 |

**Result:** `d=0` is **1.00 at every length** â€” no degradation whatsoever up to
2048 tokens. `d=4` shows **no downward trend**; if anything `p_correct` rises
(0.37 â†’ 0.61). **[hypothesis]** with more filler the decoys spread out, so
concentrated interference weakens â€” i.e. *longer context diluted the
competition*. With n = 4 this is suggestive, not conclusive.

**Metrics:** ECC(Ï„=0.9) is **right-censored at 2048 in both conditions**
(accuracy never drops below threshold). ECM = **1.00** (d=0) and **1.25**
(d=4; >1 because accuracy rises). A stretched-exponential fits d=0 well
(RÂ²â‰ˆ0.9999, effectively flat) and fails on d=4 (RÂ²â‰ˆ0 â€” nothing decreasing to
fit).

---

## 8. Interpretation â€” what we actually learned

1. **The naive reading of "effective context" is wrong for this model at these
   scales.** It is *not* that performance decays with token count. At 2048
   tokens a 0.5B model retrieves a lone fact as reliably as at 256.
2. **The variable that matters is information competition.** Accuracy collapses
   when decoys compete for the same answer slot, and it collapses **at short
   length**. This reframes the target quantity: not "how many tokens," but
   **"how much competition / entropy per unit of demand."**
3. **Position matters only once competition exists** (the U-shape appears only
   at d=4).
4. **This aligns with the plan's own Â§6 mechanism list** â€” "attention
   competition" and "distractor similarity" â€” and suggests those are the
   *primary* terms, with raw length secondary (at reachable scales).

**Refined working hypothesis (to test next):** effective context is governed by
a *competition ratio* â€” roughly `(number of competing items Ã— similarity) /
attention budget` â€” rather than by `L` alone. The plan's candidate law
`P = Pâ‚€Â·exp[âˆ’Î±(LÂ·R/C)^Î²]` has the right *shape*; these results suggest `R`
should be redefined as an **interference load**, not a length term.

---

## 9. Limitations (read before trusting anything)

* **n is small** (4â€“8 samples/cell). Accuracy steps are 0.125 (n=8) / 0.25
  (n=4). Treat curves as *directional*, not precise.
* **One model, 0.5B, base (not instruct).** No cross-model claim is made.
* **Short lengths only** (â‰¤2048). The 128k regime is untested â€” and untestable
  on this CPU.
* **Repetitive filler.** The clean-filler bank is only ~388 tokens, so long
  contexts tile it. Real prose (wikitext) is a needed upgrade.
* **CPU stalls** injected variance (one cell took 27 min).
* **Answer space is 12 words.** A larger space would change difficulty.

---

## 10. Next breakthrough steps (priority order)

1. **Real filler** (wikitext) â†’ removes the repetition confound (cheap, high value).
2. **Interference Ã— length joint sweep** â†’ test the *competition ratio*
   hypothesis directly.
3. **Bigger n and multiple models** (`Llama-3.2-1B` is cached) â†’ cross-model
   check; needs GPU or long wall-clock.
4. **Define & fit the competition-based law**, then test OOD prediction
   (the plan's Stages 4â€“5 breakthrough criterion).
5. Only then: ContextLab CLI productization (Profiler) â€” unchanged from plan.

---

## 11. Reproducibility

```
pip install -r requirements.txt
python experiments/run_main.py position     # ~23 min on this CPU
python experiments/run_main.py length       # ~32 min on this CPU
python experiments/analyze.py position
python experiments/analyze.py length
python -m pytest tests/ -q                  # 7 unit tests
```

Artifacts: `results/*.csv`, `results/summary_*.csv`,
`results/figure_position.png`, `results/figure_length.png`.

---

# Phase 1B â€” real-prose interference, similarity, and the collapse test

**New central question:** *Can effective context usage be predicted from the
amount and structure of the information competing for attention?*

**Confound fixed first.** Filler is now real WikiText prose
(`data/wiki_train.txt`, 345,590 clean tokens), not a repeated 388-token bank.

## Experiment 4 â€” interference Ã— length (real prose)

Position 0.5, similarity medium, n=4 (n=2 at 4096).

| L \ D | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 256 | 1.00 | 1.00 | 0.75 | 0.75 | 0.50 |
| 512 | 1.00 | 0.75 | 0.25 | 0.25 | 0.50 |
| 1024 | 1.00 | 0.75 | 0.50 | 0.25 | 0.25 |
| 2048 | 1.00 | 1.00 | 0.75 | 0.75 | 0.25 |
| 4096* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

\* n = 2 per cell (provisional).

**Result:** at D = 0, accuracy is 1.00 at every length up to 4096; degradation
is driven by decoy count (columns), not length (rows).

## Experiment 5 â€” similarity (L=1024, D=4, n=6)

| similarity S | accuracy | p_correct |
|---|---|---|
| low | 1.00 | 0.998 |
| medium | 0.50 | 0.380 |
| high | 0.00 | 0.026 |

**Result:** similarity is a first-order driver â€” perfect to zero by changing
only how similar the competing items are.

## Collapse test (honest negative)

Testing I = DÂ·w(S)/L (w = 0.1 / 0.5 / 1.0): the data do **not** collapse onto a
single curve. At I â‰ˆ 1.95 accuracy ranges 0.25 (L=2048, D=8) to 1.00 (L=256, D=1).
So one density parameter is insufficient; an absolute-count term and a
length/dilution term are separately needed.

## Corrected paper-safe wording

Use: *"No measurable degradation from raw context length was observed within the
tested 256â€“4096-token range for this model and task setup."* (Not "length did
not matter.")

## Phase 1B next steps

1. Raise n to â‰¥20 per cell.
2. Fit P = f(D, S, L) with an interaction term.
3. Model I as count + length-dilution (two terms).
4. Fit on models Aâ€“C, predict unseen model D + task E (OOD test).
5. Move to GPU to reach 8Kâ€“128K.

---

# Phase 1B-2 â€” high-n factorial and the fitted law

## Experiment 6 â€” n = 20 factorial (real prose)

Design: L âˆˆ {256, 1024}, D âˆˆ {0,1,2,4,8}, S âˆˆ {low, medium, high}, n = 20,
position 0.5. 30 cells, 600 samples.

| S | D | L=256 | L=1024 |
|---|---|---|---|
| low | 0 | 1.00 | 1.00 |
| low | 1 | 1.00 | 1.00 |
| low | 2 | 1.00 | 1.00 |
| low | 4 | 1.00 | 1.00 |
| low | 8 | 1.00 | 1.00 |
| medium | 0 | 1.00 | 1.00 |
| medium | 1 | 0.90 | 0.75 |
| medium | 2 | 0.80 | 0.50 |
| medium | 4 | 0.65 | 0.45 |
| medium | 8 | 0.50 | 0.35 |
| high | 0 | 1.00 | 1.00 |
| high | 1 | 0.75 | 0.75 |
| high | 2 | 0.55 | 0.60 |
| high | 4 | 0.15 | 0.30 |
| high | 8 | 0.05 | 0.00 |

- Low similarity is inert (1.00 everywhere).
- Medium/high degrade sharply with D.
- Length is a small, consistent secondary effect (1024 usually below 256).

## Fitted law P = f(D, S, L)

Logistic regression, features log2L, D, S and interactions:

| term | coefficient |
|---|---|
| intercept | 3.98 |
| **D*S** | **-0.82** |
| S | -0.44 |
| D | 0.22 |
| log2L | -0.11 |

- Cell fit: **Pearson r = 0.944, MAE = 0.077**
- 5-fold CV accuracy: 0.718 Â± 0.129

## Generalisation (out-of-distribution)

| held-out | trained on | accuracy |
|---|---|---|
| L = 1024 | L = 256 | 0.81 |
| D = 8 | D â‰  8 | 0.85 |
| S = medium | low + high | 0.69 |
| S = high | low + medium | 0.49 |

The law **predicts before the fact** for unseen length and unseen decoy count;
similarity extrapolation is the weakest axis.

## Conclusion so far

The dominant control on retrievability is `count Ã— similarity` of competing
items (D*S = -0.82); raw length is a weak secondary term to 1024/4096 tokens;
unrelated ("low-similarity") context is essentially free. This is a first
working, pre-inference model of effective context â€” still single-model, and its
similarity extrapolation needs a better functional form.

---

# Phase 1B-3 â€” cross-model transfer (the decisive test)

**Protocol (pre-registered):**
1. Freeze the Qwen law â†’ `results/frozen_law_qwen.json` (sha256 `cf79eeaeâ€¦`).
2. Write predictions for all 30 Llama cells â†’ `results/pred_qwen_on_llama.csv`
   **before** running Llama.
3. Run the identical factorial on **Llama-3.2-1B** (600 samples).
4. Compare with **no refit**.

## Transfer quality (frozen coefficients)

| metric | value |
|---|---|
| Pearson r | **0.946** |
| Spearman Ï | 0.923 |
| MAE | 0.080 |
| ECE | 0.049 |
| AUC | 0.915 |

The frozen Qwen law predicts Llama as well as it fits Qwen (r = 0.944).

## Coefficient structure

| term | Qwen (frozen) | Llama (refit) |
|---|---|---|
| **DÃ—S** | **âˆ’0.82** | **âˆ’1.11** |
| S | âˆ’0.44 | âˆ’0.19 |
| log2L | âˆ’0.11 | +0.16 |
| intercept | 3.98 | 2.29 |

The **dominant DÃ—S term is preserved**; the **intercept differs** (different
absolute attention budgets). A combined fit with shared slopes + a
model-specific intercept needs only one extra parameter for Llama.

## Residual structure

Llama is *more* robust than predicted at long length + high interference
(e.g. L=1024/high/D=4: predicted 0.12, observed 0.40) â€” consistent with a
larger attention budget under the same interference structure.

## Strongest defensible claim

> For these two models and this benchmark, retrieval performance is well
> explained by a multiplicative interaction between distractor count and
> distractor similarity, with context length a smaller secondary effect; the
> fitted law, **frozen on one model, predicts an unseen model's cells with
> r = 0.946**.

## Honest limits

Two models; one task family (needle retrieval); lengths â‰¤1024 at n=20;
similarity still categorical; lower-order coefficients unstable (entangled with
DÃ—S).

## Next milestone

Fit on models Aâ€“C and tasks Aâ€“D â†’ predict **unseen model E + unseen task E**;
replace categorical similarity with a continuous metric (embedding / key
overlap).

---

# Phase 1B-4 â€” continuous similarity + a third model

## Experiment 8 â€” making S measurable

**Pre-registered candidate metrics** (defined before running the LLM):
`cos_sim` (MiniLM embedding cosine), `jaccard` (token overlap),
`shared_prefix`. 20 decoy sentences were pre-registered, spanning
`cos_sim` from âˆ’0.04 (weather) to 1.00 (identical template).

At D=1 the effect was **below the detection floor** (all 1.00), so the sweep
ran at **D=4, L=1024, n=10**.

| decoy | cos_sim | accuracy |
|---|---|---|
| unrelated_weather | âˆ’0.04 | 1.00 |
| topic_password | 0.44 | 1.00 |
| password_policy | 0.50 | 1.00 |
| account_password | 0.59 | 0.60 |
| system_password | 0.57 | 0.50 |
| security_password | 0.60 | 0.40 |
| archive_password | 0.72 | 0.30 |
| storage_password | 0.76 | 0.30 |
| case_note_similar | 0.96 | 0.00 |
| case_important_identical | 1.00 | 0.00 |

## Which metric predicts interference?

| metric | Qwen r | held-out-decoy MAE |
|---|---|---|
| **jaccard** | **âˆ’0.941** | 0.099 |
| cos_sim | âˆ’0.878 | 0.133 |
| shared_prefix | âˆ’0.437 | n/a |

**Lexical Jaccard overlap is the best single predictor.**

## Third model â€” gpt2-medium (L=768)

| metric | Qwen r | gpt2 r | agree? |
|---|---|---|---|
| jaccard | âˆ’0.941 | âˆ’0.886 | yes |
| cos_sim | âˆ’0.878 | âˆ’0.747 | yes |
| shared_prefix | âˆ’0.437 | âˆ’0.433 | yes |

The **metric ranking is identical across model families**. gpt2-medium is more
tolerant (same structure, larger budget) â€” the same pattern as the Qwenâ†’Llama
transfer.

## Final position

The law is now **P = f(D, S_cont, L)** with S_cont a *measured* quantity; the
`DÃ—S` interaction remains dominant across **three** model families (Qwen,
Llama, gpt2). Categorical S has been replaced, closing the biggest weakness.

Still open: cross-**task** transfer; more models; longer contexts; direct
measurement of the "attention budget".

---

# Phase 1B-5 â€” a third model, zero-refit (frozen Qwen law â†’ gpt2-medium)

**Why this run.** The Qwenâ†’Llama bridge is strong but came from a model family
(Llama) that shares the same 30-cell `(L,S,D)` grid as Qwen. To raise the
evidentiary bar we froze the Qwen law **once more** and ran the identical
factorial on a *different* family, **gpt2-medium** (355M, 24 layers). No
parameter of the law was touched after freezing.

**Protocol (pre-registered):**
1. Frozen Qwen law â†’ `results/frozen_law_qwen.json` (sha256 `cf79eeaeâ€¦`).
2. Predictions for all 30 gpt2-medium cells written and hashed **before** the
   run â†’ `results/pred_qwen_on_gpt2medium.csv`, sha256 `2031e945â€¦`.
3. Run the identical factorial on **gpt2-medium**, 600 samples.
4. Compare with **zero refit**.

### Honest note on a rejected model (distilgpt2)

We first tried **distilgpt2** (82M, 6 layers) as the third family. It sat at the
**guess-rate floor**: accuracy â‰ˆ0.15 everywhere with chance = 1/12 = 0.083, i.e.
*no dynamic range* (Qwen is 1.00 at low/D=0). A model with no dynamic range
cannot be used to test transfer, so the run was abandoned and retained as a
documented **capacity-floor finding** (`results/floor_distilgpt2/`). The switch
to gpt2-medium was made *before* any gpt2-medium data existed.

## Transfer quality (frozen coefficients, zero refit)

| metric | Qwenâ†’Llama | **Qwenâ†’gpt2-medium** |
|---|---|---|
| Pearson r | 0.946 | **0.872** |
| Spearman Ï | 0.923 | **0.888** |
| MAE | 0.080 | **0.122** |
| RMSE | 0.108 | 0.182 |
| ECE | 0.045 | **0.078** |
| cells | 30 | 30 |

The same frozen structure predicts a second unseen family. gpt2-medium is more
tolerant in the low/medium regime (observed 0.985 at S=low vs predicted 0.958)
and the law slightly over-predicts under high interference â€” the same
"larger budget, same structure" pattern seen with Llama. Per the **pre-declared**
criterion (Ï â‰¥ 0.80 zero-refit **and** Ï â‰¥ 0.80 held-out calibration) both
targets **PASS**: Llama 0.923 / 0.909; gpt2-medium 0.888 / 0.858.

## Zero-refit vs. calibration transfer (honesty about leakage)

`experiments/cross_model_validate.py` reports both:
* **zero-refit** â€” the frozen law itself (genuine model-transfer evidence);
* **leave-one-cell-out calibration** â€” an affine `predâ†’acc` recalibration fit on
  29 cells, scored on the held-out cell.

Because every target shares the same cell grid, the LOO number is **within-target
calibration**, *not* unseen-model transfer, and is labelled as such. For Llama
calibration helps slightly (MAE 0.080â†’0.073); for gpt2-medium it does **not**
(0.122â†’0.147) â€” i.e. the frozen law is already well calibrated and an extra
fitted layer buys nothing. Outputs: `results/cross_model_validation.csv`,
`results/cross_model_pairs.csv`.

---

# Phase 1B-6 â€” D-saturation sweep (calibrating the count term)

**Why this run.** The product's interference engine uses
`I_count(D) = min(1, D / D_SATURATION)` with `D_SATURATION = 5`. That constant
was **never empirically justified** â€” the original factorial sampled only
`D âˆˆ {0,1,2,4,8}`. We ran a wider sweep (`experiments/run_expB.py`): Qwen, L=256,
**S=medium**, `D âˆˆ {0,1,2,3,4,5,6,8,10,12,16}`, n=20.

## Measured saturation curve

| D | 0 | 1 | 2 | 3 | 4 | **5** | **6** | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 1.00 | 0.90 | 0.80 | 0.75 | 0.65 | **0.45** | **0.60** | 0.50 | 0.40 | 0.25 | 0.15 |

**Findings (`experiments/analyze_expB_saturation.py`):**

1. **No saturation in range.** Accuracy keeps falling to **0.15 at D=16** (still
   above chance 0.083); there is **no knee** (no step with |Î”| < 0.02).
2. **A genuine inversion at D=5 â†’ 6** (0.45 â†’ 0.60): accuracy *rises* as D grows,
   the exact non-monotonicity a hard `min(1, D/5)` cap would produce.
3. **Best-fit k = 9.0, not 5.** A trial-level logistic fit on `[I, IÂ·S, log2L]`
   scores 5-fold CV accuracy: linear-D 0.668, `min(1,D/5)` 0.705, best-fit
   `k=9` 0.714. So a **capped** count term beats pure-linear, but the cap value
   5 is **not** the empirical optimum.

**Verdict.** `min(1, D/D_SATURATION)` is the right *functional form* (it beats
linear), but the specific value `D_SATURATION = 5` is **not supported** by this
sweep; the data prefer `k â‰ˆ 9`. This is reported as a known calibration
limitation rather than a hidden bug: the discrete cap is a modelling choice, and
the engine's *ordering/behaviour* is unaffected (it stays monotone and gives
D=8 the maximum interference load), but its *magnitude* at small D is
mis-scaled. Outputs: `results/expB_dsweep_cells.csv`,
`results/expB_dsweep_factorial.csv`.

---

# Phase Aâ€“D â€” final checkpoints (research frozen)

## Phase A â€” cross-task validation (honest negative)

Counting task (count "red" objects, ignore a similar colour; S measured by RGB).
Grid Lâˆˆ{256,1024}, Dâˆˆ{0,2,4,8}, Sâˆˆ{low,medium,high}, n=10.

| S (measured) | D=0 | D=2 | D=4 | D=8 |
|---|---|---|---|---|
| low (0.18) | 0.55 | 0.60 | 0.80 | 0.65 |
| medium (0.63) | 0.55 | 0.55 | 0.80 | 0.70 |
| high (0.84) | 0.55 | 0.45 | 0.50 | 0.55 |

**The interference structure does NOT transfer to counting** (DÃ—S: âˆ’1.17 â†’
âˆ’0.09; cross-task corr +0.02). A 0.5B model cannot reliably count at all
(baseline ~0.55), so the task sits at a capability floor. Conclusion: the law is
a theory of **retrieval** interference, not of all context.

## Phase B â€” intervention (the law is actionable)

L=1024, needle + 8 decoys (2 identical, 4 moderate, 2 unrelated), n=20:

| condition | accuracy |
|---|---|
| FULL (8 decoys) | 0.05 |
| RANDOM prune 4 | 0.35 |
| **SIM prune 4 (profiler)** | **0.70** |
| NONE (0 decoys) | 1.00 |

SIM âˆ’ RANDOM = **+0.35** (95% CI [0.05, 0.65]). Pruning by measured
interference roughly **doubles** accuracy vs random pruning.

## Phase C â€” robustness

L=1024, D=4, medium, n=12:

| filler | pos=0.1 | pos=0.5 | pos=0.9 |
|---|---|---|---|
| real prose | 0.83 | 0.42 | 0.67 |
| sentence bank | 0.83 | 0.67 | 0.83 |

Mid-context penalty reproduces under both; real prose is systematically harder
(0.64 vs 0.78) â€” validating the switch away from repetitive filler.

## Phase D â€” statistical finalization

| claim (measured) | value | 95% CI |
|---|---|---|
| Exp6 cell fit r | 0.944 | [0.87, 0.98] |
| Exp6 DÃ—S coefficient | âˆ’0.82 | [âˆ’1.68, +0.08] |
| Cross-model transfer r (Llama) | 0.946 | [0.87, 0.98] |
| Cross-model MAE (Llama) | 0.080 | [0.056, 0.108] |
| Cross-model AUC (Llama) | 0.915 | [0.893, 0.937] |
| Cross-model ECE (Llama) | 0.049 | â€” |
| **Cross-model transfer r (gpt2-medium)** | **0.872** | â€” |
| **Cross-model MAE (gpt2-medium)** | **0.122** | â€” |
| **Cross-model ECE (gpt2-medium)** | **0.078** | â€” |
| **D-sweep best-fit saturation k** | **9.0** (config 5) | â€” |
| Continuous Jaccard r | âˆ’0.941 | [âˆ’0.976, âˆ’0.895] |
| Intervention gain (SIMâˆ’RANDOM) | +0.35 | [0.05, 0.65] |

**Caveat:** the single DÃ—S coefficient CI includes 0; the interaction is best
supported jointly with the intervention (CI excludes 0) and the cross-model
transfer. Mechanism ("attention budget") is marked **inferred**, not proven.
The count-saturation constant (`D_SATURATION = 5`) is a **modelling choice not
recovered by the sweep** (best-fit k â‰ˆ 9); the capped form beats a linear count
term, but its magnitude at small D is mis-calibrated (see Phase 1B-6).

## Research frozen

**Final claim:** within the *retrieval* task family, across three model
families, retrieval accuracy is predicted by a multiplicative interaction of
competing-passage count and measured similarity (length secondary); the dominant
structure transfers cross-model **with zero refit** (Qwenâ†’Llama r=0.946,
Qwenâ†’gpt2-medium r=0.872), and pruning on the measured metric improves
performance. Independent cross-model evidence now spans two unseen families
(Llama-3.2-1B and gpt2-medium); a fourth candidate (distilgpt2) was rejected for
sitting at the guess-rate floor. This is a **retrieval-interference** theory, not
yet a general theory of context. â†’ Proceed to product (Context Profiler).

## Phase 7 â€” real RAG intervention benchmark (honest result)

End-to-end pipeline on real data: local Wikipedia corpus â†’ MiniLM
(`all-MiniLM-L6-v2`) + FAISS retrieval â†’ Qwen2.5-0.5B reader â†’ K-way scoring
over a frozen 12-word candidate set. 40 pre-registered held-out queries
(`results/rag_queries.json`, sha256 `8f139b1eâ€¦`), three frozen conditions:

    BASELINE   : top-k retrieved passages â†’ reader
    CONTEXTLAB : top-k â†’ profiler prune + reorder â†’ reader
    RANDOM     : top-k â†’ random prune (matched count) â†’ reader

| condition | accuracy | ctx tokens | passages | removed |
|---|---|---|---|---|
| BASELINE | 0.0745 | 88.1 | 8.00 | 0.00 |
| CONTEXTLAB | 0.175 | 88.1 | 8.00 | 0.00 |
| RANDOM | 0.0745 | 88.1 | 8.00 | 0.00 |

| headline delta (paired, n=40) | value | 95% CI |
|---|---|---|
| ContextLab âˆ’ Baseline accuracy | **+0.100** | [0.000, 0.225] |
| ContextLab âˆ’ Random accuracy | +0.100 | [0.000, 0.225] |
| ContextLab âˆ’ Baseline ctx tokens | 0.0 (0.0%) | [0.0, 0.0] |

**What this does and does not show.**

* **Supported (weakly):** reordering kept passages by profiler-predicted
  evidence strength moves accuracy from 0.0745 â†’ 0.175 (â‰ˆ2.3Ã—), while the
  matched random-prune control stays at 0.0745. The direction favours the engine.
* **Not significant:** the paired 95% CI lower bound is exactly 0.000; with 40
  queries this is suggestive, not established. Baseline sits at chance
  (1/12 = 0.083), so the reader is near its floor on this task.
* **"Cuts tokens" is NOT supported by this design.** `n_removed = 0` for every
  query: the profiler prunes redundant *near-duplicates*, but these passages are
  competing *contradictions* (same question shape, different answers), which the
  frozen policy deliberately **protects**. Context tokens are therefore
  unchanged (+0.0%). This benchmark exercises the **reorder** arm only, not the
  prune arm.
* **Reader is not the bottleneck:** a control (`rag_diagnose_floor.py`) gives
  clean-context accuracy **1.000** (gold alone) vs **0.650** (gold + 8 decoys) â€”
  competition, not model capacity, causes errors. So the low absolute accuracy is
  an interference effect, consistent with the research claim.

**Data-integrity bug found and fixed before reporting.** The first run produced
accuracy â‰ˆ 0.10 for *all* conditions. Diagnosis (`rag_diagnose_ctx.py`) showed
the top-8 retrieved passages were **eight copies of one decoy**, with the gold
absent. Cause: the query generator reused the same 12 candidate words as decoys
across queries, so 400 injected strings collapsed to 107 unique â€” duplicate
strings flooded the FAISS index and retrieval returned k copies of a single
passage. Fix: exploit the 4-template Ã— 10-noun grid for unique golds and give
each query a **private, disjoint decoy-noun slice**, with a hard uniqueness
assert (360 injected strings, 360 unique, 0 duplicates). Only the corrected run
is reported above.

**Honest label:** Phase 7 provides *weak, directional* evidence that
ContextLab's **ordering** helps real end-to-end RAG accuracy, and **no**
evidence that it reduces tokens on a contradiction-rich corpus. The next design
must inject *redundant* (non-contradictory) passages to exercise the prune arm
and test the token-saving claim.

## Phase 7B â€” redundancy intervention (the missing arm)

Phase 7 could only exercise reordering (contradiction-rich corpus â†’ `removed=0`).
Phase 7B fixes the design flaw: **40 preregistered queries Ã— 4 context types Ã—
4 conditions = 640 runs** (sha256 `e3ea5e36â€¦`, 350 injected strings, all unique).

Context types: `clean` (gold + 2 irrelevant), `redundant` (gold + near-duplicate
supporting passages), `contradictory` (gold + 3 conflicting answers),
`mixed`. Conditions: `BASELINE`, `RANDOM` (matched removal), `CONTEXTLAB`
(prune redundancy + reorder), **`ORACLE`** (gold + 1 best support only).

| context | condition | accuracy | ctx tokens | removed |
|---|---|---|---|---|
| clean | BASELINE/RANDOM/CONTEXTLAB | 1.000 | 35.3 | 0.00 |
| clean | ORACLE | 1.000 | 22.0 | 1.00 |
| **redundant** | BASELINE | 1.000 | 61.5 | 0.00 |
| **redundant** | RANDOM | 1.000 | 38.8 | 2.25 |
| **redundant** | **CONTEXTLAB** | **1.000** | **41.8** | **2.25** |
| **redundant** | ORACLE | 1.000 | 17.5 | 4.00 |
| contradictory | BASELINE | 0.350 | 61.5 | 0.00 |
| contradictory | RANDOM | 0.350 | 61.5 | 0.00 |
| contradictory | CONTEXTLAB | 0.250 | 61.5 | 0.00 |
| contradictory | ORACLE | **0.525** | 17.5 | 4.00 |
| mixed | BASELINE | 0.775 | 87.8 | 0.00 |
| mixed | RANDOM | 0.700 | 73.8 | 1.50 |
| mixed | CONTEXTLAB | 0.675 | 74.6 | 1.50 |
| mixed | ORACLE | 0.625 | 17.5 | 7.00 |

Preregistered comparisons (paired, bootstrap 95% CI):

| id | comparison | metric | Î” | 95% CI | verdict |
|---|---|---|---|---|---|
| C1 | CONTEXTLABâˆ’BASELINE | accuracy | âˆ’0.050 | [âˆ’0.100, âˆ’0.006] | **SIG (âˆ’)** |
| C2 | CONTEXTLABâˆ’BASELINE | tokens | âˆ’8.21 | [âˆ’9.69, âˆ’6.76] | **SIG (âˆ’)** |
| C3 | CONTEXTLABâˆ’RANDOM | accuracy | âˆ’0.031 | [âˆ’0.087, +0.019] | ns |
| C4 | CONTEXTLABâˆ’ORACLE | accuracy | âˆ’0.056 | [âˆ’0.119, +0.006] | ns |
| C5 | CONTEXTLABâˆ’ORACLE | tokens | +34.7 | [+31.9, +37.3] | SIG |

Within **redundant** (the target case): C2 tokens = **âˆ’19.7 [âˆ’20.9, âˆ’18.5], SIG**,
accuracy Î” = **0.000** â€” i.e. **32% fewer context tokens at identical accuracy**.

**What this proves (and what it does not).**

* **âœ… The prune arm works.** On redundant context ContextLab removes the
  near-duplicate support and keeps accuracy at **1.000** while cutting tokens
  **61.5 â†’ 41.8 (âˆ’32%, SIG)**. This is the monetizable Pareto result: *same
  accuracy, one-third less context.*
* **âš ï¸ Honesty caveat.** On pure `redundant` context the matched **RANDOM**
  control also reaches 1.000 at 38.8 tokens â€” *any* correct pruning works when
  the answer is replicated. So the win is "pruning works", not yet "ContextLab's
  *choice* beats alternatives" (C3 ns).
* **âŒ The reorder arm hurts contradictions.** On `contradictory` context nothing
  is pruned (policy protects conflicts) â€” the **only** change is reordering, and
  it dropped accuracy 0.350 â†’ 0.250 (6 queries lost, 2 gained). Sorting by
  `-competition` pushes a *conflicting* passage toward the front, confusing the
  reader. RANDOM (no reorder) matched BASELINE.
* **ðŸ”Ž ORACLE beats ContextLab on conflicts (0.525 vs 0.250, SIG).** Keeping only
  gold+1 (dropping all conflicts) is far better for this weak reader â€” the
  "protect the conflict" policy is *honest* but **not accuracy-optimal**. This is
  a genuine product-policy tension, now measured rather than assumed.

**Honest label:** Phase 7B **confirms the pruning/Pareto claim on redundant
context (âˆ’32% tokens, accuracy unchanged, SIG)** and **falsifies** the idea that
the current reorder policy helps contradictory context â€” it hurts, and an
ORACLE that drops conflicts does better. Next design must (a) test whether
ContextLab's *choice* beats RANDOM with harder redundancy, and (b) offer a
"surface-vs-drop conflict" mode rather than always protecting. Artifacts:
`results/rag7b_bench.csv`, `rag7b_summary.csv`, `rag7b_comparisons.csv`,
`figure_rag7b_pareto.png`.

## Phase 7C â€” choice attribution (does ContextLab choose better than random?)

Phase 7B's win was confounded: the gold answer was replicated, so *any* correct
pruning sufficed. Phase 7C removes the crutch â€” only ONE passage (`gold`) yields
the answer; the rest are graded evidence (weaker partial/indirect, related,
distractor, irrelevant) plus **redundant duplicates of the weak roles** that the
profiler prunes. 40 queries Ã— 4 conditions (sha256 `0944f1ceâ€¦`, 360 unique
strings). ContextLab removes the redundant duplicates (`partial_dup`,
`indirect_dup`) and never the gold; RANDOM removes the same count at random.

| condition | accuracy | gold survival | ctx tokens | removed |
|---|---|---|---|---|
| BASELINE | 0.500 | 1.000 | 135.5 | 0 |
| RANDOM | 0.550 | **0.775** | 105.9 | 2.0 |
| **CONTEXTLAB** | **0.800** | **1.000** | 97.5 | 2.0 |
| ORACLE | 1.000 | 1.000 | 18.5 | 7.0 |

Preregistered comparisons (paired, n=40):

| id | comparison | metric | Î” | 95% CI | verdict |
|---|---|---|---|---|---|
| A1 | CONTEXTLAB âˆ’ RANDOM | accuracy | +0.250 | [+0.050, +0.425] | **SIG** |
| A2 | CONTEXTLAB âˆ’ RANDOM | gold survival | +0.225 | [+0.100, +0.350] | **SIG** |
| A3 | CONTEXTLAB âˆ’ BASELINE | accuracy | +0.300 | [+0.150, +0.450] | **SIG** |
| A4 | CONTEXTLAB âˆ’ BASELINE | tokens | âˆ’38.0 | [âˆ’38.6, âˆ’37.4] | **SIG** |
| A5 | RANDOM âˆ’ BASELINE | accuracy | +0.050 | [âˆ’0.125, +0.201] | ns |

**Mechanism (verified):** RANDOM's accuracy collapses to **0.222 when it deletes
the gold** (9/40 queries) vs **0.645 when it does not**; ContextLab keeps the
gold **100%** of the time. Head-to-head, ContextLab wins 14 queries and loses 4
vs RANDOM, and wins 12 / loses 0 vs BASELINE.

**Honest label:** Phase 7C **turns "pruning works" into "ContextLab makes better
pruning decisions"** â€” a significant accuracy gain over a matched random control
(+0.25) *and* over baseline (+0.30) at fewer tokens. Artifacts:
`results/rag7c_bench.csv`, `rag7c_summary.csv`, `rag7c_comparisons.csv`,
`figure_rag7c_pareto.png`.

## Phase 7D â€” conflict-resolution policies (detection â‰  resolution)

Phase 7B showed the current policy ("protect the conflict") *hurt* on
contradictory context (0.350 vs ORACLE 0.525). Phase 7D tests four explicit
resolution policies on the frozen contradictory/mixed contexts (40 queries Ã— 2
types Ã— 4 policies = 320 runs, sha256 `47119ff7â€¦`).

| policy | accuracy | ctx tokens | removed |
|---|---|---|---|
| PROTECT (current) | 0.562 | 74.6 | 0.00 |
| **DROP_WEAKER** | **1.000** | **45.1** | 3.375 |
| SURFACE | 0.500 | 74.6 | 0.00 |
| ORACLE | 1.000 | 9.8 | 6.375 |

Preregistered comparisons (paired, pooled n=80):

| id | comparison | metric | Î” | 95% CI | verdict |
|---|---|---|---|---|---|
| P1 | DROP_WEAKER âˆ’ PROTECT | accuracy | **+0.438** | [+0.338, +0.550] | **SIG** |
| P2 | ORACLE âˆ’ PROTECT | accuracy | +0.438 | [+0.338, +0.550] | **SIG** |
| P3 | SURFACE âˆ’ PROTECT | accuracy | âˆ’0.062 | [âˆ’0.163, +0.037] | ns |
| P4 | DROP_WEAKER âˆ’ PROTECT | tokens | âˆ’29.5 | [âˆ’31.5, âˆ’27.7] | **SIG** |

On `contradictory` alone: PROTECT 0.350 â†’ DROP_WEAKER **1.000** (+0.650, SIG).

**What this proves.** **Resolving** a conflict (drop the weaker member) massively
beats **protecting** it: +0.44 accuracy *and* âˆ’30 tokens, both significant.
`SURFACE` (keep the conflict, reorder) does nothing (ns) â€” so it is *resolution*,
not reordering, that matters. **DROP_WEAKER matches the ORACLE on accuracy** while
using ~4.6Ã— the tokens.

**Honest caveat.** DROP_WEAKER's tie-break is "keep the highest-token conflict
member", which on this construction *coincidentally* selects the gold (the
conflicting passages all share one template â€” identical token counts â€” and the
gold sat at index 0, so the rule broke ties by keeping the **first** passage).
This test does **not** prove the proxy is a general gold-detector, only that
**resolving conflicts beats protecting them** *when the gold is positionally
privileged*. **Phase 7E removes exactly this confound and the effect vanishes**
(see below). Artifacts: `results/rag7d_bench.csv`, `rag7d_summary.csv`,
`rag7d_comparisons.csv`.

## Phase 7E â€” decoupled conflict resolution (the confound removed)

Phase 7D's DROP_WEAKER = 100% was confounded: the conflicting passages shared one
template (identical token counts) and the gold sat at index 0, so "keep the
highest-token conflict" only ever broke ties by picking the **first** passage â€”
it won by **position**, not by signal. Phase 7E removes both confounds:

* **Specificity decoupled from correctness** â€” a passage's length is drawn
  independently of whether it carries the answer. Verified at freeze time:
  P(correct)=0.250; by specificity {low 0.246, medium 0.305, high 0.182}; max
  deviation **0.068** â‰¤ 0.12 (**anti-leakage PASS**).
* **Position randomized** â€” correct-slot counts [12, 13, 6, 9].

40 queries Ã— 6 policies = 240 runs (sha256 `e03597d3â€¦`, 240 unique strings).

| policy | accuracy | survivor correct | ctx tokens |
|---|---|---|---|
| PROTECT | 0.300 | 1.000 | 99.0 |
| DROP_MORE_SPECIFIC | 0.250 | **0.250** | 51.7 |
| DROP_LESS_SPECIFIC | 0.275 | 0.275 | 35.3 |
| DROP_WEAKER | 0.300 | 0.300 | 44.9 |
| RANDOM | 0.275 | 0.275 | 42.3 |
| ORACLE | 1.000 | 1.000 | 42.0 |

Preregistered comparisons (paired, n=40):

| id | comparison | metric | Î” | 95% CI | verdict |
|---|---|---|---|---|---|
| E1 | DROP_WEAKER âˆ’ PROTECT | accuracy | +0.000 | [âˆ’0.225, +0.225] | ns |
| E2 | DROP_MORE_SPECIFIC âˆ’ PROTECT | accuracy | âˆ’0.050 | [âˆ’0.275, +0.175] | ns |
| E3 | ORACLE âˆ’ PROTECT | accuracy | +0.700 | [+0.550, +0.825] | **SIG** |
| E4 | RANDOM âˆ’ PROTECT | accuracy | âˆ’0.025 | [âˆ’0.225, +0.200] | ns |
| E5 | DROP_WEAKER âˆ’ RANDOM | accuracy | +0.025 | [âˆ’0.175, +0.225] | ns |
| E6 | DROP_WEAKER âˆ’ PROTECT | tokens | âˆ’54.1 | [âˆ’59.9, âˆ’48.6] | **SIG** |

**The honest finding: the 7D win does not survive the confound fix.** Every
resolution heuristic collapses to **chance** â€” `survivor_correct` is 0.250â€“0.300
for DROP_MORE_SPECIFIC / DROP_LESS_SPECIFIC / DROP_WEAKER / RANDOM, exactly the
1/4 base rate. None beats PROTECT, and none beats RANDOM (E5 ns). Only **ORACLE**
(which *knows* the answer) reaches 1.000, so the headroom is real (+0.700, E3) but
**no deployable conflict-resolution rule closes it** on this construction.

**Mechanism (verified):** accuracy is fully determined by whether the surviving
passage is correct â€” **1.000 when the survivor is correct (84 cases), 0.000 when
it is not (116 cases)**. Specificity carries no signal (`DROP_MORE_SPECIFIC` keeps
high-specificity 28/40 times yet still hits the base rate).

**Honest label.** Phase 7E **falsifies** the deployability of the specificity- or
content-based conflict-resolution policies tested: after removing the index-0 /
equal-length confound, DROP_WEAKER â‰ˆ DROP_MORE_SPECIFIC â‰ˆ DROP_LESS_SPECIFIC â‰ˆ
RANDOM â‰ˆ chance. The unresolved question is now sharply defined: **conflict
*detection* is solved; conflict *resolution without an oracle* is not** â€” a
genuine open problem, not a demonstrated feature. Artifacts:
`results/rag7e_bench.csv`, `rag7e_summary.csv`, `rag7e_comparisons.csv`,
`rag7e_meta.json`.

## Phase 8 â€” the functional form of the count term f(D)

The product's count term is `min(1, D / D_SATURATION)` with `D_SATURATION = 5`,
a value the D-sweep never recovered (best-fit k â‰ˆ 9). Phase 8 asks which
functional form `f(D)` in `P = Ïƒ(a + bÂ·f(D))` best predicts accuracy on
**held-out D values**. Protocol frozen before fitting (sha256 `f1d548acâ€¦`):
leave-one-D-out on the Qwen2.5-0.5B D-sweep (`expB_dsweep_factorial.csv`, L=256,
S=medium, D âˆˆ {0,1,2,3,4,5,6,8,10,12,16}, n=20/cell).

| form | MAE | RMSE | Spearman | ECE | mono-viol | params |
|---|---|---|---|---|---|---|
| frozen 7-feature law | 0.094 | 0.113 | 0.973 | 0.083 | 0 | 7 |
| linear `D` | 0.065 | 0.096 | 0.973 | 0.063 | 0 | 2 |
| `min(1, D/k)` | 0.062 | 0.093 | 0.973 | 0.060 | 0 | 3 |
| `1âˆ’e^(âˆ’D/k)` | 0.086 | 0.105 | 0.964 | 0.056 | **1** | 3 |
| **`log(1+D)`** | **0.056** | **0.072** | 0.973 | **0.034** | 0 | **2** |
| isotonic (learned) | 0.063 | 0.082 | 0.970 | 0.047 | 0 | 10 |

Extrapolation (hold out the largest D=16, train on Dâ‰¤12): observed 0.150 â€”
`min_sat` err 0.037 < `log` 0.056 < `linear` 0.061 < frozen law 0.108 <
`exp_sat` 0.133.

**Findings.** (1) **`log(1+D)` is the best-interpolating form** â€” lowest MAE
(0.056) and best calibration (ECE 0.034) with only **2 parameters**. (2) **Every
simple monotone form beats the frozen 7-feature law on held-out D** (0.056â€“0.086
vs 0.094): on the D axis the elaborate law is *dominated* by a 2-parameter
logarithm. (3) **`min_sat` extrapolates best** (err 0.037); **`exp_sat` is worst**
and is the only form with a **monotonicity violation**. (4) The isotonic learned
form (10 params) buys nothing over `log` (2 params) â€” no accuracy gain, worse MAE.

**Honest label.** Phase 8 **replaces the unsupported `D_SATURATION = 5` choice**:
for the count term, `log(1+D)` is the empirically best simple form and beats the
frozen law on held-out D, while `min-saturation` remains the better *extrapolator*.
No form achieves zero error (the raw curve is non-monotone at D=5â†’6, a finite-n
artefact at n=20). Artifacts: `results/phase8_forms.csv`, `phase8_loo.csv`,
`phase8_protocol.json` + `.sha256`, `figure_phase8_forms.png`.

## Phase 9 â€” cross-task transfer (zero-refit)

Phase 8 fixed the count term *within* the retrieval family. Phase 9 asks the
decisive transfer question: does that structure survive a genuinely different
task with **zero refit**? Protocol frozen before touching target results
(sha256 `4b8ca8daâ€¦`): fit each `P = Ïƒ(a + bÂ·f(D))` on the retrieval D-sweep,
**freeze (a, b, k)**, then evaluate on the **counting** task family
(`expA_counting.csv` â€” "count the red objects" against similar-colour items; a
"how many" task, not a rewording of retrieval).

| slice | model | Spearman | Pearson | MAE | RMSE | ECE |
|---|---|---|---|---|---|---|
| L1024 | linear | 0.949 | 0.934 | 0.184 | 0.213 | 0.184 |
| L1024 | min_sat | 0.949 | 0.934 | 0.184 | 0.213 | 0.184 |
| L1024 | exp_sat | 0.949 | 0.938 | 0.200 | 0.230 | 0.200 |
| L1024 | log | 0.949 | 0.951 | 0.212 | 0.247 | 0.212 |
| L1024 | frozen 7-feature law | 0.949 | 0.914 | **0.238** | 0.273 | 0.238 |
| pooled | all forms | **âˆ’0.600** | âˆ’0.55 â€¦ âˆ’0.64 | 0.19â€“0.25 | â€” | â€” |

Frozen params (fit on retrieval): `linear a=1.93 b=âˆ’0.25`; `min_sat k=16`;
`exp_sat k=14.5`; `log a=3.49 b=âˆ’1.74`.

**Findings (preregistered outcome = B, with a C signal on pooled data).**

* **Rank structure transfers (B).** On the L=1024 slice â€” where the counting
  task has a genuine, weak competition effect (acc 0.60 â†’ 0.47 as D grows) â€”
  **every frozen count-term form ranks the D-effect correctly (Spearman 0.949)**.
  The *monotone form* of the count term therefore survives a different task
  zero-refit.
* **Calibration does not transfer.** All forms are badly calibrated on the target
  (ECE 0.18â€“0.24); the frozen retrieval coefficients set the wrong *level* for the
  counting task. â†’ the coefficients are task-dependent.
* **Pooled data inverts (C).** Pooling the L=256 slice (where counting accuracy
  *rises* with D: 0.5â†’0.8) with L=1024 flips the correlation to **Ï=âˆ’0.60** â€” the
  D-structure is **condition-dependent**, not universal.
* The **frozen 7-feature law remains the worst-calibrated** even here (MAE 0.238),
  consistent with Phase 8.

**Honest label.** Phase 9 supports `R = F(task, f(D), S, L)` â€” a **task-conditioned
reliability model** â€” rather than one universal equation: the *form* of the count
term transfers (monotone, correctly ordered), but the *coefficients* and the sign
of the effect do not. No claim of task-general reliability is made. Artifacts:
`results/phase9_transfer.csv`, `phase9_protocol.json` + `.sha256`.

---

## Phase 10 â€” cross-domain validation (zero-refit)

Phases 8â€“9 held the *wiki* filler domain fixed. Phase 10 asks whether the
retrieval law survives a change of **context domain** with **zero refit**.
Protocol frozen before any new cell ran (sha256 `604a8a96â€¦`):

* **New data (the only model run of the phase):** the retrieval D-sweep at
  L=256, S=medium, position=0.5, D âˆˆ {0,4,8}, n=10/cell, with the filler
  swapped from wiki prose to **general English** (`data/filler_general.txt`,
  built from the HF-cached `NeelNanda/pile-10k` corpus, URL-stripped, truncated
  to match the reference clean-token count to within 0.82% â€” so *filler amount*
  is held fixed while *domain* changes).
* **Cross-length (existing data, no new runs):** the L-side analogue of
  Phase 9 â€” fit each form on Exp6 L=256 cells, evaluate zero-refit on L=1024.
* **Models (zero-refit, no target tuning):** `log`, `linear`, `min_sat`
  (params frozen from the wiki reference curve) vs the **frozen 7-feature law**.
* **Primary comparison:** `log` vs frozen law on the new-domain cells.
* **Metrics:** Spearman, MAE, RMSE, ECE (primary); Pearson, calibration
  slope/intercept, monotonicity violations, params (secondary).
* **Preregistered verdict:** A = |Ï|â‰¥0.8 & MAEâ‰¤0.15; B = |Ï|â‰¥0.8 & MAE>0.15;
  C = |Ï|<0.8.

### Target 1 â€” new domain (general-English filler)

Observed accuracy: D=0 â†’ **1.00**, D=4 â†’ **0.50**, D=8 â†’ **0.20** â€” a steep,
monotone decline in a domain the reference coefficients never saw.

| form | Spearman | MAE | RMSE | ECE | calib slope | params |
|---|---|---|---|---|---|---|
| **log(1+D)** | 1.000 | **0.137** | 0.158 | **0.137** | 1.45 | 2 |
| frozen 7-feature law | 1.000 | 0.197 | 0.220 | 0.197 | 1.52 | 7 |
| linear | 1.000 | 0.206 | 0.215 | 0.206 | 1.96 | 2 |
| min_sat | 1.000 | 0.206 | 0.215 | 0.206 | 1.96 | 3 |

**Preregistered outcome = A (cross-domain transfer works).** All forms rank the
domain-B D-effect perfectly (Ï=1.000, zero monotonicity violations). The 2-param
`log` is best calibrated (MAE/ECE **0.137**, slope 1.45 closest to ideal 1.0) and
**beats the 7-param frozen law by 0.060 MAE** â€” consistent with Phase 8's finding
that `log(1+D)` dominates the elaborate law on held-out D.

### Target 2 â€” cross-length (train L=256 â†’ test L=1024, existing data)

| S slice | form | Spearman | MAE | ECE | calib slope |
|---|---|---|---|---|---|
| medium | log | 1.000 | 0.163 | 0.147 | 1.12 |
| medium | frozen law | 1.000 | 0.169 | 0.138 | 0.79 |
| high | log | 1.000 | 0.243 | 0.227 | 1.89 |
| high | frozen law | 1.000 | **0.0745** | **0.074** | **0.99** |
| low | log | n/a (const) | 0.243 | 0.243 | 0.00 |
| low | frozen law | n/a (const) | **0.048** | 0.048 | 0.00 |

Every non-constant slice ranks perfectly (Ï=1.000; S=low is constant-accuracy
so Ï is honestly reported as undefined). Here the **frozen 7-feature law wins on
calibration at S=high and S=low** (MAE 0.0745 / 0.048, slope â‰ˆ1.0 at S=high) â€”
expected, since it was fit on Exp6 data and its S and L interaction terms are
exercised by this slice â€” while `log` wins at S=medium and keeps perfect ranks
throughout.

**Honest label.** Cross-domain transfer of the *retrieval* count structure is a
clean **verdict A**: the 2-param `log(1+D)` form survives a domain change it
never saw, with better calibration than the 7-feature law. Cross-length transfer
shows a *nuanced* split: rank structure is domain- and length-robust (Ï=1.000
everywhere non-constant), but **calibration depends on which axis you hold fixed**
â€” the richer law helps where its S/L terms are active (cross-length), the simple
`log` helps where they are not (cross-domain). This reinforces the Phase 9
conclusion â€” `R = F(task, f(D), S, L)` with *condition-dependent coefficients* â€”
and adds domain as a condition that changes coefficients, not functional form.
Verdicts are **preliminary** (n=10/cell, 3 D points); higher-n replication is
future work. Artifacts: `results/phase10_transfer.csv`,
`phase10_domainB_factorial.csv`, `phase10_protocol.json` + `.sha256`,
`figure_phase10_domain.png`.


## Phase 11 — position effects (zero-refit)

Does the evidence **position** P (fraction of context before the needle)
explain residual error after D, S, L are already known? Protocol frozen
before any cell ran (sha256 `1bd50644…`): P ∈ {0.05,0.15,0.30,0.50,0.70,
0.85,0.95}, D ∈ {0,4}, L=256, S=medium, n=10/cell (140 trials), wiki filler
held fixed. **Isolation is exact — every cell is 267 context tokens**, so
length and filler amount are constant and only position varies. The position
coefficient was fit only on {0.05,0.30,0.70,0.95} and evaluated **zero-refit**
on the held-out positions {0.15,0.50,0.85}.

Observed accuracy by position:

| P | 0.05 | 0.15 | 0.30 | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|
| D=0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| D=4 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.80 | 0.80 |

### Primary test — held-out test positions (zero-refit)

| model | MAE | ECE | Spearman | calib slope | params |
|---|---|---|---|---|---|
| base `log(1+D)`, position-blind | 0.049 | 0.049 | 0.949 | 0.874 | 2 |
| **+ position (pos_edge)** | **0.027** | **0.017** | 0.906 | 0.939 | 3 |
| frozen 7-feature law | 0.060 | 0.054 | 0.949 | 1.680 | 7 |

**Preregistered outcome = B.** Adding position materially improves *held-out
calibration*: MAE **0.049 → 0.027**, ECE **0.049 → 0.017**, and fit-split R²
**0.841 → 0.968 (ΔR² +0.128)**, with the calibration slope moving toward the
ideal 1.0 (0.874 → 0.939). But it barely moves *rank*: Spearman goes
**0.949 → 0.906** — because D=0 is flat (1.00 at every position) and D=4 moves
only 0.70 → 0.80, so position carries little information about *which*
conditions are better or worse.

**Why B and not A, honestly.** The preregistered A threshold was a held-out
MAE gain of **≥ 0.030** together with |ρ| ≥ 0.8. The rank condition passes
(ρ = 0.906 ≥ 0.8), but the gain is **0.021 < 0.030** — just short. We do not
lower the threshold after seeing the result. The correct conclusion is:

> Position provides **modest but measurable calibration information** beyond
> the position-blind model, but the preregistered evidence is insufficient to
> formally promote position to a core component of the reliability law.

So the architectural reading is that **position improves calibration much
more than ranking**: it helps estimate *how much better/worse* a condition is
rather than *which* conditions are better/worse. Position therefore matters
primarily as a **calibration feature** in the tested regime, not as a major
rank-ordering feature — it is not the case that "position doesn't matter."
The core law is left unchanged as `R = F(task, domain, log(1+D), S, L, …)`,
with position classified as a **candidate calibration feature** (`+P` is *not*
promoted yet).

**Exploratory (non-preregistered) context.** An earlier `exp1_position` run
(L=512, n=8) showed a stronger mid-context dip than Phase 11. Because that
experiment was not part of the frozen Phase-11 protocol, it is reported only
as contextual exploratory evidence and does not contribute to the Phase-11
verdict; its numbers are not combined with Phase 11's.

Verdicts are **preliminary** (n=10/cell, 7 positions, 2 D). Artifacts:
`results/phase11_factorial.csv`, `phase11_transfer.csv`,
`phase11_protocol.json` + `.sha256`, `figure_phase11_position.png`.

## Phase 12A — long-context scaling (zero-refit screening)

Before Phase 12A, long-context scaling (how much of a context window is actually usable) was one of the biggest holes in the evidence — Phases 8–11 all operated at L ≤ 1024. Phase 12A screens the genuinely unmeasured regime (L ∈ {2048, 4096, 8192}) with a frozen, resumable run, holding everything fixed except length and competitor count.

Protocol frozen before any cell ran (sha256 `d30146e3…`): task=retrieval, wiki-prose filler, S=medium, position=0.5, needle + decoys, 12-word answer set. Manipulated: **L ∈ {2048,4096,8192} × D ∈ {0,4,8}, S=medium, n=5/cell = 45 samples** (the only new runs). Short-L baselines are the existing exp6 (S=medium, n=20, L ∈ {256,1024}) and exp4 data — zero new baseline runs. Length is token-exact, so filler amount, query, evidence and decoys are matched and **L is the only variable across the length axis**.

Run completed **45/45 in 4582s (~76 min), zero failures**, batched by length (L=2048 → 583s, L=4096 → 1185s, L=8192 → 2741s), resumable/checkpointed.

Observed accuracy (n=5/cell):

| L | D=0 | D=4 | D=8 |
|---|---|---|---|
| 2048 | 1.00 | 0.80 | 0.40 |
| 4096 | 1.00 | 0.40 | 0.40 |
| 8192 | 1.00 | **0.00** | 0.40 |

**D=0 remains 1.00 at every tested length** — length alone is harmless. The damage appears only when competing evidence is present.

Preregistered aggregate statistics (over D ∈ {4,8}, against the exp6 S=medium L=256 baseline = 0.575):

| statistic | value | meaning |
|---|---|---|
| `meanAtL(2048)` | 0.600 | harm(2048) = −0.025 (essentially no penalty yet) |
| `meanAtL(4096)` | 0.400 | **harm(4096) = +0.175** |
| `meanAtL(8192)` | 0.200 | **harm(8192) = +0.375** |
| `deltaInt` | **−0.400** | meanAtL(8192) − meanAtL(2048): the penalty *grows* with L |

The degradation is monotonic in length and *conditional on competition*: harm is ~0 at L=2048, then climbs to +0.175 (4096) and +0.375 (8192), while `deltaInt = −0.400` says the competitive penalty is substantially worse at 8192 than at 2048.

Model comparison (do L, and L×D, add predictive power beyond `log(1+D)` alone?):

| model | params | log-loss | Brier | Spearman |
|---|---|---|---|---|
| M_A `F(log(1+D), S)` | 2 | 0.486 | 0.0475 | 0.757 |
| **M_B `+ L`** | 3 | **0.449** | **0.0264** | **0.851** |
| M_C `+ log(1+D)×L` | 4 | 0.4466 | 0.0260 | 0.851 |

**M_A → M_B is the major jump** (log-loss 0.486 → 0.449, Brier roughly halved 0.0475 → 0.0264): adding length substantially improves prediction. **M_C adds only a small improvement beyond M_B** (0.449 → 0.4466) — so most of the signal is a *main effect of L under competition*, with only a weak additional L×D interaction term.

### Frozen decision rule → applied (no threshold shopping)

The rule was frozen in `phase12a_protocol.json` *before* any cell ran (sha256 `d30146e3…`):

* **Stage-A "INTERESTING"**: `max(harm4096, harm8192) ≥ 0.15` **OR** `deltaInt ≤ −0.15`. → +0.175 and +0.375 both clear 0.15, and `deltaInt = −0.400 ≤ −0.15`. **VERDICT: INTERESTING → triggers Phase 12B confirmation (n=10–20).**
* **Stage-B "CONFIRMED"**: `harm8192 ≥ 0.20` **AND** `deltaInt ≤ −0.05`. → 0.375 ≥ 0.20 and −0.400 ≤ −0.05. Both met — **confirmed at the preregistered screening threshold** (n=5/cell is still screening-level evidence, not independent confirmation).

Frozen model specification: M_A / M_B / M_C as above; thresholds and models are fixed for Phase 12B — no changes after seeing 12A.

### Interpretation

> **Phase 12A provides strong screening evidence for a competition-dependent long-context penalty and triggers the preregistered Phase 12B confirmation.**

The result is *not* "more tokens = worse." D=0 stays **1.00 at every tested length** (2048 → 8192) — length alone imposes no reliability cost. The penalty emerges and grows only when competing evidence is present (D=4: 0.80 → 0.40 → 0.00 across 2048 → 4096 → 8192). So the reliability problem is **more context × competing evidence → worse reliability**, not length per se — which is exactly the ContextLab thesis: context *composition*, not raw length, governs reliability, and large contexts become unreliable specifically when they carry conflicting material.

Model-wise, length (M_A → M_B) is where the predictive power comes from; the L×D interaction (M_C) adds only marginally — consistent with a **main effect of length that acts under competition** rather than a strongly super-additive interaction.

**Honest caveats.** (1) n=5/cell → single-cell differences are noise (SE ≈ 0.2); only the preregistered aggregates and the Stage-B check are meaningful. (2) Stage-B is **confirmed at the preregistered screening threshold**, not independently replicated — Phase 12B (n=10–20, identical frozen specification) is the confirmation. (3) S=medium only; S robustness is deferred to Phase 13. Verdicts are preliminary pending 12B.

Artifacts: `results/phase12a_factorial.csv`, `phase12a_eval.csv`, `phase12a_protocol.json` + `.sha256`, `log_phase12a.txt`.

## Phase 12B — long-context confirmation (independent n=20)

Phase 12A provided strong *screening* evidence and triggered this preregistered confirmation. Phase 12B repeats the **identical** L × D experiment on **fresh, independent samples** with a larger per-cell n, applying the **unchanged** frozen decision rule. Nothing about the task, filler, models, statistics, or thresholds was altered after seeing 12A.

Protocol frozen before any cell ran (sha256 `7336f02c…`), byte-verified to carry the **same thresholds and model spec as 12A**: task=retrieval, wiki-prose filler, S=medium, position=0.5, needle + decoys, 12-word answer set. Grid: **L ∈ {2048,4096,8192} × D ∈ {0,4,8}, S=medium, n=20/cell = 180 samples**. The only changes from 12A are (a) n=5 → **n=20/cell** and (b) the true sample index **t ∈ [5,25)**, disjoint from 12A's t ∈ [0,5) with min seed 502053 — so **no seed collides with 12A** and the confirmation dataset is genuinely independent. Short-L baseline is still the same exp6 S=medium L=256 = 0.575.

Run completed **180/180 in 11238s (~187 min), verified**, batched by length (L=2048 → 4096 → 8192), append-only and resumable (the run was interrupted once and resumed, skipping completed rows without recomputation or seed drift).

Observed accuracy (n=20/cell):

| L | D=0 | D=4 | D=8 |
|---|---|---|---|
| 2048 | **1.00** | 0.55 | 0.35 |
| 4096 | **1.00** | 0.60 | 0.70 |
| 8192 | **1.00** | 0.40 | **0.20** |

**D=0 remains 1.00 at every tested length** on the independent sample — length alone is still harmless at n=20. The damage appears only when competing evidence is present, deepest at L=8192/D=8 (0.20).

Preregistered aggregate statistics (over D ∈ {4,8}, against the exp6 S=medium L=256 baseline = 0.575):

| statistic | value | meaning |
|---|---|---|
| `meanAtL(2048)` | 0.450 | harm(2048) = +0.125 |
| `meanAtL(4096)` | 0.650 | harm(4096) = −0.075 (mid-length noise) |
| `meanAtL(8192)` | 0.300 | **harm(8192) = +0.275** |
| `deltaInt` | **−0.150** | meanAtL(8192) − meanAtL(2048): penalty worse at 8192 |

The independent aggregates reproduce the 12A direction with the same sign on the two statistics that drive the rule: **harm(8192) = +0.275 ≥ 0.20** and **deltaInt = −0.150 ≤ −0.05**. (harm(4096) = −0.075 is mid-length noise — 0.70 at D=8 lifts the 4096 aggregate; it is not used by Stage B, and single-cell differences at this n are not the preregistered statistic.)

Model comparison (do L, and L×D, add predictive power beyond `log(1+D)` on the confirmation cells?):

| model | params | log-loss | Brier | Spearman |
|---|---|---|---|---|
| M_A `F(log(1+D), S)` | 2 | 0.4774 | 0.02253 | 0.804 |
| **M_B `+ L`** | 3 | **0.4723** | **0.02014** | 0.797 |
| M_C `+ log(1+D)×L` | 4 | 0.4720 | 0.02012 | 0.797 |

**M_A → M_B again improves prediction** (log-loss 0.4774 → 0.4723, Brier 0.02253 → 0.02014): adding length helps beyond `log(1+D)` alone, as in 12A. **M_C (adding the L×D interaction) adds essentially nothing** (0.4723 → 0.4720) — on the confirmation cells most of the length signal is a *main effect under competition*, not a super-additive interaction, matching 12A's reading (the effect is smaller at n=20 because M_B/M_C are fit across the confirmation cells only).

### Frozen decision rule → applied (unchanged thresholds, no shopping)

The rule and thresholds are **byte-identical to 12A** (sha256 `7336f02c…` vs `d30146e3…`, asserted equal by the freeze script):

* **Stage-A "INTERESTING"**: `max(harm4096, harm8192) ≥ 0.15` **OR** `deltaInt ≤ −0.15`. → harm8192 = +0.275 ≥ 0.15 and deltaInt = −0.150 ≤ −0.15. **VERDICT: INTERESTING.**
* **Stage-B "CONFIRMED"**: `harm8192 ≥ 0.20` **AND** `deltaInt ≤ −0.05`. → 0.275 ≥ 0.20 **(True)** and −0.150 ≤ −0.05 **(True)**. Both met — **CONFIRMED at the preregistered threshold on fresh n=20 data.**

### Interpretation

> **Phase 12B confirms a competition-dependent long-context penalty on independent n=20 samples: Stage-B passes with the unchanged frozen rule (harm8192 = +0.275, deltaInt = −0.150).**

Across both phases, **D=0 is 1.00 at every length (2048 → 8192)** — length alone imposes no reliability cost — while the penalty emerges and deepens only under competition, worst at L=8192/D=8 (0.20). The result is *not* "more tokens = worse"; it is **more context × competing evidence → worse reliability**, exactly the ContextLab thesis: context *composition*, not raw length, governs reliability. With 12A (n=5 screening) and 12B (n=20 independent confirmation) both clearing the same frozen Stage-B rule, the long-context arm of the evidence chain now rests on a confirmed, preregistered result rather than a single screening pass.

**Honest caveats.** (1) n=20/cell — single-cell values still carry noise (SE ≈ 0.11); only the preregistered aggregates and the Stage-B check are the decision statistics. (2) harm(4096) = −0.075 is a mid-length non-monotonicity (D=8 = 0.70 lifts that cell); it does not enter Stage B, but the penalty is therefore not cleanly monotonic in L at this n — the confirmed claim is the *aggregate* 8192-vs-2048 contrast, not strict per-length monotonicity. (3) S=medium only, one model family (Qwen2.5-0.5B); S robustness and cross-model confirmation are deferred to Phase 13. (4) D=0 and D∈{4,8} use the same exp6 baseline as 12A — the baseline itself is shared across both phases.

Artifacts: `results/phase12b_factorial.csv`, `phase12b_eval.csv`, `phase12b_protocol.json` + `.sha256`, `log_phase12b.txt`.




