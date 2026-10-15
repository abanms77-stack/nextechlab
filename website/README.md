# ContextLab — website

Production-ready marketing site + product (Context Profiler) built with
**Next.js 14 (App Router) + TypeScript + Tailwind CSS**.

## Run

```bash
npm install
npm run dev        # http://localhost:3000
# or
npm run build && npm start
```

## Pages

| Route | Purpose |
|---|---|
| `/` | Home: hero, live profiler, problem, evidence, CTA |
| `/product` | Context Profiler · Optimizer · Before/After · API |
| `/playground` | Full-screen profiler + optimizer |
| `/developers` | API docs, SDK, examples, limits |
| `/research` | Public, high-level research summary |
| `/pricing` | Free / Developer / Team / Enterprise |
| `/about` | Research → Product story |
| `/trust` | Trust & security |
| `/status` | Service status |
| `/dashboard` | Overview · Analyze · Runs · API keys · Settings |
| `/api/profile` | **POST** — profile a context (see below) |

## API

```bash
curl -X POST http://localhost:3000/api/profile \
  -H "content-type: application/json" \
  -d '{"query":"What is the refund policy?","context":"passage 1\n\npassage 2"}'
```

Body: `{ query, context? , passages? , modelWindow? }`
Returns: `{ risk, predictedSuccess, confidence, features, passages[],
highRiskIds[], recommendations[], optimized }`

Privacy: the endpoint analyses structure only and **does not persist raw
passage text**; responses contain short previews.

## Architecture

```
app/            routes (marketing + product + api)
components/     Nav, Footer, Profiler, ResultPanel, Dashboard, ui primitives
lib/profile.ts  similarity layer (measured), token estimate, constants, types
lib/profiler.ts profileContext() engine + splitPassages()
lib/samples.ts  demo contexts
scripts/        probe.mjs (manual) + test-engine.mjs (behavioural tests)
```

## Engine pipeline (profiler-2.0)

```
Query ─┬─► relevance(passage)         = query term coverage
       └─► similarity(passage, other) = stemmed-token Jaccard
                                        + char-trigram Jaccard
                    │
                    ▼
        competing pairs  (similarity >= floor AND both are query candidates)
                    │
                    ▼
        clusters (union-find)  →  D = competitors, S = mean pair similarity
                    │
                    ▼
        interference  I = min(1, D / 5) × S   (+ smaller length term)
                    │
                    ▼
        predictedSuccess = logistic(I, length)  →  risk + recommendations
```

Similarity is **measured**, not hand-labelled: token Jaccard is blended with
character-trigram Jaccard so morphology (`refund`/`refunds`) does not break it.
Competition is measured **between passages**, matching the research variable,
not merely query-to-passage matching.

## Two-layer architecture (v2.1)

A single binary `competes` flag cannot represent the ways two passages relate.
The engine therefore has **two independent layers**, exactly as the review
specified:

**Layer 1 — Interference (research metric, preserved)**
```
competes  = similarity >= COMPETITION_FLOOR  AND  both passages on-topic
D         = passages in a competing cluster
S         = mean similarity of competing pairs
I         = min(1, D / 5) * S        -> predictedSuccess
```
This is the variable validated against the research benchmark. It is **not**
coupled to the logical relations, because the research's interference is a
*surface* effect (measured similarity × count) that holds even when passages
address different keys. Coupling it broke the bridge correlation (0.746 → −0.19).

**Layer 2 — Relations (drives the optimizer + conflict flag)**
```
support    compatible answers, different evidence  -> KEEP
redundant  compatible answers, near-duplicate      -> OPTIMIZE (keep strongest)
contradict INCOMPATIBLE answers to the same slot   -> FLAG, never auto-resolve
related    on-topic, not the same answer slot      -> KEEP
irrelevant not about the query                     -> KEEP (never delete for this)
```
Answers are extracted with a small deterministic pattern library
(`lib/relations.ts`) covering `who wrote X`, `who is the author of X`,
`what is the <attr> of X`, and definitional `what is X`. When the query shape is
unknown, classification falls back conservatively: only a near-duplicate
(similarity ≥ 0.8) is "redundant", otherwise a similar on-topic pair is
"support". It can therefore **never** report a contradiction it did not extract.

## Optimizer rules

- No redundant/contradictory pairs → **context returned unchanged** (`removed = 0`).
- Redundant group → keep the strongest evidence, prune the rest.
- **Group containing a contradiction → keep every member** and surface the
  conflict; the profiler must not silently delete one side of a factual conflict.
- Low query relevance is **never** a reason to delete context.

## Field semantics
- `passage.nearestSimilarity` = raw max similarity to any other passage.
- `passage.competition` = max similarity to a partner whose **relation** is
  redundant/contradict (0 for purely supporting evidence). This is what the UI
  shows as "competition".

## Thresholds
`COMPETITION_FLOOR` (0.35) and the logistic constants were **not** tuned to pass
the demo; the fixes are in feature extraction, relation classification and
optimizer semantics.

## Diagnostics

`POST /api/profile?debug=1` (or `{"debug": true}`) returns a `diagnostics`
block: query tokens, per-passage stemmed tokens + relevance, and per-pair
`similarity / relevanceA / relevanceB / strength / passesCompetitionFloor /
passesCandidateFilter / competes`. This pinpoints where the pipeline breaks.

## Bridge validation (research → product)

`npm run validate:research` reconstructs the **exact experimental contexts**
(same needle/decoy templates, same filler neutrality), runs them through the
public engine, and compares against the **observed** accuracy in
`../results/exp6_factorial.csv` (n=20 per cell).

Result on the 30 research cells (D ∈ {0,1,2,4,8} × S ∈ {low,med,high} × L ∈ {256,1024}):

- **Spearman rank correlation (observed vs predicted) = 0.746**
- **Monotonicity matched in all 6 similarity×length families** — as D rises, both
  observed accuracy and predicted success fall together.
- **Similarity families ordered correctly**: mean observed 100% / 69% / 51% for
  low / medium / high ≈ predicted 94% / 83% / 77%.

**Honest limitation — calibration is compressed.** The public engine predicts in
a narrower band (~58–94%) than the research observed (0–100%). Ordering is
validated; absolute calibration is deliberately *not* — the calibrated
coefficients are private. So the product claim is:

> contexts the research found harder are consistently profiled as riskier.

It does **not** claim the public engine reproduces the research's absolute
accuracies.

## End-to-end loop

`npm run demo` runs **Detect → Explain → Optimize → Re-profile**, every number
measured from the engine:

```
CLEAN        D= 0  I=0.00  pred= 94%  risk=low
COMPETING    D= 4  I=0.69  pred= 27%  risk=high
OPTIMIZED    D= 0  I=0.00  pred= 94%  risk=low
predicted success: 94% -> 27% -> 94%
```

## Tests

```bash
npm run dev              # terminal 1 (needed for test:engine / test:invariant)
npm run test:engine      # 19 behavioural assertions
npm run test:invariant   # adversarial sweep of the D=0 invariant
npm run diag             # function-level dump (tsx, no server)
npm run validate:research# research-vs-product ordering check (tsx, no server)
npm run demo             # end-to-end detect→optimize→re-profile (tsx)
npx tsx scripts/diag-relations.ts   # relation taxonomy on the reviewer's cases
```

Latest results: **26/26 behavioural tests**, invariant holds, bridge
Spearman ρ = 0.746, demo 94% → 27% → 94%.

Covers: morphology regression, clean → zero interference, unrelated → still
low, competing → higher, more competitors → higher, higher similarity → higher,
removing competitors → lower, optimizer prunes, risk ordering, **D=0 ⇒ nothing
pruned**, **low-relevance passages are kept**, and the reviewer's clean-vs-
redundant A/B contexts.

## PUBLIC vs PRIVATE (IP boundary — deliberate)

**Public (this repo/site):**
- the *principles* (interference = competing count × measured similarity)
- a quality-blind profiler that computes features from the query + context
- high-level research summary numbers

**Private (never shipped to the client or published):**
- the calibrated predictive coefficients and their fitting pipeline
- the evaluation set, prompt templates, and exact model(s)
- the exact optimization policy and its validation results
- the full 26-page research report and raw data

The public engine in `lib/` exposes only principles. Do not add the fitted
coefficients, evaluation set, or optimization policy to this app.

## Design notes

Technical, minimal, trustworthy: white/black, precise numbers, data-viz,
subtle grid, no AI clichés. The product should read like developer
infrastructure, not a consumer AI app.