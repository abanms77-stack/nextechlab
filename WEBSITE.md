# ContextLab — project layout

Two clearly separated halves.

## 1. Research (private)

- `contextlab/` — benchmark + profiler research code
- `experiments/` — experiment runners and analyses
- `results/` — raw data (CSVs, figures)
- `data/` — WikiText filler corpus
- `tests/` — unit tests
- `REPORT.md`, `ContextLab_Report.pdf` — full 26-page report

Status: **frozen.** Final claim: a *retrieval-interference* theory — within the
retrieval task family, across three model families, accuracy is predicted by a
multiplicative interaction of competitor count and measured similarity; the
dominant structure transfers cross-model and pruning on the measured metric
improves performance.

## 2. Website / product (to publish)

- `website/` — Next.js 14 + TypeScript + Tailwind app
  - marketing pages, playground, dashboard, and `POST /api/profile`

Run:

```bash
cd website
npm install
npm run dev      # http://localhost:3000
```

See `website/README.md` for the public/private IP boundary.

## The bridge

`lib/profile.ts` / `lib/profiler.ts` in the website implement the **public,
quality-blind** version of the research ideas: they compute per-passage
similarity to the query, count competitors, aggregate interference, and estimate
retrieval risk — using only principles, never the fitted coefficients, the
evaluation set, or the optimization policy.