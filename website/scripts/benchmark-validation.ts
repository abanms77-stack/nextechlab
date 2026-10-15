// BENCHMARK VALIDATION (frozen engine; real research cells + real templates).
// Reconstructs each cell of results/summary_exp6_cells.csv (n=20/cell) with
// templates mirroring contextlab/tasks/generator.py, runs the FROZEN public
// engine, and compares ordering vs OBSERVED accuracy. No tuning here.
//
//   npx tsx scripts/benchmark-validation.ts            (summary)
//   npx tsx scripts/benchmark-validation.ts --csv      (per-cell CSV rows)
//   npx tsx scripts/benchmark-validation.ts --out <path>

import { readFileSync, writeFileSync } from "node:fs";
import { profileContext } from "../lib/profiler";

interface Cell { length: number; similarity: string; D: number; n: number; acc: number; }

function readCells(): Cell[] {
  const text = readFileSync("../results/summary_exp6_cells.csv", "utf8").trim().split("\n");
  return text.slice(1).map((line) => {
    const [length, similarity, D, n, acc] = line.split(",");
    return { length: +length, similarity, D: +D, n: +n, acc: +acc };
  });
}

// Real generator templates (mirror of contextlab/tasks/generator.py).
const QUERY = "What is the vault password?";
const NEEDLE = (w: string) => `Important: the vault password is ${w}.`;
const DECOY_MED = (noun: string, w: string) => `Note: the ${noun} password is ${w}.`;
const DECOY_HIGH = (noun: string, w: string) => `Important: the ${noun} password is ${w}.`;
const LOW_SENTENCES = [
  "The weather in the northern region was unusually warm this week.",
  "The council postponed the meeting until further notice.",
  "A new bakery opened near the station last month.",
  "The orchestra rehearsed late into the evening.",
  "Ticket sales rose sharply after the announcement.",
  "The bridge will be closed for repairs on Sunday.",
];
const MEDIUM_NOUNS = ["storage","archive","mail","gate","access","backup","intranet","guest","admin","media"];
const HIGH_NOUNS = ["safe","strongbox","lockbox","strongroom","vault door","vault backup","safe deposit","ledger"];
const ANSWER_WORDS = ["apple","river","tiger","cloud","stone","candle","silver","garden","planet","forest","harbor","lantern"];
const FILLER = [
  "The committee met on Tuesday to review the quarterly budget.",
  "Local officials announced a new schedule for road maintenance.",
  "The train service was delayed due to signal upgrades on the north line.",
  "Researchers presented their findings at the annual conference.",
  "The bakery on the corner opens early every weekday morning.",
  "Attendance at the town hall meeting was higher than expected.",
];

function estimateTokens(s: string): number {
  return Math.round(s.trim().split(/\s+/).filter(Boolean).length * 1.33);
}

function buildContext(length: number, similarity: string, D: number): string[] {
  const passages = [NEEDLE("apple")];
  for (let i = 0; i < D; i++) {
    const w = ANSWER_WORDS[(i + 3) % ANSWER_WORDS.length];
    if (similarity === "low") passages.push(LOW_SENTENCES[i % LOW_SENTENCES.length]);
    else if (similarity === "medium") passages.push(DECOY_MED(MEDIUM_NOUNS[i % MEDIUM_NOUNS.length], w));
    else passages.push(DECOY_HIGH(HIGH_NOUNS[i % HIGH_NOUNS.length], w));
  }
  let idx = 0;
  let guard = 0;
  while (passages.reduce((a, p) => a + estimateTokens(p), 0) < length && guard++ < 600) {
    passages.push(FILLER[idx++ % FILLER.length]);
  }
  return passages;
}

function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]) => {
    const idx = v.map((x, i) => [x, i] as [number, number]).sort((p, q) => p[0] - q[0]);
    const r = new Array(v.length).fill(0);
    let k = 0;
    while (k < idx.length) {
      let j = k;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[k][0]) j++;
      const avg = (k + j) / 2;
      for (let t = k; t <= j; t++) r[idx[t][1]] = avg;
      k = j + 1;
    }
    return r;
  };
  const ra = rank(a);
  const rb = rank(b);
  const n = a.length;
  const ma = ra.reduce((s, x) => s + x, 0) / n;
  const mb = rb.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - mb) ** 2;
  }
  return da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db);
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  const ma = a.reduce((s, x) => s + x, 0) / n;
  const mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db);
}

function ece(pred: number[], obs: number[], bins = 5): number {
  let err = 0;
  for (let b = 0; b < bins; b++) {
    const lo = b / bins;
    const hi = (b + 1) / bins;
    const idx = pred.map((p, i) => (p >= lo && (p < hi || (b === bins - 1 && p <= hi)) ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) continue;
    const mp = idx.reduce((s, i) => s + pred[i], 0) / idx.length;
    const mo = idx.reduce((s, i) => s + obs[i], 0) / idx.length;
    err += (idx.length / pred.length) * Math.abs(mp - mo);
  }
  return err;
}
const cells = readCells().sort(
  (x, y) => x.length - y.length || x.similarity.localeCompare(y.similarity) || x.D - y.D,
);

const rows = cells.map((c) => {
  const r = profileContext({ query: QUERY, passages: buildContext(c.length, c.similarity, c.D) });
  return {
    L: c.length, S_tier: c.similarity, D_research: c.D, n: c.n,
    observed_accuracy: c.acc, predicted_success: r.predictedSuccess,
    D_profiler: r.features.competitors, S_profiler: r.features.competitorSimilarity,
    I_profiler: r.features.interference, risk: r.risk,
    support: r.features.supportingPairs, redundant: r.features.redundantPairs,
    contradict: r.features.contradictoryPairs, related: r.features.relatedPairs,
  };
});

const obs = rows.map((r) => r.observed_accuracy);
const pred = rows.map((r) => r.predicted_success);
const rho = spearman(obs, pred);
const rPear = pearson(obs, pred);
const mae = rows.reduce((s, x) => s + Math.abs(x.predicted_success - x.observed_accuracy), 0) / rows.length;
const cal = ece(pred, obs);
const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;

const args = process.argv.slice(2);
if (args.includes("--csv") || args.some((a) => a.startsWith("--out"))) {
  const header = "L,S_tier,D_research,n,observed_accuracy,predicted_success,D_profiler,S_profiler,I_profiler,risk,support,redundant,contradict,related";
  const lines = rows.map((r) =>
    [r.L, r.S_tier, r.D_research, r.n, r.observed_accuracy.toFixed(3), r.predicted_success.toFixed(4),
     r.D_profiler, r.S_profiler.toFixed(4), r.I_profiler.toFixed(4), r.risk,
     r.support, r.redundant, r.contradict, r.related].join(","),
  );
  const csv = header + "\n" + lines.join("\n") + "\n";
  const outIdx = args.findIndex((a) => a === "--out");
  if (outIdx >= 0 && args[outIdx + 1]) writeFileSync(args[outIdx + 1], csv);
  else process.stdout.write(csv);
} else {
  console.log("=== BENCHMARK VALIDATION (frozen engine vs real research cells) ===");
  console.log("source: results/summary_exp6_cells.csv (n=20/cell); templates mirror contextlab/tasks/generator.py");
  console.log("L     S-tier   Dres  observed  predicted  Dp  Sp    I      risk   sup red con rel");
  for (const r of rows) {
    console.log(
      `${String(r.L).padEnd(5)} ${r.S_tier.padEnd(8)} ${String(r.D_research).padEnd(4)} ` +
      `${(r.observed_accuracy * 100).toFixed(0).padStart(6)}%  ${(r.predicted_success * 100).toFixed(0).padStart(7)}%  ` +
      `${String(r.D_profiler).padStart(2)}  ${(r.S_profiler * 100).toFixed(0).padStart(3)}%  ` +
      `${r.I_profiler.toFixed(2)}  ${r.risk.padEnd(6)} ${r.support}   ${r.redundant}   ${r.contradict}   ${r.related}`,
    );
  }
  console.log(`\nSpearman rho (observed vs predicted) = ${rho.toFixed(3)}`);
  console.log(`Pearson r  (observed vs predicted) = ${rPear.toFixed(3)}`);
  console.log(`MAE (predicted vs observed)        = ${mae.toFixed(3)}`);
  console.log(`ECE (5-bin, predicted vs observed) = ${cal.toFixed(3)}`);
  console.log("\n=== monotonicity: higher D -> lower-or-equal predicted success? ===");
  let violations = 0;
  for (const sim of ["low", "medium", "high"]) {
    for (const L of [256, 1024]) {
      const g = rows.filter((r) => r.S_tier === sim && r.L === L).sort((a, b) => a.D_research - b.D_research);
      if (!g.length) continue;
      const preds = g.map((r) => r.predicted_success);
      const obsv = g.map((r) => r.observed_accuracy);
      const mono = (v: number[]) => v.every((x, i) => i === 0 || x <= v[i - 1] + 0.02);
      if (!mono(preds)) violations++;
      console.log(
        `  L=${String(L).padEnd(5)} S=${sim.padEnd(7)} ` +
        `obs ${obsv.map((x) => (x * 100).toFixed(0)).join(">")} ` +
        `| pred ${preds.map((x) => (x * 100).toFixed(0)).join(">")} ` +
        `| predMono=${mono(preds)} obsMono=${mono(obsv)}`,
      );
    }
  }
  console.log(`monotonicity violations (predicted): ${violations}/6 families`);
  console.log("\n=== per-similarity breakdown ===");
  for (const sim of ["low", "medium", "high"]) {
    const g = rows.filter((r) => r.S_tier === sim);
    console.log(
      `  S=${sim.padEnd(7)} rho=${spearman(g.map((r) => r.observed_accuracy), g.map((r) => r.predicted_success)).toFixed(3)} ` +
      `MAE=${(g.reduce((s, r) => s + Math.abs(r.predicted_success - r.observed_accuracy), 0) / g.length).toFixed(3)} ` +
      `mean obs=${(mean(g.map((r) => r.observed_accuracy)) * 100).toFixed(0)}% mean pred=${(mean(g.map((r) => r.predicted_success)) * 100).toFixed(0)}%`,
    );
  }
  console.log("\n=== per-length breakdown ===");
  for (const L of [256, 1024]) {
    const g = rows.filter((r) => r.L === L);
    console.log(
      `  L=${L} rho=${spearman(g.map((r) => r.observed_accuracy), g.map((r) => r.predicted_success)).toFixed(3)} ` +
      `MAE=${(g.reduce((s, r) => s + Math.abs(r.predicted_success - r.observed_accuracy), 0) / g.length).toFixed(3)}`,
    );
  }
  console.log("\n=== relationship audit (frozen relation layer) ===");
  console.log(
    `  support=${rows.reduce((s, r) => s + r.support, 0)} ` +
    `redundant=${rows.reduce((s, r) => s + r.redundant, 0)} ` +
    `contradict=${rows.reduce((s, r) => s + r.contradict, 0)} ` +
    `related=${rows.reduce((s, r) => s + r.related, 0)}`,
  );
  console.log(`Claim: 'On the current product benchmark, profiler predictions rank observed retrieval outcomes with Spearman rho = ${rho.toFixed(3)}.'`);
}

