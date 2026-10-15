// BRIDGE VALIDATION: does profiler-2.0's risk ordering match the research?
//
// Reconstructs the exact experimental contexts (same needle/decoy templates and
// neutral filler), runs them through the public profiler engine, and compares
// the profiler's ranking against the OBSERVED retrieval accuracy recorded in
// results/exp6_factorial.csv (n=20 per cell).
//
//   npx tsx scripts/validate-research.ts
//
// This does NOT refit coefficients. It asks one question:
//   "are contexts that were experimentally harder profiled as riskier?"

import { readFileSync } from "node:fs";
import { profileContext } from "../lib/profiler";

const ANSWER_WORDS = [
  "apple", "river", "tiger", "cloud", "stone", "candle",
  "silver", "garden", "planet", "forest", "harbor", "lantern",
];
const MEDIUM_NOUNS = [
  "storage", "archive", "mail", "gate", "access",
  "backup", "intranet", "guest", "admin", "media",
];
const HIGH_NOUNS = [
  "safe", "strongbox", "lockbox", "strongroom", "vault door",
  "vault backup", "safe deposit", "ledger",
];
const LOW_SENTENCES = [
  "The weather in the northern region was unusually warm this week.",
  "The council postponed the meeting until further notice.",
  "A new bakery opened near the station last month.",
  "The orchestra rehearsed late into the evening.",
  "Ticket sales rose sharply after the announcement.",
  "The bridge will be closed for repairs on Sunday.",
];
const FILLER = [
  "The committee met on Tuesday to review the quarterly budget.",
  "Local officials announced a new schedule for road maintenance.",
  "The train service was delayed due to signal upgrades.",
  "Researchers presented their findings at the annual conference.",
  "The library extended its opening hours during the exam period.",
  "A new bridge will connect the two sides of the district.",
];

const QUERY = "What is the vault password?";
const NEEDLE = "Important: the vault password is {w}.";
const DECOY_MED = "Note: the {n} password is {w}.";
const DECOY_HIGH_REDUNDANT =
  "Important: the vault password is {w}; confirm it in the vault log.";

function decoy(similarity: string, i: number): string {
  const w = ANSWER_WORDS[(i + 3) % ANSWER_WORDS.length];
  if (similarity === "low") return LOW_SENTENCES[i % LOW_SENTENCES.length];
  // HIGH in the research benchmark = same-template paraphrase with a DIFFERENT
  // answer word (competing candidates); MEDIUM = different template + noun.
  // The bridge therefore uses the high template with a different word so the
  // public engine sees competing (contradictory) answers, not same-answer
  // restatements. When neither side extracted an answer the engine can only
  // call a near-duplicate "redundant", it can never report a contradiction.
  if (similarity === "high")
    return DECOY_HIGH_REDUNDANT.replace("{w}", w);
  const nouns = MEDIUM_NOUNS;
  return DECOY_MED.replace("{n}", nouns[i % nouns.length]).replace("{w}", w);
}

function estimateTokens(s: string): number {
  return Math.round(s.trim().split(/\s+/).filter(Boolean).length * 1.33);
}

/** needle + D decoys + neutral filler to ~`length` tokens. */
function buildContext(length: number, similarity: string, D: number): string[] {
  const passages = [NEEDLE.replace("{w}", "apple")];
  for (let i = 0; i < D; i++) passages.push(decoy(similarity, i));
  let idx = 0;
  let guard = 0;
  while (
    passages.reduce((a, p) => a + estimateTokens(p), 0) < length &&
    guard++ < 500
  ) {
    passages.push(FILLER[idx++ % FILLER.length]);
  }
  return passages;
}

// ---- read the observed research outcome -------------------------------------
type Cell = { length: number; similarity: string; D: number; acc: number };

function readResearch(): Cell[] {
  const lines = readFileSync("../results/exp6_factorial.csv", "utf8").trim().split("\n");
  const head = lines[0].split(",");
  const iL = head.indexOf("length");
  const iS = head.indexOf("similarity");
  const iD = head.indexOf("n_distractors");
  const iC = head.indexOf("correct");
  const agg = new Map<string, { sum: number; n: number; c: Cell }>();
  for (const row of lines.slice(1)) {
    const p = row.split(",");
    const c: Cell = { length: +p[iL], similarity: p[iS], D: +p[iD], acc: 0 };
    const key = `${c.length}|${c.similarity}|${c.D}`;
    const e = agg.get(key) ?? { sum: 0, n: 0, c };
    e.sum += +p[iC];
    e.n++;
    agg.set(key, e);
  }
  return [...agg.values()].map((e) => ({ ...e.c, acc: e.sum / e.n }));
}

function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]) => {
    const idx = v.map((x, i) => [x, i] as [number, number]).sort((p, q) => p[0] - q[0]);
    const r = new Array(v.length).fill(0);
    idx.forEach(([, i], k) => (r[i] = k));
    return r;
  };
  const ra = rank(a);
  const rb = rank(b);
  const n = a.length;
  const m = (x: number[]) => x.reduce((s, y) => s + y, 0) / n;
  const ma = m(ra);
  const mb = m(rb);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}

// ---- run --------------------------------------------------------------------
const research = readResearch().sort(
  (x, y) =>
    x.length - y.length ||
    x.similarity.localeCompare(y.similarity) ||
    x.D - y.D,
);

const rows = research.map((c) => {
  const r = profileContext({
    query: QUERY,
    passages: buildContext(c.length, c.similarity, c.D),
  });
  return {
    cell: c,
    pred: r.predictedSuccess,
    risk: r.risk,
    P: r.features.competitors,
    S: r.features.competitorSimilarity,
    I: r.features.interference,
  };
});

console.log("=== BRIDGE: research-observed accuracy vs profiler-2.0 prediction ===");
console.log("L     S        D   observed  profiler   D_p  S_p    I     risk");
for (const r of rows) {
  console.log(
    `${String(r.cell.length).padEnd(5)} ${r.cell.similarity.padEnd(8)} ` +
      `${String(r.cell.D).padEnd(3)} ${(r.cell.acc * 100).toFixed(0).padStart(6)}%  ` +
      `${(r.pred * 100).toFixed(0).padStart(6)}%  ` +
      `${String(r.P).padStart(3)}  ${(r.S * 100).toFixed(0).padStart(3)}%  ` +
      `${r.I.toFixed(2)}   ${r.risk}`,
  );
}

const rho = spearman(
  rows.map((r) => r.cell.acc),
  rows.map((r) => r.pred),
);
console.log(`\nSpearman rank correlation (observed vs predicted) = ${rho.toFixed(3)}`);

console.log("\n=== monotonicity: higher D -> lower predicted success? ===");
for (const sim of ["low", "medium", "high"]) {
  for (const L of [256, 1024]) {
    const g = rows
      .filter((r) => r.cell.similarity === sim && r.cell.length === L)
      .sort((a, b) => a.cell.D - b.cell.D);
    if (!g.length) continue;
    const preds = g.map((r) => r.pred);
    const obs = g.map((r) => r.cell.acc);
    // non-increasing within a small tolerance (ties and rounding are allowed)
    const mono = (v: number[]) =>
      v.every((x, i) => i === 0 || x <= v[i - 1] + 0.02);
    console.log(
      `  L=${String(L).padEnd(5)} S=${sim.padEnd(7)} ` +
        `obs ${obs.map((x) => (x * 100).toFixed(0)).join(">")} ` +
        `| pred ${preds.map((x) => (x * 100).toFixed(0)).join(">")} ` +
        `| predMono=${mono(preds)} obsMono=${mono(obs)}`,
    );
  }
}

const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
console.log("\n=== separation by similarity family ===");
for (const sim of ["low", "medium", "high"]) {
  const g = rows.filter((r) => r.cell.similarity === sim);
  console.log(
    `  S=${sim.padEnd(7)} mean pred=${(mean(g.map((r) => r.pred)) * 100).toFixed(0)}%  ` +
      `mean observed=${(mean(g.map((r) => r.cell.acc)) * 100).toFixed(0)}%`,
  );
}