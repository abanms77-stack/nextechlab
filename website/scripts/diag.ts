// Unit-level diagnostic: exercises the REAL engine functions directly
// (no server, no UI). Run:  node scripts/diag.ts
import { queryCoverage, textSimilarity } from "../lib/profile.ts";
import { profileContext } from "../lib/profiler.ts";

const Q = "Who wrote the novel 1984?";
const pct = (x: number) => (x * 100).toFixed(0) + "%";

console.log("=== function-level checks ===");
console.log(
  "queryCoverage(Q, 'George Orwell wrote the novel 1984.') =",
  pct(queryCoverage(Q, "George Orwell wrote the novel 1984.")),
);
console.log(
  "queryCoverage(Q, 'The novel follows Winston Smith.')     =",
  pct(queryCoverage(Q, "The novel follows Winston Smith.")),
);
console.log(
  "queryCoverage(Q, 'A bakery opened near the station.')    =",
  pct(queryCoverage(Q, "A bakery opened near the station.")),
);
console.log(
  "textSimilarity(1984-by-Orwell, 1984-by-Huxley)           =",
  pct(
    textSimilarity(
      "George Orwell wrote the novel 1984.",
      "Aldous Huxley wrote the novel 1984.",
    ),
  ),
);

const clean = [
  "George Orwell wrote 1984.",
  "1984 was published in 1949.",
  "Winston Smith is the protagonist.",
  "The novel explores surveillance.",
  "Orwell also wrote Animal Farm.",
];
const redundant = [
  "George Orwell wrote 1984.",
  "George Orwell was the author of 1984.",
  "The author of 1984 was George Orwell.",
  "Orwell wrote the novel 1984.",
  "1984 was written by George Orwell.",
  "George Orwell's 1984 is a dystopian novel.",
];

for (const [name, ctx] of [
  ["CLEAN (context A)", clean],
  ["REDUNDANT (context B)", redundant],
] as [string, string[]][]) {
  const r = profileContext({ query: Q, passages: ctx });
  console.log(`\n=== ${name} ===`);
  console.log(
    `D=${r.features.competitors} S=${pct(r.features.competitorSimilarity)} ` +
      `I=${r.features.interference.toFixed(2)} pred=${pct(r.predictedSuccess)} risk=${r.risk}`,
  );
  console.log("per-passage:");
  for (const p of r.passages) {
    console.log(
      `  ${p.id} relevance=${pct(p.relevance)} competition=${pct(p.competition)} ` +
        `keep=${p.keep} risk=${p.risk}`,
    );
  }
  console.log("pairs:", r.competingPairs.map((p) => `${p.a}-${p.b} ${pct(p.similarity)}`).join(", ") || "(none)");
  console.log(
    `optimizer: removed=${r.optimized.removed} kept=${r.optimized.passages.length}`,
  );
}