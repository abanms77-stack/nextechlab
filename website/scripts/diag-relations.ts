// Relationship-taxonomy diagnostic (the reviewer's test cases).
//   npx tsx scripts/diag-relations.ts
import { profileContext } from "../lib/profiler";
import { parseQuery } from "../lib/relations";

const pct = (x: number) => (x * 100).toFixed(0) + "%";

function run(name: string, query: string, passages: string[]) {
  const r = profileContext({ query, passages });
  console.log(`\n=== ${name} ===`);
  console.log(`query="${query}"  shape=${JSON.stringify(parseQuery(query))}`);
  console.log(
    `D=${r.features.competitors} S=${pct(r.features.competitorSimilarity)} ` +
      `I=${r.features.interference.toFixed(2)} pred=${pct(r.predictedSuccess)} ` +
      `risk=${r.risk} conflict=${r.hasConflict}`,
  );
  console.log(
    `pairs: support=${r.features.supportingPairs} redundant=${r.features.redundantPairs} ` +
      `contradict=${r.features.contradictoryPairs} related=${r.features.relatedPairs}`,
  );
  for (const p of r.passages) {
    console.log(
      `  ${p.id} rel=${pct(p.relevance)} nearest=${pct(p.nearestSimilarity)} ` +
        `competition=${pct(p.competition)} relation=${p.relation} keep=${p.keep}`,
    );
  }
  for (const p of r.competingPairs) {
    console.log(
      `  pair ${p.a}-${p.b} sim=${pct(p.similarity)} relation=${p.relation}` +
        (p.answerA || p.answerB ? `  answers=[${p.answerA ?? "?"} | ${p.answerB ?? "?"}]` : ""),
    );
  }
  console.log(`optimizer: kept=${r.optimized.passages.length}/${r.features.passages} removed=${r.optimized.removed}`);
  return r;
}

// Test 7: supporting evidence must NOT compete or be pruned
run("Test 7 - supporting (Canberra)", "What is the capital of Australia?", [
  "Canberra is the capital of Australia.",
  "Canberra was selected as Australia's capital in 1908.",
]);

// Test 3: two different authors = CONTRADICTION (not 3-way competition)
run("Test 3 - contradiction (authors)", "Who wrote the novel 1984?", [
  "George Orwell wrote 1984.",
  "Aldous Huxley wrote 1984.",
  "The novel was published in 1949.",
]);

// Redundant: six paraphrases of the same claim
run("Redundant (paraphrases)", "Who wrote the novel 1984?", [
  "George Orwell wrote 1984.",
  "George Orwell was the author of 1984.",
  "The author of 1984 was George Orwell.",
  "Orwell wrote the novel 1984.",
  "1984 was written by George Orwell.",
  "George Orwell's 1984 is a dystopian novel.",
]);

// Contradiction on a numeric slot
run("Contradiction (refund windows)", "What is the refund window?", [
  "The refund window is 30 days from the date of purchase.",
  "The refund window is 14 days from the date of purchase.",
  "The refund window is 60 days from the date of purchase.",
]);

// Clean
run("Clean (single fact)", "Who wrote the novel 1984?", [
  "George Orwell wrote 1984.",
  "1984 was published in 1949.",
  "Winston Smith is the protagonist.",
]);