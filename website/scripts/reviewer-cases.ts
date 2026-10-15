import { profileContext } from "../lib/profiler";

const join = (a: string[]) => a.join("\n\n");

// Reviewer's Test 7: Canberra support must NOT compete
const r7 = profileContext({
  query: "What is the capital of Australia?",
  passages: [
    "Canberra is the capital of Australia.",
    "Canberra was selected as Australia's capital in 1908.",
  ],
});
// Reviewer's Test 3: Orwell vs Huxley -> CONTRADICT, publication -> RELATED
const r3 = profileContext({
  query: "Who wrote the novel 1984?",
  passages: [
    "George Orwell wrote 1984.",
    "Aldous Huxley wrote 1984.",
    "The novel was published in 1949.",
  ],
});
console.log("TEST7 capital support:");
console.log(`  D=${r7.features.competitors} S=${Math.round(r7.features.competitorSimilarity * 100)}% I=${r7.features.interference.toFixed(2)} removed=${r7.optimized.removed}`);
console.log(`  relations: support=${r7.features.supportingPairs} redundant=${r7.features.redundantPairs} contradict=${r7.features.contradictoryPairs}`);
console.log(`  p1 relation=${r7.passages[0].relation} competition=${Math.round(r7.passages[0].competition * 100)}% nearest=${Math.round(r7.passages[0].nearestSimilarity * 100)}%`);
console.log("TEST3 authors contradict:");
console.log(`  D=${r3.features.competitors} S=${Math.round(r3.features.competitorSimilarity * 100)}% I=${r3.features.interference.toFixed(2)} removed=${r3.optimized.removed} hasConflict=${r3.hasConflict}`);
console.log(`  relations: support=${r3.features.supportingPairs} redundant=${r3.features.redundantPairs} contradict=${r3.features.contradictoryPairs} related=${r3.features.relatedPairs}`);
for (const p of r3.competingPairs) console.log(`  pair ${p.a}<->${p.b} sim=${Math.round(p.similarity * 100)}% rel=${p.relation} answers=${p.answerA} vs ${p.answerB}`);
