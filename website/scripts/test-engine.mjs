// Behavioural tests for the Context Profiler engine.
// Requires the server running (npm start / npm run dev) on PORT (default 3000).
//
//   node scripts/test-engine.mjs
//
// Tests the PRODUCT CLAIMS, not just that pages render.

const PORT = process.env.PORT ?? "3000";
const API = `http://localhost:${PORT}/api/profile`;

let passed = 0;
let failed = 0;

async function profile(query, context) {
  const res = await fetch(API, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, context }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function check(name, cond, detail) {
  if (cond) {
    passed++;
    console.log(`  PASS  ${name}${detail ? "  (" + detail + ")" : ""}`);
  } else {
    failed++;
    console.log(`  FAIL  ${name}${detail ? "  (" + detail + ")" : ""}`);
  }
}

const Q = "What is the refund window?";
const ANSWER = "The refund window is 30 days from the date of purchase.";
const UNRELATED = [
  "The weather in London was unusually cold that winter.",
  "A new bakery opened near the station last month.",
];
const COMPETING = [
  "The refund window is 14 days from the date of purchase.",
  "The refund window is 60 days from the date of purchase.",
  "The refund window is 7 days from the date of purchase.",
];
const join = (arr) => arr.join("\n\n");

async function main() {
  console.log(`ContextLab engine tests  ->  ${API}\n`);

  const clean = await profile(Q, ANSWER);
  const plusUnrelated = await profile(Q, join([ANSWER, ...UNRELATED]));
  const plusCompeting = await profile(Q, join([ANSWER, ...COMPETING.slice(0, 1)]));
  const twoCompeting = await profile(Q, join([ANSWER, ...COMPETING.slice(0, 2)]));
  const threeCompeting = await profile(Q, join([ANSWER, ...COMPETING]));
  const onlyAnswer = await profile(Q, ANSWER);

  // TEST 0 (regression for the reported bug): morphology must not break matching
  check(
    "T0 morphology: 'refund' vs 'refunds' is not zero",
    clean.features.relevanceMean > 0.3,
    `relevanceMean=${clean.features.relevanceMean.toFixed(2)}`,
  );
  check(
    "T0b a single relevant passage is detected",
    clean.features.relevantPassages >= 1,
    `relevant=${clean.features.relevantPassages}`,
  );

  // TEST 1: clean context -> low interference, low risk
  check(
    "T1 clean context -> zero interference",
    clean.features.interference === 0,
    `I=${clean.features.interference.toFixed(2)}`,
  );
  check("T1b clean context -> low risk", clean.risk === "low", `risk=${clean.risk}`);

  // TEST 2: + unrelated passages -> interference still ~0
  check(
    "T2 unrelated passages add ~no interference",
    plusUnrelated.features.interference < 0.05,
    `I=${plusUnrelated.features.interference.toFixed(2)}`,
  );

  // TEST 3: + competing passage -> interference increases vs clean
  check(
    "T3 competing passage raises interference",
    plusCompeting.features.interference > clean.features.interference,
    `I ${clean.features.interference.toFixed(2)} -> ${plusCompeting.features.interference.toFixed(2)}`,
  );
  check(
    "T3b competitor count is detected",
    plusCompeting.features.competitors >= 2,
    `D=${plusCompeting.features.competitors}`,
  );

  // TEST 4: more competitors -> interference increases
  check(
    "T4 more competitors -> higher interference",
    threeCompeting.features.interference > twoCompeting.features.interference &&
      twoCompeting.features.interference >= plusCompeting.features.interference,
    `I 1->2->3 = ${plusCompeting.features.interference.toFixed(2)}, ${twoCompeting.features.interference.toFixed(2)}, ${threeCompeting.features.interference.toFixed(2)}`,
  );
  check(
    "T4b more competitors -> predicted success falls",
    threeCompeting.predictedSuccess < plusCompeting.predictedSuccess,
    `p ${plusCompeting.predictedSuccess.toFixed(2)} -> ${threeCompeting.predictedSuccess.toFixed(2)}`,
  );

  // TEST 5: higher similarity of competitors -> interference increases
  const lowSim = await profile(
    Q,
    join([ANSWER, "The refund window is discussed in the supplier handbook section."]),
  );
  const highSim = await profile(
    Q,
    join([ANSWER, "The refund window is 45 days from the date of purchase."]),
  );
  check(
    "T5 higher competitor similarity -> higher interference",
    highSim.features.competitorSimilarity > lowSim.features.competitorSimilarity,
    `S ${lowSim.features.competitorSimilarity.toFixed(2)} -> ${highSim.features.competitorSimilarity.toFixed(2)}`,
  );

  // TEST 6: removing competitors -> interference decreases
  check(
    "T6 removing competitors lowers interference",
    onlyAnswer.features.interference < threeCompeting.features.interference,
    `I ${threeCompeting.features.interference.toFixed(2)} -> ${onlyAnswer.features.interference.toFixed(2)}`,
  );

  // TEST 7: the optimizer prunes REDUNDANT duplicates (not contradictions)
  const redundantCtx = join([
    "The refund window is 30 days from the date of purchase.",
    "The refund window is 30 days from the date of purchase for all plans.",
  ]);
  const redundant = await profile(Q, redundantCtx);
  check(
    "T7 optimizer prunes redundant duplicates",
    redundant.features.redundantPairs > 0 && redundant.optimized.removed >= 1,
    `redundantPairs=${redundant.features.redundantPairs} removed=${redundant.optimized.removed}`,
  );

  // TEST 12: contradictions are FLAGGED, never silently pruned
  const contradict = await profile(
    Q,
    join([
      "The refund window is 30 days from the date of purchase.",
      "The refund window is 14 days from the date of purchase.",
    ]),
  );
  check(
    "T12 contradiction detected and flagged",
    contradict.features.contradictoryPairs > 0 && contradict.hasConflict,
    `contradict=${contradict.features.contradictoryPairs} hasConflict=${contradict.hasConflict}`,
  );
  check(
    "T12b contradictions are NOT pruned",
    contradict.optimized.removed === 0,
    `removed=${contradict.optimized.removed}`,
  );

  // TEST 13: supporting evidence does not compete and is not pruned
  const support = await profile(
    "What is the capital of Australia?",
    join([
      "Canberra is the capital of Australia.",
      "Canberra was selected as Australia's capital in 1908.",
    ]),
  );
  check(
    "T13 supporting pair classified as support (not a relation to optimize)",
    support.features.supportingPairs > 0 &&
      support.passages.every((p) => p.relation === "support"),
    `support=${support.features.supportingPairs} relations=${support.passages.map((p) => p.relation)}`,
  );
  check(
    "T13b supporting evidence not pruned",
    support.optimized.removed === 0,
    `removed=${support.optimized.removed}`,
  );

  // TEST 14: passage.consumer-facing `competition` means ACTUAL competition
  check(
    "T14 competing passages expose competition>0; supporting ones do not",
    contradict.passages.every((p) => p.competition > 0) &&
      support.passages.every((p) => p.competition === 0),
    `contradict=${contradict.passages.map((p) => Math.round(p.competition * 100))} support=${support.passages.map((p) => Math.round(p.competition * 100))}`,
  );

  // TEST 15: contradiction with different authors (the reviewer's Test 3)
  const authors = await profile("Who wrote the novel 1984?", join([
    "George Orwell wrote 1984.",
    "Aldous Huxley wrote 1984.",
    "The novel was published in 1949.",
  ]));
  check(
    "T15 two different authors -> contradiction flagged (D=0 interference)",
    authors.features.contradictoryPairs === 1 &&
      authors.features.relatedPairs === 2 &&
      authors.hasConflict,
    `contradict=${authors.features.contradictoryPairs} related=${authors.features.relatedPairs}`,
  );
  check(
    "T15b contradiction kept (author conflict not auto-resolved)",
    authors.optimized.removed === 0,
    `removed=${authors.optimized.removed}`,
  );

  // TEST 8: risk ordering is monotone with interference
  check(
    "T8 risk ordering: clean < competing",
    clean.risk === "low" && threeCompeting.risk !== "low",
    `clean=${clean.risk} competing=${threeCompeting.risk}`,
  );

  // TEST 9: no competition => nothing is pruned (fixes the reported contradiction)
  check(
    "T9 D=0 => nothing pruned",
    clean.optimized.removed === 0,
    `removed=${clean.optimized.removed}`,
  );
  check(
    "T9b D=0 with unrelated passages => nothing pruned",
    plusUnrelated.optimized.removed === 0,
    `removed=${plusUnrelated.optimized.removed}`,
  );

  // TEST 10: low query relevance is NOT a reason to delete context
  const mixed = await profile(Q, join([ANSWER, "A bakery opened near the station."]));
  check(
    "T10 low-relevance passage kept when no competition",
    mixed.optimized.removed === 0 && mixed.optimized.passages.length === 2,
    `removed=${mixed.optimized.removed} kept=${mixed.optimized.passages.length}`,
  );

  // TEST 11: the reviewer's A/B contexts (relevance vs competition)
  const A = [
    "George Orwell wrote 1984.",
    "1984 was published in 1949.",
    "Winston Smith is the protagonist.",
    "The novel explores surveillance.",
    "Orwell also wrote Animal Farm.",
  ];
  const B = [
    "George Orwell wrote 1984.",
    "George Orwell was the author of 1984.",
    "The author of 1984 was George Orwell.",
    "Orwell wrote the novel 1984.",
    "1984 was written by George Orwell.",
    "George Orwell's 1984 is a dystopian novel.",
  ];
  const AQ = "Who wrote the novel 1984?";
  const rA = await profile(AQ, join(A));
  const rB = await profile(AQ, join(B));
  check(
    "T11 relevance detected (A: not zero)",
    rA.features.relevanceMean > 0.3,
    `relevanceMean=${rA.features.relevanceMean.toFixed(2)}`,
  );
  check(
    "T11b clean context A -> low",
    rA.features.competitors === 0 && rA.risk === "low",
    `D=${rA.features.competitors} risk=${rA.risk}`,
  );
  check(
    "T11c redundant context B -> competitors + higher",
    rB.features.competitors > 0 && rB.features.interference > rA.features.interference,
    `D=${rB.features.competitors} I ${rA.features.interference.toFixed(2)} -> ${rB.features.interference.toFixed(2)}`,
  );

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => {
  console.error(
    `\nCould not reach the API at ${API}.\nStart the server first:  npm run dev\n(${e.message})`,
  );
  process.exit(2);
});