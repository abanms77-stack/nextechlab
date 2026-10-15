// Invariant + adversarial sweep for the ContextLab profiler.
//
// The engine guarantees:  D === 0  =>  interference === 0  =>  predictedSuccess >= ~0.90
// (because z = INTERCEPT - W_INTERFERENCE*I - W_LENGTH*len, and I = 0 leaves
//  z >= 2.8 - 0.5 = 2.3, sigmoid(2.3) = 0.909).
//
// So a report of "low predicted success WITH D = 0" would be a contradiction.
// This script searches for any such case and prints the engine version, so a
// stale build can be detected.

const PORT = process.env.PORT ?? "3000";
const API = `http://localhost:${PORT}/api/profile`;

async function profile(query, passages) {
  const res = await fetch(API, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, passages }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

const R = "The refund window is 30 days from the date of purchase.";
const UNREL = [
  "The weather in London was unusually cold that winter.",
  "A bakery opened near the station last month.",
  "The orchestra rehearsed late into the evening.",
  "Ticket sales rose sharply after the announcement.",
];
const COMP = [
  "The refund window is 14 days from the date of purchase.",
  "The refund window is 60 days from the date of purchase.",
  "The refund window is 7 days from the date of purchase.",
  "The refund window is 21 days from the date of purchase.",
  "The refund window is 45 days from the date of purchase.",
  "The refund window is 90 days from the date of purchase.",
];

// Build a wide set of adversarial cases, including the reported "D=0" shape.
const cases = [];
cases.push(["1 passage (single)", "What is the refund window?", [R]]);
cases.push(["1 passage + 4 unrelated", "What is the refund window?", [R, ...UNREL]]);
for (let k = 1; k <= 6; k++) {
  cases.push([
    `1 + ${k} competing`,
    "What is the refund window?",
    [R, ...COMP.slice(0, k)],
  ]);
}
cases.push(["only unrelated", "What is the refund window?", [...UNREL]]);
cases.push([
  "empty-query-ish",
  "zzzqqq",
  [R, ...COMP],
]);
cases.push([
  "very long single passage",
  "What is the refund window?",
  [R + " " + R + " " + R + " " + R + " " + R],
]);
cases.push([
  "identical duplicates x5",
  "What is the refund window?",
  [R, R, R, R, R],
]);
cases.push([
  "unrelated-only x6 (long)",
  "What is the refund window?",
  [...UNREL, ...UNREL, ...UNREL],
]);

let violations = 0;
let version = null;

console.log(`Engine invariant sweep -> ${API}\n`);
for (const [name, q, passages] of cases) {
  const j = await profile(q, passages);
  version = version ?? j.engineVersion;
  const f = j.features;
  const bad = f.competitors === 0 && j.predictedSuccess < 0.85;
  if (bad) violations++;
  console.log(
    `${bad ? "VIOLATION" : "ok       "}  ${name.padEnd(28)} ` +
      `n=${String(f.passages).padStart(2)} D=${String(f.competitors).padStart(2)} ` +
      `S=${(f.competitorSimilarity * 100).toFixed(0).padStart(3)}% ` +
      `I=${f.interference.toFixed(2)} len=${f.lengthFactor.toFixed(2)} ` +
      `pred=${(j.predictedSuccess * 100).toFixed(0)}% risk=${j.risk}`,
  );
}

console.log(`\nengineVersion = ${version}`);
console.log(
  violations === 0
    ? `INVARIANT HOLDS: no case had D=0 with predictedSuccess < 85%.`
    : `INVARIANT BROKEN: ${violations} case(s) contradict D=0.`,
);
process.exit(violations ? 1 : 0);