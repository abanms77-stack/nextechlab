// Probe the live profiler with the reviewer's example and variants.
const API = "http://localhost:3000/api/profile";

const cases = [
  {
    name: "A. single relevant passage",
    query: "Who wrote the novel 1984?",
    context: "George Orwell wrote the novel 1984.",
  },
  {
    name: "B. relevant + unrelated",
    query: "Who wrote the novel 1984?",
    context:
      "George Orwell wrote the novel 1984.\n\nThe weather in London was cold that winter.\n\nA new bakery opened near the station.",
  },
  {
    name: "C. high competition (authors)",
    query: "Who wrote the novel 1984?",
    context: [
      "George Orwell wrote the novel 1984.",
      "Aldous Huxley wrote the novel 1984.",
      "Ray Bradbury wrote the novel 1984.",
      "Ernest Hemingway wrote the novel 1984.",
      "Mark Twain wrote the novel 1984.",
    ].join("\n\n"),
  },
  {
    name: "D. refund simple",
    query: "What is the refund window?",
    context: "Refunds are available within 30 days of purchase.",
  },
  {
    name: "E. refund competing",
    query: "What is the refund window?",
    context:
      "The refund window is 30 days.\n\nThe refund window is 14 days.\n\nThe refund window is 60 days.",
  },
];

for (const c of cases) {
  const res = await fetch(API, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query: c.query, context: c.context }),
  });
  const j = await res.json();
  const pct = (x) => (x * 100).toFixed(0) + "%";
  const f = j.features;
  console.log(
    `${c.name}\n  risk=${j.risk} pred=${pct(j.predictedSuccess)} ` +
      `D(competitors)=${f.competitors}/${f.passages} ` +
      `S(competitor sim)=${pct(f.competitorSimilarity)} ` +
      `I(interference)=${f.interference.toFixed(2)}`,
  );
  console.log(
    "  per-passage: " +
      j.passages
        .map((p) => `${p.id} rel=${pct(p.relevance)} comp=${pct(p.competition)}`)
        .join(" | "),
  );
  if (j.competingPairs?.length)
    console.log(
      "  competing pairs: " +
        j.competingPairs.map((p) => `${p.a}-${p.b} ${pct(p.similarity)}`).join(", "),
    );
}