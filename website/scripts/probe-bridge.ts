import { parseQuery, extractAnswer, classifyPair } from "../lib/relations";
import { queryCoverage, textSimilarity } from "../lib/profile";
const Q = "What is the vault password?";
console.log("shape:", JSON.stringify(parseQuery(Q)));
const needle = "Important: the vault password is apple.";
const med = "Note: the storage password is river.";
const high = "Important: the vault password is river; confirm it in the vault log.";
for (const [name, p] of [["needle", needle],["med", med],["high", high]] as const) {
  console.log(name, "rel=", queryCoverage(Q, p).toFixed(2));
}
console.log("sim needle-med =", textSimilarity(needle, med).toFixed(3));
console.log("sim needle-high =", textSimilarity(needle, high).toFixed(3));
const shape = parseQuery(Q);
console.log("ans needle:", extractAnswer(shape, needle), "ans med:", extractAnswer(shape, med), "ans high:", extractAnswer(shape, high));
for (const [name, a, b] of [["needle-med", needle, med],["needle-high", needle, high]] as const) {
  const ra = queryCoverage(Q, a), rb = queryCoverage(Q, b), s = textSimilarity(a, b);
  console.log(name, JSON.stringify(classifyPair(shape, a, b, ra, rb, s, 0.35)));
}
