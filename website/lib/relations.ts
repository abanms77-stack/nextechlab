/**
 * Pair-relationship classification.
 *
 * A single binary "competes" flag cannot represent the different ways two
 * passages relate. This module extracts the ANSWER a passage gives to the
 * query (when the query matches a known shape) so pairs can be classified as:
 *
 *   support     - compatible answers, different evidence
 *   redundant   - compatible answers, near-duplicate evidence
 *   contradict  - INCOMPATIBLE answers to the same question
 *   related     - on-topic but not answering the same slot
 *   irrelevant  - not about the query
 *
 * Only `redundant` and `contradict` are treated as interference.
 *
 * SCOPE (honest): `extractAnswer` is a deliberately small, deterministic
 * pattern library covering common question shapes. It is NOT a universal QA
 * extractor. When the query shape is unknown, classification falls back to
 * similarity alone and can never report a contradiction.
 */

import { normalize, stemSet, type PairRelation } from "./profile";

/**
 * Similarity at/above which two passages are treated as near-duplicates even
 * when answer equality cannot be confirmed. Deliberately high: we prefer to
 * call a pair "support" than to wrongly prune useful evidence.
 */
const NEAR_DUPLICATE_FLOOR = 0.8;

/** Key used to compare two answers (order-independent, stemmed). */
export function answerKey(text: string): string {
  return [...stemSet(text)].sort().join(" ");
}

/**
 * Same referent? Exact key match, or one key's tokens are a subset of the
 * other ("orwell" vs "george orwell"), which happens with names.
 */
export function sameAnswer(a: string, b: string): boolean {
  if (a === b) return true;
  const A = new Set(a.split(" ").filter(Boolean));
  const B = new Set(b.split(" ").filter(Boolean));
  if (!A.size || !B.size) return false;
  const subset = (x: Set<string>, y: Set<string>) =>
    [...x].every((t) => y.has(t));
  return subset(A, B) || subset(B, A);
}

type QueryShape =
  | { kind: "author"; entity: string }
  | { kind: "attribute"; attr: string; entity: string }
  | { kind: "definition"; entity: string }
  | { kind: "password"; slot: string };

function stripArticle(s: string): string {
  return s.replace(/^(?:the|a|an)\s+/, "").trim();
}

/** Recognise the question shape. Returns null for unsupported shapes. */
export function parseQuery(query: string): QueryShape | null {
  const q = normalize(query);

  // "who wrote X" / "who authored X"  (drop a leading "novel"/"book"/"film")
  let m = q.match(/^(?:who\s+)?(?:wrote|authored|writes)\s+(.+)$/);
  if (m) {
    return {
      kind: "author",
      entity: stripArticle(m[1]).replace(/^(?:novel|book|film|story)\s+/, "").trim(),
    };
  }

  // "who is the author of X" / "the author of X"
  m = q.match(/^(?:who\s+(?:is|was)\s+)?(?:the\s+)?author\s+of\s+(.+)$/);
  if (m) {
    return {
      kind: "author",
      entity: stripArticle(m[1]).replace(/^(?:novel|book|film|story)\s+/, "").trim(),
    };
  }

  // "what is the <attr> of <entity>"   e.g. "what is the capital of France"
  m = q.match(/^what\s+(?:is|was|are|were)\s+the\s+(.+?)\s+of\s+(.+)$/);
  if (m) return { kind: "attribute", attr: m[1].trim(), entity: stripArticle(m[2]) };

  // "what is X"  (definitional)
  const defM = q.match(/^what\s+(?:is|was|are|were)\s+(.+)$/);
  if (defM) {
    const slotM = defM[1].match(/^(?:the\s+)?(.+?)\s+password$/);
    if (slotM) {
      return { kind: "password", slot: stripArticle(slotM[1]) };
    }
    return { kind: "definition", entity: stripArticle(defM[1]) };
  }

  return null;
}

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Extract the answer a passage gives to the query, or null if this passage
 * does not answer it. Keys are order-independent stemmed token sets.
 */
export function extractAnswer(
  shape: QueryShape | null,
  passage: string,
): string | null {
  if (!shape) return null;
  const p = normalize(passage);

  if (shape.kind === "author") {
    const x = esc(shape.entity);
    // allow an optional article / "novel|book|film|story" before the entity
    const nx = `(?:the\\s+)?(?:(?:novel|book|film|story)\\s+)?${x}`;
    const pats = [
      new RegExp(`^(.+?)\\s+wrote\\s+${nx}$`),
      new RegExp(`^(.+?)\\s+(?:was|is)\\s+(?:the\\s+)?author\\s+of\\s+${nx}$`),
      new RegExp(`^(?:the\\s+)?author\\s+of\\s+${nx}\\s+(?:was|is)\\s+(.+)$`),
      new RegExp(`^${nx}\\s+(?:was|is)\\s+(?:written|authored)\\s+by\\s+(.+)$`),
    ];
    for (const re of pats) {
      const m = p.match(re);
      if (m?.[1]) return answerKey(m[1]);
    }
    return null;
  }

  if (shape.kind === "attribute") {
    const a = esc(shape.attr);
    const x = esc(shape.entity);
    const pats = [
      // "the capital of Australia is Canberra"  -> object is the answer
      new RegExp(`^the\\s+${a}\\s+of\\s+(?:the\\s+)?${x}\\s+(?:is|was)\\s+(.+)$`),
      // "Australia's capital is Canberra"        -> object is the answer
      new RegExp(`^${x}(?:\\s+s)?\\s+${a}\\s+(?:is|was)\\s+(.+)$`),
      // "Canberra is the capital of Australia"  -> SUBJECT is the answer
      new RegExp(`^(.+?)\\s+(?:is|was)\\s+the\\s+${a}\\s+of\\s+(?:the\\s+)?${x}$`),
    ];
    for (const re of pats) {
      const m = p.match(re);
      if (m?.[1]) return answerKey(m[1]);
    }
    return null;
  }

  if (shape.kind === "password") {
    const slotPats = [
      new RegExp(`\\b${esc(shape.slot)}\\s+password\\s+(?:is|was)\\s+(.+)$`),
      /^(?:note|important)?\s*:?\s*(?:the\s+)?(.+?)\s+password\s+(?:is|was)\s+(.+)$/,
    ];
    for (const re of slotPats) {
      const m = p.match(re);
      const raw = m?.[m.length - 1];
      if (raw) {
        const v = cleanPasswordValue(raw);
        if (v) return answerKey(v);
      }
    }
    return null;
  }

  // definitional: "<entity> is <answer>". The entity may appear anywhere in
  // the sentence ("Important: the vault password is apple"), so search for it
  // rather than anchoring at position 0.
  const x = esc(shape.entity);
  const defs = [
    new RegExp(`\\b${x}\\s+(?:is|was|are|were)\\s+(.+)$`),
    new RegExp(`^(?:note|important)\\s*:?\\s*(?:the\\s+)?${x}\\s+(?:is|was)\\s+(.+)$`),
  ];
  for (const re of defs) {
    const m = p.match(re);
    if (m?.[1]) {
      // ignore "is discussed/described/mentioned" style non-answers
      if (/^(?:described|discussed|mentioned|noted|listed|explained)\b/.test(m[1])) {
        continue;
      }
      return answerKey(cleanPasswordValue(m[1]) || m[1]);
    }
  }
  // Any "<something> password is <value>" statement also asserts a password
  // candidate: competing password slots interfere, even when the slot noun
  // differs ("storage password" vs "vault password"). This mirrors the
  // research's medium-similarity decoys.
  if (shape.kind === "definition" || shape.kind === "password") {
    const pm = p.match(/^(?:note|important)?\s*:?\s*(?:the\s+)?(.+?)\s+password\s+(?:is|was)\s+(.+)$/);
    if (pm?.[1] && pm?.[2]) {
      const v = cleanPasswordValue(pm[2]);
      if (v) return answerKey(v);
    }
  }
  return null;
}

/**
 * Remove trailing subordinate clauses ("; confirm it in the vault log",
 * ", which ...") so answer values compare by their head phrase.
 */
function cleanPasswordValue(s: string): string {
  return s
    .split(/[;:,]/)[0]
    .replace(/\b(?:confirm|check|verify|see|refer to|in the|for the)\b.*$/, "")
    .trim();
}

/**
 * Classify the relationship between two passages.
 * Conservative: never reports "contradict" unless both answers were extracted.
 */
export function classifyPair(
  shape: QueryShape | null,
  a: string,
  b: string,
  relevanceA: number,
  relevanceB: number,
  similarity: number,
  competitionFloor: number,
): { relation: PairRelation; answerA: string | null; answerB: string | null } {
  const minRel = Math.min(relevanceA, relevanceB);
  if (minRel === 0) {
    return { relation: "irrelevant", answerA: null, answerB: null };
  }

  const answerA = extractAnswer(shape, a);
  const answerB = extractAnswer(shape, b);

  // Case 1: BOTH answers known -> we can reason about compatibility.
  if (answerA && answerB) {
    if (sameAnswer(answerA, answerB)) {
      return {
        relation: similarity >= 0.55 ? "redundant" : "support",
        answerA,
        answerB,
      };
    }
    return { relation: "contradict", answerA, answerB };
  }

  // Case 2: equality cannot be confirmed. Be conservative:
  //   - only a NEAR-DUPLICATE is called "redundant"
  //   - otherwise a similar, on-topic pair is "support" (healthy, kept)
  if (similarity >= NEAR_DUPLICATE_FLOOR) {
    return { relation: "redundant", answerA, answerB };
  }
  if (similarity >= competitionFloor) {
    return { relation: "support", answerA, answerB };
  }
  return { relation: "related", answerA, answerB };
}