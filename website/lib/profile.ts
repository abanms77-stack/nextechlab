/**
 * ContextLab - public profiler engine: similarity layer, constants, types.
 *
 * Similarity is MEASURED, not hand-labelled. Two distinct quantities:
 *   relevance(q, p)   - coverage of the query's key terms by a passage
 *   similarity(a, b)  - symmetric similarity between two texts
 *                       (stemmed-token Jaccard blended with character-trigram
 *                       Jaccard, so morphology does not break it)
 *
 * The interference variable of the research is competition BETWEEN passages
 * (competing information), not merely query-to-passage matching.
 *
 * IP NOTE: only principles are exposed here. Calibrated coefficients, the
 * evaluation set, prompt templates and the optimization policy are private.
 */

export type Risk = "low" | "medium" | "high";

/**
 * A pair of passages can relate in several ways. Only "redundant" and
 * "contradict" count as interference; "support" and "related" are healthy.
 */
export type PairRelation =
  | "support"
  | "redundant"
  | "contradict"
  | "related"
  | "irrelevant";

export interface Passage {
  id: string;
  index: number;
  text: string;
  preview: string;
  tokens: number;
  relevance: number; // query coverage, 0..1
  nearestSimilarity: number; // max similarity to ANY other passage (was "competition")
  competition: number; // max similarity to a COMPETING partner; 0 if none
  relation: PairRelation; // strongest relationship this passage participates in
  risk: Risk;
  keep: boolean;
}

export interface ProfileFeatures {
  passages: number;
  tokens: number;
  queryTokens: number;
  relevanceMean: number;
  relevantPassages: number;
  competitors: number; // D - passages inside a competing cluster
  competitorSimilarity: number; // S - mean similarity of competing pairs
  interference: number; // I = min(1, D/5) * S
  lengthFactor: number;
  // relationship counts (interference = redundant + contradictory only)
  supportingPairs: number;
  redundantPairs: number;
  contradictoryPairs: number;
  relatedPairs: number;
}

export interface CompetingPair {
  a: string;
  b: string;
  similarity: number; // raw measured passage↔passage similarity (research S)
  relevanceA: number; // query coverage of a
  relevanceB: number; // query coverage of b
  strength: number; // similarity x min(relevance)
  relation: PairRelation; // support | redundant | contradict | related | irrelevant
  answerA: string | null; // extracted answer key (when the query shape allows)
  answerB: string | null;
}

/** Two passages that give incompatible answers to the same question. */
export interface Conflict {
  a: string;
  b: string;
  answerA: string;
  answerB: string;
  similarity: number;
}

/** Optional per-passage / per-pair diagnostics (only requested by ?debug=1). */
export interface Diagnostics {
  queryTokens: string[];
  passages: {
    id: string;
    tokens: string[];
    relevance: number;
    nearestSimilarity: number;
    competition: number;
    relation: PairRelation;
  }[];
  pairs: {
    a: string;
    b: string;
    similarity: number;
    relevanceA: number;
    relevanceB: number;
    strength: number;
    relation: PairRelation;
    answerA: string | null;
    answerB: string | null;
    passesCompetitionFloor: boolean;
    passesCandidateFilter: boolean;
    competes: boolean;
  }[];
}

export interface ProfileResult {
  features: ProfileFeatures;
  similarityDistribution: { bucket: string; count: number }[];
  competingPairs: CompetingPair[];
  risk: Risk;
  predictedSuccess: number;
  confidence: number;
  tokensPct: number;
  passages: Passage[];
  highRiskIds: string[];
  recommendations: string[];
  optimized: { passages: Passage[]; tokens: number; removed: number };
  conflicts: Conflict[];
  hasConflict: boolean;
  engineVersion: string;
  /** D in this product = passages in a competing cluster. NOTE: this is a
   *  product metric and is NOT numerically identical to the experimental D
   *  (which counted decoy passages in a controlled benchmark). */
  diagnostics?: Diagnostics;
}

// --- internal, calibrated constants (private to the product) ----------------
const COMPETITION_FLOOR = 0.35;
const RELEVANCE_FLOOR = 0.15;
const COMPETITION_WEIGHT = 0.6;
const D_SATURATION = 5;
const LENGTH_SHORT = 256;
const LENGTH_LONG = 8192;
const W_INTERFERENCE = 5.5;
const W_LENGTH = 0.5;
const INTERCEPT = 2.8;
const ENGINE_VERSION = "profiler-2.1";

export function estimateTokens(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words * 1.33));
}

const STOP = new Set(
  "a an the and or of to in is are was were be been being for on at by with from as that this these those it its what which who whom how when where why do does did can could should would may might will shall not no".split(
    " ",
  ),
);

export function normalize(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function rawTokens(s: string): string[] {
  return (normalize(s).match(/[a-z0-9][a-z0-9]*/g) ?? []).filter(
    (w) => w.length > 1 && !STOP.has(w),
  );
}

/** Light suffix stripping so "refunds"/"refund" and "windows"/"window" match. */
function stem(w: string): string {
  for (const suf of ["ing", "edly", "ed", "es", "s"]) {
    if (w.length > suf.length + 2 && w.endsWith(suf)) {
      return w.slice(0, -suf.length);
    }
  }
  return w;
}

export function stemSet(s: string): Set<string> {
  return new Set(rawTokens(s).map(stem));
}

function trigrams(s: string): Set<string> {
  const x = " " + normalize(s) + " ";
  const out = new Set<string>();
  for (let i = 0; i + 3 <= x.length; i++) out.add(x.slice(i, i + 3));
  return out;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}

/** Symmetric measured similarity in [0, 1] between two texts. */
export function textSimilarity(a: string, b: string): number {
  const tok = jaccard(stemSet(a), stemSet(b));
  const tri = jaccard(trigrams(a), trigrams(b));
  return Math.min(1, COMPETITION_WEIGHT * tok + (1 - COMPETITION_WEIGHT) * tri);
}

/** Stemmed tokens of a text (exposed for diagnostics only). */
export function tokenStems(text: string): string[] {
  return [...stemSet(text)].sort();
}

/** Fraction of the query's key terms that appear in the passage, in [0, 1]. */
export function queryCoverage(query: string, passage: string): number {
  const q = stemSet(query);
  if (!q.size) return 0;
  const p = stemSet(passage);
  let hit = 0;
  for (const t of q) if (p.has(t)) hit++;
  return hit / q.size;
}

export function bucket(sim: number): string {
  if (sim < 0.1) return "0-10%";
  if (sim < 0.25) return "10-25%";
  if (sim < 0.4) return "25-40%";
  if (sim < 0.55) return "40-55%";
  if (sim < 0.7) return "55-70%";
  return "70-100%";
}

export const riskOf = (p: number): Risk =>
  p >= 0.8 ? "low" : p >= 0.55 ? "medium" : "high";

export {
  COMPETITION_FLOOR,
  RELEVANCE_FLOOR,
  D_SATURATION,
  LENGTH_SHORT,
  LENGTH_LONG,
  W_INTERFERENCE,
  W_LENGTH,
  INTERCEPT,
  ENGINE_VERSION,
};