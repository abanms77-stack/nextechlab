import { classifyPair, parseQuery } from "./relations";
import {
  COMPETITION_FLOOR,
  CompetingPair,
  Conflict,
  PairRelation,
  D_SATURATION,
  ENGINE_VERSION,
  INTERCEPT,
  LENGTH_LONG,
  LENGTH_SHORT,
  Passage,
  ProfileResult,
  RELEVANCE_FLOOR,
  Risk,
  W_INTERFERENCE,
  W_LENGTH,
  bucket,
  estimateTokens,
  queryCoverage,
  riskOf,
  textSimilarity,
  tokenStems,
} from "./profile";

export type {
  Passage,
  ProfileResult,
  ProfileFeatures,
  Risk,
  CompetingPair,
} from "./profile";
export {
  textSimilarity,
  queryCoverage,
  estimateTokens,
  bucket,
  riskOf,
  tokenStems,
} from "./profile";

/** Union-find over competing pairs to find clusters of competing passages. */
function clusters(n: number, pairs: [number, number][]): number[] {
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number =>
    parent[x] === x ? x : (parent[x] = find(parent[x]));
  for (const [a, b] of pairs) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  }
  return parent.map((_, i) => find(i));
}

export function profileContext(opts: {
  query: string;
  passages: string[];
  modelWindow?: number;
  debug?: boolean;
}): ProfileResult {
  const modelWindow = opts.modelWindow ?? 128000;
  const query = opts.query.trim();
  const passages = opts.passages.map((s) => s.trim()).filter(Boolean);
  const n = passages.length;

  // 1. relevance of each passage to the query
  const relevance = passages.map((p) => queryCoverage(query, p));
  const toks = passages.map(estimateTokens);
  const queryTokens = estimateTokens(query);
  const tokens = toks.reduce((a, b) => a + b, 0) + queryTokens;

  // 2. passage <-> passage similarity + RELATIONSHIP classification.
  //
  // A pair is interference ONLY when it is classified "redundant" or
  // "contradict". Merely being similar and on-topic ("support"/"related") is
  // healthy evidence and must not be treated as competition.
  const shape = parseQuery(query);
  const sim: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const relOf: PairRelation[][] = Array.from({ length: n }, () =>
    new Array(n).fill("irrelevant" as PairRelation),
  );
  const competingPairIdx: [number, number][] = [];
  const relPairIdx: [number, number][] = [];
  const competingPairs: CompetingPair[] = [];
  const conflicts: Conflict[] = [];
  const relCount = { support: 0, redundant: 0, contradict: 0, related: 0, irrelevant: 0 };
  const pairDiag: NonNullable<ProfileResult["diagnostics"]>["pairs"] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const s = textSimilarity(passages[i], passages[j]);
      const relW = Math.min(relevance[i], relevance[j]);
      const st = s * relW;
      sim[i][j] = sim[j][i] = s;
      const cls = classifyPair(
        shape,
        passages[i],
        passages[j],
        relevance[i],
        relevance[j],
        s,
        COMPETITION_FLOOR,
      );
      relOf[i][j] = relOf[j][i] = cls.relation;
      relCount[cls.relation]++;

      const passesCompetitionFloor = s >= COMPETITION_FLOOR;
      const passesCandidateFilter = relW > 0;
      // LAYER 1 (research interference, preserved): a pair competes ONLY when
      // it is measured-similar, both passages are on-topic, AND the relation
      // is redundant or contradictory. Merely similar + on-topic
      // ("support"/"related") is healthy evidence and must NOT inflate D x S.
      const isRelationalPre =
        (cls.relation === "redundant" && passesCompetitionFloor) ||
        cls.relation === "contradict";
      const competes = passesCandidateFilter && isRelationalPre;

      // LAYER 2 (logical relations): drives the OPTIMIZER and the conflict
      // flag, and is deliberately independent of D x S.
      const isRelational = isRelationalPre;
      if (isRelational) relPairIdx.push([i, j]);

      const pair: CompetingPair = {
        a: `p${i + 1}`,
        b: `p${j + 1}`,
        similarity: s,
        relevanceA: relevance[i],
        relevanceB: relevance[j],
        strength: st,
        relation: cls.relation,
        answerA: cls.answerA,
        answerB: cls.answerB,
      };
      if (competes) {
        competingPairIdx.push([i, j]);
        competingPairs.push(pair);
      }
      if (cls.relation === "contradict" && cls.answerA && cls.answerB) {
        conflicts.push({
          a: pair.a,
          b: pair.b,
          answerA: cls.answerA,
          answerB: cls.answerB,
          similarity: s,
        });
      }
      pairDiag.push({
        ...pair,
        passesCompetitionFloor,
        passesCandidateFilter,
        competes,
      });
    }
  }

  // 3. D = passages inside a competing cluster; S = mean competing similarity
  const clusterId = clusters(n, competingPairIdx);
  const inCluster = new Set(competingPairIdx.flat());
  const D = inCluster.size;
  const S = competingPairs.length
    ? competingPairs.reduce((a, p) => a + p.similarity, 0) / competingPairs.length
    : 0;

  // 4. interference = normalised competitor count x competitor similarity
  const interference = Math.min(1, D / D_SATURATION) * S;
  const lengthFactor = Math.max(
    0,
    Math.min(
      1,
      Math.log2(Math.max(tokens, LENGTH_SHORT) / LENGTH_SHORT) /
        Math.log2(LENGTH_LONG / LENGTH_SHORT),
    ),
  );
  const z = INTERCEPT - W_INTERFERENCE * interference - W_LENGTH * lengthFactor;
  const predictedSuccess = 1 / (1 + Math.exp(-z));

  // 5. per-passage views
  const nearestSimilarityOf = (i: number) =>
    n > 1 ? Math.max(...sim[i].map((v, j) => (j === i ? 0 : v))) : 0;

  const strongestRelation = (i: number): PairRelation => {
    const order: PairRelation[] = [
      "contradict", "redundant", "support", "related", "irrelevant",
    ];
    for (const r of order) {
      for (let j = 0; j < n; j++) if (j !== i && relOf[i][j] === r) return r;
    }
    return "irrelevant";
  };

  // competition = max similarity to a partner whose RELATION is redundant or
  // contradictory (0 if none). Deliberately different from nearestSimilarity,
  // which is the raw max and can be high for purely supporting evidence.
  const competitionOf = (i: number) => {
    let best = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const r = relOf[i][j];
      if (r === "redundant" || r === "contradict") {
        best = Math.max(best, sim[i][j]);
      }
    }
    return best;
  };

  const passageObjs: Passage[] = passages.map((text, i) => {
    const rel = strongestRelation(i);
    const pRisk: Risk =
      rel === "contradict"
        ? "high"
        : rel === "redundant"
          ? interference >= 0.6
            ? "high"
            : "medium"
          : "low";
    return {
      id: `p${i + 1}`,
      index: i,
      text,
      preview: text.length > 140 ? text.slice(0, 140) + "…" : text,
      tokens: toks[i],
      relevance: relevance[i],
      nearestSimilarity: nearestSimilarityOf(i),
      competition: competitionOf(i),
      relation: rel,
      risk: pRisk,
      keep: true,
    };
  });

  // 6. optimization.
  //    RULE 1: if no competition is detected, KEEP EVERYTHING. Low query
  //    relevance is NOT by itself a reason to delete context.
  //    RULE 2: when competition exists, keep the most query-relevant member
  //    of each competing cluster, and keep every passage outside a cluster.
  const keep = new Set<number>();
  const relClusterId = clusters(n, relPairIdx);
  const inRelCluster = new Set(relPairIdx.flat());
  const groups = new Map<number, number[]>();
  for (const i of inRelCluster) {
    const c = relClusterId[i];
    if (!groups.has(c)) groups.set(c, []);
    groups.get(c)!.push(i);
  }
  if (inRelCluster.size === 0) {
    // no redundant or contradictory pairs -> context is left unchanged
    for (let i = 0; i < n; i++) keep.add(i);
  } else {
    for (const members of groups.values()) {
      // A group containing a CONTRADICTION keeps every member: we must not
      // silently delete one side of a factual conflict based on relevance.
      const hasContradiction = members.some((m) =>
        members.some((o) => o !== m && relOf[m][o] === "contradict"),
      );
      if (hasContradiction) {
        for (const m of members) keep.add(m);
      } else {
        // redundant/support group: keep the strongest evidence
        let best = members[0];
        for (const m of members) if (relevance[m] > relevance[best]) best = m;
        keep.add(best);
      }
    }
    for (let i = 0; i < n; i++) {
      if (!inRelCluster.has(i)) keep.add(i);
    }
  }
  for (const p of passageObjs) p.keep = keep.has(p.index);

  const kept = passageObjs.filter((p) => p.keep);
  const optimizedTokens = kept.reduce((a, p) => a + p.tokens, 0) + queryTokens;

  // 7. distribution of pairwise competition similarities
  const dist = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const b = bucket(sim[i][j]);
      dist.set(b, (dist.get(b) ?? 0) + 1);
    }
  }

  const highRisk = passageObjs.filter((p) => p.risk === "high");
  const relevantPassages = relevance.filter((r) => r >= RELEVANCE_FLOOR).length;

  const removed = passageObjs.length - kept.length;
  const rec: string[] = [];

  // contradiction first: it is a correctness signal, not a pruning decision
  if (conflicts.length > 0) {
    rec.push(
      `\u26a0 ${conflicts.length} passage pair${conflicts.length > 1 ? "s" : ""} provide CONFLICTING answers. Verification is recommended before optimization.`,
    );
    for (const c of conflicts.slice(0, 3)) {
      rec.push(`   ${c.a} says "${c.answerA}" while ${c.b} says "${c.answerB}".`);
    }
  }
  if (relCount.redundant > 0) {
    rec.push(
      `${relCount.redundant} pair${relCount.redundant > 1 ? "s" : ""} contain overlapping (redundant) evidence.`,
    );
  }
  if (relCount.support + relCount.related > 0) {
    rec.push(
      `${relCount.support + relCount.related} pair${relCount.support + relCount.related > 1 ? "s" : ""} support or merely relate to the query and were kept.`,
    );
  }

  if (inRelCluster.size === 0) {
    rec.push("No conflicting or redundant passage pairs were detected.");
    rec.push("Context left unchanged (0 passages pruned).");
    if (relevantPassages > 0) {
      rec.push(
        "Estimated retrieval reliability is high; no optimization needed.",
      );
    }
  } else {
    rec.push(
      `${inRelCluster.size} of ${n} passages form ${groups.size} redundant/contradictory group${groups.size > 1 ? "s" : ""}.`,
    );
    if (removed > 0) {
      rec.push(
        `Kept the strongest evidence in each redundancy group: ${kept.length} kept, ${removed} redundant passage${removed > 1 ? "s" : ""} pruned.`,
      );
    } else {
      rec.push(
        "Every passage was kept because the group contains a contradiction (see the warning above).",
      );
    }
  }
  if (D > 0) {
    rec.push(
      `Interference (research D x S): ${D} of ${n} passages compete at mean similarity ${Math.round(S * 100)}% (I = ${interference.toFixed(2)}).`,
    );
  }
  if (relevantPassages === 0) {
    rec.push(
      "No passage covers the query's terms — retrieval may fail for a different reason (missing evidence). This is not an interference problem.",
    );
  }
  if (tokens / modelWindow > 0.6) {
    rec.push(
      `Context uses ${Math.round((tokens / modelWindow) * 100)}% of the model window; length is a smaller, secondary factor once interference is present.`,
    );
  }
  if (predictedSuccess < 0.8 && D > 0) {
    rec.push(
      "Place the surviving evidence at the start or end of the context — mid-context evidence is the least reliable.",
    );
  }

  return {
    features: {
      passages: n,
      tokens,
      queryTokens,
      relevanceMean: relevance.length
        ? relevance.reduce((a, b) => a + b, 0) / relevance.length
        : 0,
      relevantPassages,
      competitors: D,
      competitorSimilarity: S,
      interference,
      lengthFactor,
      supportingPairs: relCount.support,
      redundantPairs: relCount.redundant,
      contradictoryPairs: relCount.contradict,
      relatedPairs: relCount.related,
    },
    similarityDistribution: [...dist.entries()]
      .map(([b, count]) => ({ bucket: b, count }))
      .sort((a, b) => a.bucket.localeCompare(b.bucket)),
    competingPairs,
    risk: riskOf(predictedSuccess),
    predictedSuccess,
    confidence: Math.max(
      0.35,
      Math.min(1, 0.4 + 0.4 * Math.min(1, n / 6) + 0.2 * Math.min(1, tokens / 2000)),
    ),
    tokensPct: (tokens / modelWindow) * 100,
    passages: passageObjs,
    highRiskIds: highRisk.map((p) => p.id),
    recommendations: rec,
    optimized: {
      passages: kept,
      tokens: optimizedTokens,
      removed,
    },
    conflicts,
    hasConflict: conflicts.length > 0,
    engineVersion: ENGINE_VERSION,
    ...(opts.debug
      ? {
          diagnostics: {
            queryTokens: tokenStems(query),
            passages: passageObjs.map((p) => ({
              id: p.id,
              tokens: tokenStems(p.text),
              relevance: p.relevance,
              nearestSimilarity: p.nearestSimilarity,
              competition: p.competition,
              relation: p.relation,
            })),
            pairs: pairDiag,
          },
        }
      : {}),
  };
}

/** Parse a raw pasted context into candidate passages (blank-line separated). */
export function splitPassages(raw: string): string[] {
  return raw
    .split(/\n\s*\n+/)
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter((s) => s.length > 0);
}