"""Real retriever: MiniLM embeddings + FAISS over the local Wikipedia corpus.

This is a genuine RAG retrieval stage, not a toy: the corpus is real wiki text,
chunked into paragraphs, embedded with all-MiniLM-L6-v2, and searched with an
exact FAISS inner-product index. Gold fact passages + competing decoys are
injected into the corpus, so retrieval surfaces real competition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Retriever:
    def __init__(self, embedder: str, chunk_words: int = 80,
                 stride: int = 80, max_chunks: int = 20000):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedder)
        self.chunk_words = chunk_words
        self.stride = stride
        self.max_chunks = max_chunks
        self.texts: List[str] = []
        self._index = None

    # ------------------------------------------------------------------
    def _chunk(self, text: str) -> List[str]:
        words = text.split()
        out = []
        for start in range(0, len(words), self.stride):
            piece = words[start:start + self.chunk_words]
            if len(piece) >= 20:
                out.append(" ".join(piece))
            if len(out) >= self.max_chunks:
                break
        return out

    def build(self, corpus_path: Path, extra_facts: List[str]):
        raw = corpus_path.read_text(encoding="utf-8", errors="ignore")
        self.texts = self._chunk(raw)
        # inject gold facts + decoys at the FRONT so they are retrievable
        self.texts = list(extra_facts) + self.texts
        emb = self.model.encode(self.texts, batch_size=64,
                                show_progress_bar=False,
                                normalize_embeddings=True,
                                convert_to_numpy=True)
        self._emb = np.asarray(emb, dtype=np.float32)
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self._emb.shape[1])
            self._index.add(self._emb)
        except Exception:
            self._index = None  # numpy fallback
        return self

    def search(self, query: str, k: int):
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            idx, scores = idx[0], scores[0]
        else:
            sims = (self._emb @ q[0])
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]
        out = []
        for i, s in zip(idx.tolist(), scores.tolist()):
            if i < 0:
                continue
            out.append({"index": int(i), "text": self.texts[i],
                        "retrieval_score": float(s)})
        return out
