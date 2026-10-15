"""Local Hugging Face causal-LM adapter for ContextLab.

IMPORTANT (measured on this machine, i5-12450H, 16 GB RAM, no GPU):
    * Loading in the model's *default* dtype (bfloat16) is ~5-100x slower on
      CPU because this CPU has no native BF16 support. We therefore force
      float32, which uses the fast MKL/AVX2 path.
    * attn_implementation="sdpa" is faster than "eager" on CPU as well.

Scoring is restricted to a fixed candidate token set at the final position,
giving a clean K-way classification for needle-retrieval tasks.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class LocalHFModel:
    def __init__(
        self,
        name: str,
        threads: int = 12,
        dtype: str = "float32",
        attn: str = "sdpa",
        max_context: int = 8192,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self._torch = torch
        self.name = name
        self.max_context = max_context

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        torch_dtype = getattr(torch, dtype)

        # transformers renamed torch_dtype -> dtype; support both.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch_dtype, attn_implementation=attn
            ).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch_dtype, attn_implementation=attn
            ).eval()

    # ------------------------------------------------------------------
    def next_token_logits(self, input_ids: List[int]):
        torch = self._torch
        ids = torch.tensor([input_ids], dtype=torch.long)
        with torch.inference_mode():
            out = self.model(ids)
        return out.logits[0, -1]

    def score_candidates(
        self, input_ids: List[int], candidate_ids: List[int]
    ) -> Dict[int, float]:
        """Return softmax probability mass over the candidate tokens only."""
        torch = self._torch
        logits = self.next_token_logits(input_ids)
        cand = torch.tensor(candidate_ids, dtype=torch.long)
        probs = torch.softmax(logits[cand], dim=-1)
        return {int(c): float(p) for c, p in zip(candidate_ids, probs.tolist())}

    def predict(
        self, input_ids: List[int], candidate_ids: List[int], answer_id: int
    ):
        """Return (correct, prob_correct, argmax_id, probs_dict)."""
        probs = self.score_candidates(input_ids, candidate_ids)
        argmax_id = max(probs, key=probs.get)
        return (
            int(argmax_id == answer_id),
            float(probs.get(answer_id, 0.0)),
            int(argmax_id),
            probs,
        )
