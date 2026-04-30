"""
llm_api.py — Packaged Python interface for the topology-generation LLM.

This is a thin wrapper over LLMEngine + prompt_input + net_writer that
exposes four high-level methods:

    - generate_from_constraint(constraint, n)             → list[str]
    - generate_for_batch(constraint, batchID, n)          → list[Path]
    - generate_from_json(json_path, batchID, n)           → list[Path]
    - generate_from_text(prompt, n)                       → list[str]

All callers should reuse the same `TopologyLLM` instance because model
loading takes ~12–20s and consumes ~5.5 GB VRAM.

Example:
    from pipeline.llm_topology_generation.llm_api import TopologyLLM
    from pipeline.llm_topology_generation.prompt_input import load_constraint

    llm = TopologyLLM()
    c   = load_constraint("pipeline/data/datasets/constraints.json", idx=0)
    paths = llm.generate_for_batch(c, batchID="batch_3", n=4)
    # → [pipeline/data/batch_3/llm_output/top1.net, ...]
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from .llm_engine_minimal import LLMEngine
from .prompt_input import load_constraints, make_prompt
from .net_writer import write_netlists


DEFAULT_MODEL_ID = r"D:\Document\Course\Team_intership\LLM\models\qwen25-coder-7b"


class TopologyLLM:
    """High-level interface around LLMEngine."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str = "4bit",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.engine = LLMEngine(
            model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # ── Internal generation primitive ──────────────────────────────────

    def _generate(self, prompt: str, n: int) -> list[str]:
        """Generate `n` cleaned completions for a raw prompt string."""
        tok = self.engine.tokenizer
        model = self.engine.model
        ids = tok(prompt, return_tensors="pt").to(model.device)

        results: list[str] = []
        for _ in range(n):
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.engine._temperature,
                    top_p=self.engine._top_p,
                    pad_token_id=tok.pad_token_id,
                )
            gen_ids = out[0][ids["input_ids"].shape[1]:]
            raw = tok.decode(gen_ids, skip_special_tokens=True)
            results.append(self.engine._clean(raw))
        return results

    # ── Public API ────────────────────────────────────────────────────

    def generate_from_constraint(
        self, constraint: dict, n: int = 4
    ) -> list[str]:
        """Apply the constraint→prompt template, generate n candidates.

        Args:
            constraint: dict with keys vin_min, vin_max, vout_target,
                        efficiency_target, power_in (and optional _comment).
            n:          number of candidate netlists.

        Returns:
            list of n cleaned netlist strings.
        """
        prompt = make_prompt(constraint)
        return self._generate(prompt, n)

    def generate_for_batch(
        self,
        constraint: dict,
        batchID: str,
        n: int = 4,
    ) -> list[Path]:
        """Generate n netlists for a constraint and write them as
        ``data/<batchID>/llm_output/top1.net`` ... ``topN.net``.

        This is the standard pipeline entry point — `main.py` calls this.

        Args:
            constraint: dict with constraint keys (vin_min, vin_max, …).
            batchID:    Batch identifier — files land in
                        ``pipeline/data/<batchID>/llm_output/``.
            n:          Number of candidate netlists to generate.

        Returns:
            List of absolute paths to the written .net files (in order).
        """
        cands = self.generate_from_constraint(constraint, n=n)
        return write_netlists(netlists=cands, batchID=batchID)

    def generate_from_json(
        self,
        json_path: str | Path,
        batchID: str,
        n: int = 4,
    ) -> list[Path]:
        """Process every constraint in a JSON file, writing all candidates
        sequentially as ``top1.net`` ... ``topM.net`` into
        ``data/<batchID>/llm_output/``.

        With K constraints and n candidates each, M = K*n files are
        produced. The numbering is global to the batch (constraint 0
        gets ``top1..topN``, constraint 1 gets ``topN+1..top2N``, …) so
        nothing gets overwritten.

        Args:
            json_path: Path to a JSON file containing a list of constraint dicts.
            batchID:   Batch identifier — files land in
                       ``pipeline/data/<batchID>/llm_output/``.
            n:         Number of candidates per constraint.

        Returns:
            Flat list of absolute paths to all written .net files.
        """
        constraints = load_constraints(json_path)

        written: list[Path] = []
        next_idx = 1
        for c in constraints:
            cands = self.generate_from_constraint(c, n=n)
            paths = write_netlists(
                netlists=cands,
                batchID=batchID,
                start_index=next_idx,
            )
            written.extend(paths)
            next_idx += len(cands)
        return written

    def generate_from_text(self, prompt: str, n: int = 4) -> list[str]:
        """Generate from a raw text prompt — no template applied.

        Use this when you want full manual control over the prompt
        (e.g. trying a different system message, debugging, etc.).
        """
        return self._generate(prompt, n)


# ── Convenience singleton (lazy-loaded, thread-safe) ───────────────────

_singleton: TopologyLLM | None = None
_singleton_lock = threading.Lock()


def get_llm(**kwargs) -> TopologyLLM:
    """Return a process-wide singleton TopologyLLM.

    Thread-safe: concurrent first-time callers will only load the model
    once. Subsequent calls ignore kwargs and return the cached instance.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TopologyLLM(**kwargs)
    return _singleton