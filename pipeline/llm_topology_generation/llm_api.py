"""
llm_api.py — Packaged Python interface for the topology-generation LLM.

This is a thin wrapper over LLMEngine + prompt_input + net_writer that
exposes three high-level methods:

    - generate_from_constraint(constraint, n)  →  list[str]
    - generate_from_json(json_path, out_dir, n) →  list[Path]
    - generate_from_text(prompt, n)            →  list[str]

All callers should reuse the same `TopologyLLM` instance because model
loading is slow and Qwen3-14B consumes ~28GB of GPU memory.

Snellius usage:
    from llm_api import TopologyLLM
    llm = TopologyLLM()   # loads Qwen3-14B from shared cache, no download
    cands = llm.generate_from_constraint(
        {"vin": 12, "vout_target": 5, "efficiency_target": 0.9}, n=4
    )
    for c in cands: print(c)
"""

from __future__ import annotations

from pathlib import Path

import torch

from .llm_engine_minimal import LLMEngine, SNELLIUS_HF_CACHE, DEFAULT_MODEL_ID
from .prompt_input import load_constraints, make_prompt, slug
from .net_writer import write_netlists, get_llm_output_dir


def _save_prompt(prompt: str, batchID: str) -> None:
    """Save the prompt used for generation to data/<batchID>/prompt.txt."""
    batch_dir = get_llm_output_dir(batchID).parent  # data/<batchID>/
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "prompt.txt").write_text(prompt, encoding="utf-8")


class TopologyLLM:
    """High-level interface around LLMEngine."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str | None = None,  # None = full bfloat16 (fits on A100 80GB)
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_cache_dir: str = SNELLIUS_HF_CACHE,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.engine = LLMEngine(
            model_id=model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            hf_cache_dir=hf_cache_dir,
        )

    # ── Internal generation primitive ──────────────────────────────────

    def _generate(self, prompt: str, n: int) -> list[str]:
        """Generate `n` cleaned completions for a raw prompt string."""
        tok = self.engine.tokenizer
        model = self.engine.model
        ids = tok(prompt, return_tensors="pt")
        ids = {k: v.to(model.device) for k, v in ids.items()}

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
            constraint: dict with keys vin, vout_target, efficiency_target,
                        power_out_w (and optional _comment).
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
        """Generate n netlists for a constraint and write to data/<batchID>/llm_output/.

        Also saves the prompt to data/<batchID>/prompt.txt for traceability.

        Args:
            constraint: dict with constraint keys.
            batchID:    Batch identifier — files land in data/<batchID>/llm_output/.
            n:          Number of candidate netlists to generate.

        Returns:
            List of written .net file paths.
        """
        prompt = make_prompt(constraint)
        _save_prompt(prompt, batchID)

        label = slug(constraint, 0)
        cands = self._generate(prompt, n)
        return write_netlists(
            netlists=cands,
            constraint=constraint,
            label=label,
            batchID=batchID,
        )

    def generate_from_json(
        self,
        json_path: str | Path,
        batchID: str,
        n: int = 4,
    ) -> list[Path]:
        """Process every constraint in a JSON file, writing to data/<batchID>/llm_output/.

        Saves the prompt of the first constraint to data/<batchID>/prompt.txt.

        Args:
            json_path:  Path to a JSON file containing a list of constraint dicts.
            batchID:    Batch identifier — files land in data/<batchID>/llm_output/.
            n:          Number of candidates per constraint.

        Returns:
            Flat list of all written .net file paths.
        """
        constraints = load_constraints(json_path)

        written: list[Path] = []
        for i, c in enumerate(constraints):
            prompt = make_prompt(c)
            if i == 0:
                # Save once per batch — all constraints share the same template
                _save_prompt(prompt, batchID)
            label = slug(c, i)
            cands = self._generate(prompt, n)
            paths = write_netlists(
                netlists=cands,
                constraint=c,
                label=label,
                batchID=batchID,
            )
            written.extend(paths)
        return written

    def generate_from_text(self, prompt: str, n: int = 1) -> list[str]:
        """Generate from a raw text prompt — no template applied.

        Use this when you want full manual control over the prompt
        (e.g. trying a different system message, debugging, etc.).
        """
        return self._generate(prompt, n)


# ── Convenience singleton (lazy-loaded) ────────────────────────────────

_singleton: TopologyLLM | None = None


def get_llm(**kwargs) -> TopologyLLM:
    """Return a process-wide singleton TopologyLLM.

    Subsequent calls ignore kwargs and return the already-loaded instance.
    """
    global _singleton
    if _singleton is None:
        _singleton = TopologyLLM(**kwargs)
    return _singleton


if __name__ == "__main__":
    # Quick smoke test — verify model loads from Snellius cache
    llm = TopologyLLM()
    out = llm.generate_from_text("### Hello world:\n", n=1)
    print(out[0][:200])