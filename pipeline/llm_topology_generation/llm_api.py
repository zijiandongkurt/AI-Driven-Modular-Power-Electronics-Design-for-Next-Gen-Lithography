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
import json

from .llm_engine_minimal import LLMEngine, SNELLIUS_HF_CACHE, DEFAULT_MODEL_ID
from .prompt_input import load_constraints, make_prompt, make_prompt_demo, slug
from .net_writer import write_netlists, get_llm_output_dir
from transformers import TextStreamer

def _save_prompt(prompt: str, batchID: str) -> None:
    """Save the prompt used for generation to data/<batchID>/prompt.txt."""
    batch_dir = get_llm_output_dir(batchID).parent  # data/<batchID>/
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

def _save_raw_output(raw_outputs: list[str], batchID: str) -> None:
    """Save the raw LLM completions to data/<batchID>/raw_output.txt."""
    batch_dir = get_llm_output_dir(batchID).parent
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = batch_dir / "raw_output.txt"
    # Use append mode to handle cases where multiple constraints are in one batch
    with output_path.open("a", encoding="utf-8") as f:
        for i, raw in enumerate(raw_outputs):
            f.write(f"\n{'='*60}\n")
            f.write(f" CANDIDATE {i+1}\n")
            f.write(f"{'='*60}\n\n")
            f.write(raw)
            f.write("\n")

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

    def _generate(self, prompt: str, n: int) -> list[dict[str, str]]:
        """
        Generate `n` completions and return both raw and cleaned versions.

        Args:
            prompt: The formatted input string for the LLM.
            n: Number of candidates to generate.

        Returns:
            A list of dictionaries, each containing:
                - 'raw': The full decoded string from the model.
                - 'cleaned': The SPICE netlist extracted via engine._clean().
        """
        tok = self.engine.tokenizer
        model = self.engine.model
        ids = tok(prompt, return_tensors="pt")
        ids = {k: v.to(model.device) for k, v in ids.items()}

        streamer = TextStreamer(tok, skip_prompt=True) #

        results: list[dict[str, str]] = []
        for i in range(n):
            if n > 1:
                print(f"\n--- Generating Candidate {i+1}/{n} ---")
            
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.engine._temperature,
                    top_p=self.engine._top_p,
                    pad_token_id=tok.pad_token_id,
                    streamer=streamer,
                )
            
            gen_ids = out[0][ids["input_ids"].shape[1]:]
            raw = tok.decode(gen_ids, skip_special_tokens=True)
            results.append({
                "raw": raw,
                "cleaned": self.engine._clean(raw) #
            })
        return results

    def generate_from_constraint(self, constraint: dict, n: int = 4) -> list[str]:
        """
        Apply the constraint template and return n cleaned netlists.
        
        Note: Raw output is not saved to disk in this standalone method.
        """
        prompt = make_prompt(constraint) #
        res = self._generate(prompt, n)
        return [r["cleaned"] for r in res]

    def generate_for_batch(
        self,
        constraint: dict,
        batchID: str,
        n: int = 4,
        DEMO: bool = False,
        previous_batch_id: str = None
    ) -> list[Path]:
        """
        Generate n netlists for a batch and archive the raw model output.

        Files saved to data/<batchID>/:
            - prompt.txt: The full prompt sent to the model.
            - raw_output.txt: The unedited responses from the model.
            - llm_output/*.net: The validated/cleaned netlists.
        """
        if DEMO and previous_batch_id:
            feedback = self._aggregate_previous_batch_data(previous_batch_id) #
            prompt = make_prompt_demo(constraint, feedback) #
        else:
            prompt = make_prompt(constraint) #

        _save_prompt(prompt, batchID) #

        label = slug(constraint, 0) #
        results = self._generate(prompt, n)
        
        # Save the raw text for the user to inspect later
        _save_raw_output([r["raw"] for r in results], batchID)

        return write_netlists(
            netlists=[r["cleaned"] for r in results],
            constraint=constraint,
            label=label,
            batchID=batchID,
        ) #

    def generate_from_json(self, json_path: str | Path, batchID: str, n: int = 4) -> list[Path]:
        """
        Process constraints from a JSON file and archive raw results for the batch.

        Saves raw outputs to data/<batchID>/raw_output.txt for every completion.
        """
        constraints = load_constraints(json_path) #

        written: list[Path] = []
        for i, c in enumerate(constraints):
            prompt = make_prompt(c) #
            if i == 0:
                _save_prompt(prompt, batchID) #
            
            label = slug(c, i) #
            results = self._generate(prompt, n)
            
            _save_raw_output([r["raw"] for r in results], batchID)
            
            paths = write_netlists(
                netlists=[r["cleaned"] for r in results],
                constraint=c,
                label=label,
                batchID=batchID,
            ) #
            written.extend(paths)
        return written

    def generate_from_text(self, prompt: str, n: int = 1) -> list[str]:
        """Generate from a raw text prompt and return cleaned netlists."""
        res = self._generate(prompt, n)
        return [r["cleaned"] for r in res]
    
    def _aggregate_previous_batch_data(self, previous_batch_id: str) -> str:
        """
        Reads reward_results.json and validation_results.json from the previous batch 
        to format a detailed feedback string.
        """
        if not previous_batch_id:
            return ""

        data_dir = Path("pipeline/data")
        batch_path = data_dir / previous_batch_id
        reward_file = batch_path / "reward_results.json"
        validation_file = batch_path / "validation_results.json"

        if not reward_file.exists():
            return f"\n[System Note: Previous batch data '{previous_batch_id}' not found.]\n"

        try:
            # Load rewards for ranking and general status
            with open(reward_file, 'r') as f:
                reward_data = json.load(f)
            
            # Load validation results for specific failure details
            val_data = {}
            if validation_file.exists():
                with open(validation_file, 'r') as f:
                    val_data = json.load(f)
            
            circuits = reward_data.get("circuits", {})
            if not circuits:
                return "\n[System Note: Previous batch had no circuit data.]\n"

            # Sort by fitness score
            sorted_circuits = sorted(
                circuits.items(), 
                key=lambda item: item[1].get("fitness_score", -9999.0), 
                reverse=True
            )

            prompt_addition = f"\n\n=== FEEDBACK FROM PREVIOUS ITERATION ({previous_batch_id.split('/')[-1]}) ===\n"
            prompt_addition += "Use this feedback to improve your next designs. Higher fitness scores are better.\n\n"

            for rank, (circuit_name, details) in enumerate(sorted_circuits):
                score = details.get("fitness_score", "N/A")
                source = details.get("source", "unknown")
                
                prompt_addition += f"Rank {rank+1}: Topology '{circuit_name}'\n"
                prompt_addition += f"  - Status: {source}\n"
                prompt_addition += f"  - Fitness Score: {score if isinstance(score, str) else f'{score:.4f}'}\n"

                # Case 1: Simulation success
                if source == "simulation" and "raw_metrics" in details:
                    m = details["raw_metrics"]
                    prompt_addition += f"  - Performance: V_out={m.get('simulation_output_voltage')}V, Eff={m.get('efficiency')}\n"
                
                # Case 2: Validation Failure - pull specific tests from validation_results.json
                elif source == "validation_penalty":
                    # Lookup this specific circuit in the validation file
                    circuit_val = val_data.get(circuit_name, {})
                    
                    # A test is failed if its value is 0 or False
                    failed_tests = [test for test, val in circuit_val.items() if val == 0 or val is False]
                    
                    if failed_tests:
                        prompt_addition += f"  - FAILED TESTS: {', '.join(failed_tests)}\n"
                        prompt_addition += "  - FIX: Ensure your netlist addresses these specific violations.\n"
                    else:
                        prompt_addition += "  - Note: Circuit failed basic structural/syntax checks.\n"
                
                prompt_addition += "\n"

            return prompt_addition

        except Exception as e:
            return f"\n[System Note: Error reading feedback data: {e}]\n"


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