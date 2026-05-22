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
import re

from .llm_engine_minimal import LLMEngine, SNELLIUS_HF_CACHE, DEFAULT_MODEL_ID
from .prompt_input import load_constraints, make_prompt, make_prompt_demo, slug
from .net_writer import write_netlists, get_llm_output_dir
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

def _save_prompt(prompt: str, batchID: str, identifier: str = None) -> None:
    """Save the prompt used for generation."""
    batch_dir = get_llm_output_dir(batchID).parent
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Name the file based on the identifier (e.g., prompt_parent_00_Step_Down_cand1.txt)
    filename = f"prompt_{identifier}.txt" if identifier else "prompt.txt"
    
    (batch_dir / filename).write_text(prompt, encoding="utf-8")

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

class StopAtEndCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last 10 tokens to check if the model just typed .end
        tail_text = self.tokenizer.decode(input_ids[0][-10:], skip_special_tokens=True).lower()
        if ".end" in tail_text or ". end" in tail_text:
            return True
        return False
    

def normalize_netlist(netlist: str) -> str:
    """Strips comments and normalizes whitespace to detect cheating."""
    lines = netlist.strip().split('\n')
    cleaned = []
    for line in lines:
        line = line.strip().lower()
        if line.startswith('*') or not line:
            continue
        # Normalize multiple spaces to a single space
        cleaned.append(' '.join(line.split()))
    return '\n'.join(cleaned)

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

    def _generate(self, prompt: str, n: int, temp_override: float = None) -> list[dict[str, str]]:
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

        streamer = TextStreamer(tok, skip_prompt=True)
        stop_criteria = StoppingCriteriaList([StopAtEndCriteria(tok)])

        # Use the override if provided, otherwise default to engine's temperature
        gen_temp = temp_override if temp_override is not None else self.engine._temperature

        results: list[dict[str, str]] = []
        for i in range(n):
            if n > 1:
                print(f"\n--- Generating Candidate {i+1}/{n} ---")
            
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=gen_temp,
                    top_p=self.engine._top_p,
                    pad_token_id=tok.pad_token_id,
                    streamer=streamer,
                    stopping_criteria=stop_criteria
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
        prompt = make_prompt(constraint)
        res = self._generate(prompt, n)
        return [r["cleaned"] for r in res]

    def generate_for_batch(
        self,
        constraint: dict,
        batchID: str,
        n: int = 4,
        DEMO: bool = False,
        previous_batch_id: str = None,
        label: str = None
    ) -> list[Path]:
        if label is None:
            label = slug(constraint, 0)
        results = []
        
        for i in range(1, n + 1):
            if DEMO and previous_batch_id:
                feedback = self._aggregate_previous_batch_data(previous_batch_id, label, i)
                prompt = make_prompt_demo(constraint, feedback)
                
                # --- BULLETPROOF PREVIOUS NETLIST EXTRACTION ---
                prev_clean = ""
                # Grabs everything between "Submitted Netlist:" and ".end"
                match = re.search(r'Submitted Netlist:\n(.*?\.end)', feedback, re.IGNORECASE | re.DOTALL)
                if match:
                    prev_clean = normalize_netlist(match.group(1))
                # -----------------------------------------------
            else:
                prompt = make_prompt(constraint)
                prev_clean = ""

            _save_prompt(prompt, batchID, identifier=f"cand{i}")

            # --- THE RETRY BOUNCER ---
            max_retries = 3
            attempts = 0
            current_temp = self.engine._temperature
            current_prompt = prompt
            
            while attempts <= max_retries:
                res = self._generate(current_prompt, 1, temp_override=current_temp)[0]
                new_clean = normalize_netlist(res["cleaned"])
                
                if prev_clean and new_clean == prev_clean:
                    attempts += 1
                    if attempts <= max_retries:
                        print(f"\n[!] Duplicate netlist detected for candidate {i}. Rejecting and retrying (Attempt {attempts}/{max_retries})...")
                        current_temp += 0.15  # Bump temperature to force creativity
                        current_prompt += "\n\n[SYSTEM WARNING]: FATAL ERROR. You just outputted the EXACT SAME circuit as last time. You MUST change the component values, duty cycle, or topology to proceed. DO NOT REPEAT YOURSELF."
                    else:
                        print(f"\n[!] Max retries reached for candidate {i}. Accepting duplicate.")
                        results.append(res)
                        break
                else:
                    results.append(res)
                    break
            # --------------------------
            
        _save_raw_output([r["raw"] for r in results], batchID)

        return write_netlists(
            netlists=[r["cleaned"] for r in results],
            constraint=constraint,
            label=label,
            batchID=batchID,
        )
    

    def generate_from_states(
        self,
        constraint: dict,
        batchID: str,
        seed_states: list[dict],
        candidates_per_state: int = 4,
        label: str = None
    ) -> dict[str, str]:
        """
        Generate multiple candidate branches from a set of explicit seed states.
        
        Args:
            constraint: The active constraint set dictionary.
            batchID: The identifier for the current batch.
            seed_states: List of dicts from NetlistDatabase containing 'id', 'netlist_text', 'fitness', and 'feedback'.
            candidates_per_state: Number of branches to spawn per seed state.
            
        Returns:
            dict: Mapping of {new_candidate_id: parent_id} to track lineage depth.
        """
        if label is None:
            label = slug(constraint, 0)
        all_results = []       # Holds dicts of {"raw": ..., "cleaned": ...}
        parent_tracking = []   # Tracks the parent_id for each generated candidate
        
        cand_counter = 1
        
        for state in seed_states:
            parent_id = state["id"]
            prev_clean = normalize_netlist(state["netlist_text"])
            
            # Construct the exact feedback block expected by the LLM
            feedback_block = "\n\n=== FEEDBACK FROM PREVIOUS ITERATION ===\n"
            feedback_block += "Review your previous attempt below. Identify the failures and generate a new, corrected netlist. Higher fitness scores are better.\n\n"
            feedback_block += f"Topology '{parent_id}'\n"
            feedback_block += f"  - Fitness Score: {state['fitness']:.4f}\n"
            
            feedback_block += "  - Submitted Netlist:\n"
            for line in state["netlist_text"].strip().split('\n'):
                feedback_block += f"      {line}\n"
                
            feedback_block += "\n  - Performance / Errors:\n"
            for line in state["feedback"].strip().split('\n'):
                if line.strip():  # Avoid printing empty bullet points
                    feedback_block += f"      * {line}\n"
                
            feedback_block += "  - CRITICAL RULE: You MUST NOT output the exact same netlist. You must adjust your topology, timing parameters (PULSE), or component values to improve the score.\n"
            
            # Create the full prompt
            prompt = make_prompt_demo(constraint, feedback_block)
            
            print(f"\nBranching {candidates_per_state} candidates from parent state: {parent_id} (Fitness: {state['fitness']:.4f})")
            
            _save_prompt(prompt, batchID, identifier=f"parent_{parent_id}")

            for _ in range(candidates_per_state):
                
                # --- THE RETRY BOUNCER ---
                max_retries = 3
                attempts = 0
                current_temp = self.engine._temperature
                current_prompt = prompt
                
                while attempts <= max_retries:
                    res = self._generate(current_prompt, 1, temp_override=current_temp)[0]
                    new_clean = normalize_netlist(res["cleaned"])
                    
                    if prev_clean and new_clean == prev_clean:
                        attempts += 1
                        if attempts <= max_retries:
                            print(f"\n[!] Duplicate netlist detected (parent: {parent_id}). Rejecting and retrying (Attempt {attempts}/{max_retries})...")
                            current_temp += 0.15  # Bump temperature to force creativity
                            current_prompt += "\n\n[SYSTEM WARNING]: FATAL ERROR. You outputted the EXACT SAME circuit as the parent netlist. You MUST change the component values, duty cycle, or topology to proceed."
                        else:
                            print(f"\n[!] Max retries reached for branch of {parent_id}. Accepting duplicate.")
                            all_results.append(res)
                            parent_tracking.append(parent_id)
                            break
                    else:
                        all_results.append(res)
                        parent_tracking.append(parent_id)
                        break
                
                cand_counter += 1

        # Save all raw outputs for the batch in one go
        _save_raw_output([r["raw"] for r in all_results], batchID)

        # Write clean netlists to disk (returns list of pathlib.Path objects)
        written_paths = write_netlists(
            netlists=[r["cleaned"] for r in all_results],
            constraint=constraint,
            label=label,
            batchID=batchID,
        )
        
        # Zip the returned filenames together with their parent IDs
        parent_map = {}
        for path, p_id in zip(written_paths, parent_tracking):
            parent_map[path.stem] = p_id
            
        return parent_map

    def _aggregate_previous_batch_data(self, previous_batch_id: str, label: str, candidate_idx: int) -> str:
        """
        Reads feedback from the previous batch specifically targeting a single candidate lineage.
        Looks for the circuit named f"{label}_cand{candidate_idx}".
        """
        if not previous_batch_id:
            return ""

        data_dir = Path("pipeline/data")
        batch_path = data_dir / previous_batch_id
        reward_file = batch_path / "reward_results.json"
        validation_file = batch_path / "validation_results.json"

        prev_b_suffix = ""
        match = re.search(r'batch_(\d+)', str(previous_batch_id))
        if match:
            prev_b_suffix = f"_b{match.group(1)}"

        # Look for e.g., "00_Step_Down_cand1_b1"
        target_circuit_name = f"{label}_cand{candidate_idx}{prev_b_suffix}"

        if not reward_file.exists():
            return f"\n[System Note: Previous batch data '{previous_batch_id}' not found.]\n"

        try:
            with open(reward_file, 'r') as f:
                reward_data = json.load(f)
            
            val_data = {}
            if validation_file.exists():
                with open(validation_file, 'r') as f:
                    val_data = json.load(f)
            
            circuits = reward_data.get("circuits", {})
            active_constraints = reward_data.get("active_constraints", {})
            
            # If this specific candidate wasn't in the previous batch, skip feedback
            if target_circuit_name not in circuits:
                return f"\n[System Note: No previous iteration found for '{target_circuit_name}'.]\n"

            details = circuits[target_circuit_name]
            score = details.get("fitness_score", "N/A")
            source = details.get("source", "unknown")

            prompt_addition = f"\n\n=== FEEDBACK FROM PREVIOUS ITERATION ===\n"
            prompt_addition += f"Review your previous attempt below. Identify the failures and generate a new, corrected netlist. Higher fitness scores are better.\n\n"
            
            prompt_addition += f"Topology '{target_circuit_name}'\n"
            prompt_addition += f"  - Status: {source}\n"
            prompt_addition += f"  - Fitness Score: {score if isinstance(score, str) else f'{score:.4f}'}\n"

            # Load the actual generated netlist
            netlist_content = "[Netlist file not found]"
            netlist_path = batch_path / "LLM_output" / f"{target_circuit_name}.net"
            if netlist_path.exists():
                netlist_content = netlist_path.read_text(encoding="utf-8").strip()
            
            prompt_addition += "  - Submitted Netlist:\n"
            for line in netlist_content.split('\n'):
                prompt_addition += f"      {line}\n"

            if source == "simulation" and "raw_metrics" in details:
                m = details["raw_metrics"]
                losses = details.get("loss_breakdown", {})
                v_out = m.get('simulation_output_voltage', 0.0)
                eff = m.get('efficiency', 0.0)
                
                target_vout = active_constraints.get('vout_target', 'N/A')
                target_eff = active_constraints.get('efficiency_target', 'N/A')

                prompt_addition += f"  - Performance: V_out = {v_out:.4f}V (Target: {target_vout}V), Efficiency = {eff:.4f} (Target: {target_eff})\n"
                # 1. Inject Raw Metrics
                prompt_addition += "  - Raw Metrics:\n"
                for k, v in m.items():
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    prompt_addition += f"      * {k}: {val_str}\n"

                # 2. Inject Loss Breakdown
                if losses:
                    prompt_addition += "  - Loss Breakdown (lower is better):\n"
                    for k, v in losses.items():
                        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                        prompt_addition += f"      * {k}: {val_str}\n"

                prompt_addition += "  - CRITICAL RULE: You MUST NOT output the exact same netlist. You must adjust your topology, timing parameters (PULSE), or component values to improve the score.\n"

            elif source == "validation_penalty":
                circuit_val = val_data.get(target_circuit_name, {})
                checks = circuit_val.get("checks", {})
                failed_tests = [test for test, val in checks.items() if val == 0 or val is False]
                
                if failed_tests:
                    prompt_addition += f"  - FAILED TESTS: {', '.join(failed_tests)}\n"
                    prompt_addition += "  - FIX: Ensure your new netlist specifically resolves these violations.\n"
                else:
                    prompt_addition += "  - Note: Circuit failed basic structural/syntax checks.\n"
            
            prompt_addition += "\n"
            return prompt_addition

        except Exception as e:
            return f"\n[System Note: Error reading feedback data for '{target_circuit_name}': {e}]\n"


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