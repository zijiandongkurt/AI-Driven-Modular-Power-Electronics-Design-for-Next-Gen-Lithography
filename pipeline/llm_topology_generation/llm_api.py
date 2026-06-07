"""
llm_api.py — Packaged Python interface for the topology-generation LLM.

This wrapper exposes:
    - generate_from_constraint(constraint, n)
    - generate_for_batch(...)
    - generate_grouped_for_batch(...)
    - generate_grouped_for_parent_entries(...)

Grouped GRPO note:
    For true grouped GRPO, use generate_grouped_for_batch()
    or generate_grouped_for_parent_entries().
"""

from __future__ import annotations

from pathlib import Path

import json
import re
import torch

from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

from .llm_engine_minimal import LLMEngine, SNELLIUS_HF_CACHE, DEFAULT_MODEL_ID
from .prompt_input import make_prompt, make_prompt_demo, slug
from .net_writer import write_netlists, get_llm_output_dir


def _save_prompt(
    prompt: str,
    batchID: str,
    candidate_idx: int | None = None,
    group_id: str | None = None,
) -> None:
    """Save the prompt used for generation."""
    batch_dir = get_llm_output_dir(batchID).parent
    batch_dir.mkdir(parents=True, exist_ok=True)

    if group_id is not None:
        filename = f"prompt_{group_id}.txt"
    elif candidate_idx is not None:
        filename = f"prompt_cand{candidate_idx}.txt"
    else:
        filename = "prompt.txt"

    (batch_dir / filename).write_text(prompt, encoding="utf-8")


def _save_raw_output(raw_outputs: list[str], batchID: str) -> None:
    """Save raw LLM completions to data/<batchID>/raw_output.txt."""
    batch_dir = get_llm_output_dir(batchID).parent
    batch_dir.mkdir(parents=True, exist_ok=True)

    output_path = batch_dir / "raw_output.txt"

    with output_path.open("a", encoding="utf-8") as f:
        for i, raw in enumerate(raw_outputs):
            f.write(f"\n{'=' * 60}\n")
            f.write(f" CANDIDATE {i + 1}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(raw)
            f.write("\n")


class StopAtEndCriteria(StoppingCriteria):
    """Stop generation once .end appears."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        tail_text = self.tokenizer.decode(
            input_ids[0][-10:],
            skip_special_tokens=True,
        ).lower()

        return ".end" in tail_text or ". end" in tail_text


def normalize_netlist(netlist: str) -> str:
    """Strip comments and normalize whitespace for duplicate detection."""
    lines = netlist.strip().split("\n")
    cleaned = []

    for line in lines:
        line = line.strip().lower()

        if line.startswith("*") or not line:
            continue

        cleaned.append(" ".join(line.split()))

    return "\n".join(cleaned)


class TopologyLLM:
    """High-level interface around LLMEngine."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_cache_dir: str = SNELLIUS_HF_CACHE,
        lora_path: str | None = None,   
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
        # Load LoRA adapter if provided
        if lora_path:
            print(f"Loading LoRA adapter from: {lora_path}")
            self.engine.load_adapter("grpo", lora_path)

    def _generate(
        self,
        prompt: str,
        n: int,
        temp_override: float | None = None,
    ) -> list[dict[str, str]]:
        """Generate n completions from the same prompt.

        Args:
            prompt (str): Full prompt string to feed the model.
            n (int): Number of completions to generate.
            temp_override (float | None): Override sampling temperature; uses
                engine default when None.

        Returns:
            list[dict]: Each entry has keys ``"raw"`` (raw model text) and
                ``"cleaned"`` (cleaned SPICE netlist).
        """
        tok = self.engine.tokenizer
        model = self.engine.model

        ids = tok(prompt, return_tensors="pt")
        ids = {k: v.to(model.device) for k, v in ids.items()}

        streamer = TextStreamer(tok, skip_prompt=True)
        stop_criteria = StoppingCriteriaList([StopAtEndCriteria(tok)])

        gen_temp = (
            temp_override
            if temp_override is not None
            else self.engine._temperature
        )

        results: list[dict[str, str]] = []

        for i in range(n):
            if n > 1:
                print(f"\n--- Generating Candidate {i + 1}/{n} ---")

            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=gen_temp,
                    top_p=self.engine._top_p,
                    pad_token_id=tok.pad_token_id,
                    streamer=streamer,
                    stopping_criteria=stop_criteria,
                )

            gen_ids = out[0][ids["input_ids"].shape[1]:]
            raw = tok.decode(gen_ids, skip_special_tokens=True)

            results.append({
                "raw": raw,
                "cleaned": self.engine._clean(raw),
            })

        return results

    def _generate_with_retry(
        self,
        prompt: str,
        previous_netlist: str = "",
        max_retries: int = 3,
    ) -> dict[str, str]:
        """Generate one candidate, retrying on exact duplicates.

        If previous_netlist is provided, retry when the model outputs
        the exact same normalized netlist.

        Args:
            prompt (str): Full prompt string.
            previous_netlist (str): Netlist from a prior iteration used for
                duplicate detection; empty string disables the check.
            max_retries (int): Maximum number of retry attempts before
                accepting a duplicate.

        Returns:
            dict: ``{"raw": str, "cleaned": str}`` for the accepted candidate.
        """
        previous_clean = (
            normalize_netlist(previous_netlist)
            if previous_netlist
            else ""
        )

        attempts = 0
        current_temp = self.engine._temperature
        current_prompt = prompt

        while attempts <= max_retries:
            result = self._generate(
                current_prompt,
                1,
                temp_override=current_temp,
            )[0]

            new_clean = normalize_netlist(result["cleaned"])

            if previous_clean and new_clean == previous_clean:
                attempts += 1

                if attempts <= max_retries:
                    print(
                        f"\n[!] Duplicate netlist detected. "
                        f"Retrying {attempts}/{max_retries}..."
                    )

                    current_temp += 0.15
                    current_prompt += (
                        "\n\n[SYSTEM WARNING]: You outputted the exact same circuit. "
                        "You MUST change topology, timing, duty cycle, or component values."
                    )
                else:
                    print("\n[!] Max retries reached. Accepting duplicate.")
                    return result
            else:
                return result

        return result

    def generate_from_constraint(
        self,
        constraint: dict,
        n: int = 4,
    ) -> list[str]:
        """Generate n cleaned netlists from one constraint prompt."""
        prompt = make_prompt(constraint)
        results = self._generate(prompt, n)
        return [r["cleaned"] for r in results]

    def generate_for_batch(
        self,
        constraint: dict,
        batchID: str,
        n: int = 4,
        DEMO: bool = False,
        previous_batch_id: str | None = None,
    ) -> list[Path]:
        """Generate n netlists for a batch (legacy method).

        Kept for demo/backward compatibility.
        For grouped GRPO, use generate_grouped_for_batch().

        Args:
            constraint (dict): Design constraint dict.
            batchID (str): Batch identifier string.
            n (int): Number of candidates to generate.
            DEMO (bool): Whether to inject feedback from the previous batch.
            previous_batch_id (str | None): ID of the prior batch used when
                DEMO is True.

        Returns:
            list[Path]: Paths to the written .net files.
        """
        label = slug(constraint, 0)
        results = []

        for i in range(1, n + 1):
            if DEMO and previous_batch_id:
                feedback = self._aggregate_previous_batch_data(
                    previous_batch_id=previous_batch_id,
                    label=label,
                    candidate_idx=i,
                )

                prompt = make_prompt_demo(constraint, feedback)
                previous_netlist = self._extract_submitted_netlist(feedback)
            else:
                prompt = make_prompt(constraint)
                previous_netlist = ""

            _save_prompt(prompt, batchID, candidate_idx=i)

            result = self._generate_with_retry(
                prompt=prompt,
                previous_netlist=previous_netlist,
            )

            results.append(result)

        _save_raw_output([r["raw"] for r in results], batchID)

        return write_netlists(
            netlists=[r["cleaned"] for r in results],
            constraint=constraint,
            label=label,
            batchID=batchID,
        )
    
    def generate_grouped_for_batch(
        self,
        constraint: dict,
        batchID: str,
        parent_ids: list[str],
        previous_batch_id: str | None,
        outputs_per_parent: int = 4,
        DEMO: bool = True,
    ) -> list[Path]:
        """Generate grouped candidates for GRPO from a single previous batch.

        Each selected parent produces one prompt, and each prompt generates
        multiple children.  This wrapper supports the old interface where all
        parents come from the same previous batch.

        Args:
            constraint (dict): Design constraint dict.
            batchID (str): Batch identifier for the new generation.
            parent_ids (list[str]): Netlist IDs of selected parent circuits.
            previous_batch_id (str | None): Batch ID where the parents live.
            outputs_per_parent (int): Number of children to generate per parent.
            DEMO (bool): Whether to inject per-parent feedback into the prompt.

        Returns:
            list[Path]: Paths to all written .net files.
        """

        parent_entries = [
            {
                "netlist_id": parent_id,
                "batch_id": previous_batch_id,
            }
            for parent_id in parent_ids
        ]

        return self.generate_grouped_for_parent_entries(
            constraint=constraint,
            batchID=batchID,
            parent_entries=parent_entries,
            outputs_per_parent=outputs_per_parent,
            DEMO=DEMO,
        )

    def generate_grouped_for_parent_entries(
        self,
        constraint: dict,
        batchID: str,
        parent_entries: list[dict],
        outputs_per_parent: int = 4,
        DEMO: bool = True,
    ) -> list[Path]:
        """Generate grouped candidates for GRPO tree-search from arbitrary parents.

        Supports selecting parents from different historical batches.

        Example parent_entries entry::

            {
                "netlist_id": "00_xxx_g1_cand2_b1",
                "batch_id": "Run_001/batch_1",
                "depth": 1
            }

        Args:
            constraint (dict): Design constraint dict.
            batchID (str): Batch identifier for the new generation.
            parent_entries (list[dict]): Each entry must have ``"netlist_id"``
                and ``"batch_id"``; an optional ``"depth"`` field is ignored.
            outputs_per_parent (int): Number of children to generate per parent.
            DEMO (bool): Whether to inject per-parent feedback into the prompt.

        Returns:
            list[Path]: Paths to all written .net files.
        """

        label = slug(constraint, 0)

        all_results = []
        all_names = []

        for group_idx, parent in enumerate(parent_entries, start=1):
            group_id = f"g{group_idx}"

            parent_id = parent.get("netlist_id")
            parent_batch_id = parent.get("batch_id")

            if DEMO and parent_id and parent_batch_id:
                feedback = self._aggregate_previous_batch_data_by_id(
                    previous_batch_id=parent_batch_id,
                    target_circuit_name=parent_id,
                )
                prompt = make_prompt_demo(constraint, feedback)
                previous_netlist = self._extract_submitted_netlist(feedback)
            else:
                prompt = make_prompt(constraint)
                previous_netlist = ""

            _save_prompt(prompt, batchID, group_id=group_id)

            for cand_idx in range(1, outputs_per_parent + 1):
                result = self._generate_with_retry(
                    prompt=prompt,
                    previous_netlist=previous_netlist,
                )

                all_results.append(result)

                # Example: label_g1_cand1_b2.net
                all_names.append(f"{group_id}_cand{cand_idx}")

        _save_raw_output([r["raw"] for r in all_results], batchID)

        return write_netlists(
            netlists=[r["cleaned"] for r in all_results],
            constraint=constraint,
            label=label,
            batchID=batchID,
            custom_names=all_names,
        )

    def _extract_submitted_netlist(self, feedback: str) -> str:
        """Extract the previous submitted netlist from feedback text.

        Args:
            feedback (str): Feedback string injected into the prompt.

        Returns:
            str: Netlist text up to and including ``.end``, or empty string
                if not found.
        """

        match = re.search(
            r"Submitted Netlist:\n(.*?\.end)",
            feedback,
            re.IGNORECASE | re.DOTALL,
        )

        return match.group(1) if match else ""

    def _read_json_if_exists(self, path: Path) -> dict:
        """Read JSON file if it exists, otherwise return empty dict."""

        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _aggregate_previous_batch_data_by_id(
        self,
        previous_batch_id: str,
        target_circuit_name: str,
    ) -> str:
        """Read feedback for one selected parent circuit by exact circuit id.

        Args:
            previous_batch_id (str): Batch ID containing the target circuit.
            target_circuit_name (str): Exact circuit identifier to look up.

        Returns:
            str: Formatted feedback string for injection into the prompt, or
                an empty string if no data is found.
        """

        if not previous_batch_id:
            return ""

        batch_path = Path("pipeline") / "data" / previous_batch_id
        reward_file = batch_path / "reward_results.json"
        validation_file = batch_path / "validation_results.json"

        if not reward_file.exists():
            return (
                f"\n[System Note: Previous batch data "
                f"'{previous_batch_id}' not found.]\n"
            )

        try:
            reward_data = self._read_json_if_exists(reward_file)
            val_data = self._read_json_if_exists(validation_file)

            circuits = reward_data.get("circuits", {})
            active_constraints = reward_data.get("active_constraints", {})

            if target_circuit_name not in circuits:
                return (
                    f"\n[System Note: No previous circuit found "
                    f"for '{target_circuit_name}'.]\n"
                )

            details = circuits[target_circuit_name]
            score = details.get("fitness_score", "N/A")
            source = details.get("source", "unknown")

            prompt_addition = "\n\n=== FEEDBACK FROM PREVIOUS ITERATION ===\n"
            prompt_addition += (
                "Review the selected previous attempt below. "
                "Generate a corrected or improved netlist. "
                "Higher fitness scores are better.\n\n"
            )

            prompt_addition += f"Topology '{target_circuit_name}'\n"
            prompt_addition += f"  - Status: {source}\n"
            prompt_addition += (
                f"  - Fitness Score: "
                f"{score if isinstance(score, str) else f'{score:.4f}'}\n"
            )

            netlist_content = self._load_previous_netlist(
                batch_path=batch_path,
                target_circuit_name=target_circuit_name,
            )

            prompt_addition += "  - Submitted Netlist:\n"
            for line in netlist_content.split("\n"):
                prompt_addition += f"      {line}\n"

            if source == "simulation" and "raw_metrics" in details:
                prompt_addition += self._format_simulation_feedback(
                    details=details,
                    active_constraints=active_constraints,
                )

            elif source == "validation_penalty":
                prompt_addition += self._format_validation_feedback(
                    val_data=val_data,
                    target_circuit_name=target_circuit_name,
                )

            prompt_addition += "\n"
            return prompt_addition

        except Exception as e:
            return (
                f"\n[System Note: Error reading feedback data for "
                f"'{target_circuit_name}': {e}]\n"
            )

    def _load_previous_netlist(
        self,
        batch_path: Path,
        target_circuit_name: str,
    ) -> str:
        """Load a previous netlist from disk.

        Supports both ``LLM_output`` and ``llm_output`` folder names.

        Args:
            batch_path (Path): Root directory of the target batch.
            target_circuit_name (str): Stem of the .net file to load.

        Returns:
            str: Netlist text, or ``"[Netlist file not found]"`` if missing.
        """

        netlist_path = batch_path / "LLM_output" / f"{target_circuit_name}.net"

        if not netlist_path.exists():
            netlist_path = batch_path / "llm_output" / f"{target_circuit_name}.net"

        if netlist_path.exists():
            return netlist_path.read_text(encoding="utf-8").strip()

        return "[Netlist file not found]"

    def _format_simulation_feedback(
        self,
        details: dict,
        active_constraints: dict,
    ) -> str:
        """Format simulation metrics into prompt feedback."""

        metrics = details.get("raw_metrics", {})
        losses = details.get("loss_breakdown", {})

        v_out = metrics.get("simulation_output_voltage", 0.0)
        eff = metrics.get("efficiency", 0.0)

        target_vout = active_constraints.get("vout_target", "N/A")
        target_eff = active_constraints.get("efficiency_target", "N/A")

        text = (
            f"  - Performance: V_out = {v_out:.4f}V "
            f"(Target: {target_vout}V), "
            f"Efficiency = {eff:.4f} "
            f"(Target: {target_eff})\n"
        )

        text += "  - Raw Metrics:\n"
        for key, value in metrics.items():
            value_text = f"{value:.4f}" if isinstance(value, float) else str(value)
            text += f"      * {key}: {value_text}\n"

        if losses:
            text += "  - Loss Breakdown lower is better:\n"
            for key, value in losses.items():
                value_text = f"{value:.4f}" if isinstance(value, float) else str(value)
                text += f"      * {key}: {value_text}\n"

        text += (
            "  - CRITICAL RULE: Do not output the exact same netlist. "
            "Change topology, timing, duty cycle, or component values.\n"
        )

        return text

    def _format_validation_feedback(
        self,
        val_data: dict,
        target_circuit_name: str,
    ) -> str:
        """Format validation failures into prompt feedback."""

        circuit_val = val_data.get(target_circuit_name, {})
        checks = circuit_val.get("checks", {})

        failed_tests = [
            test
            for test, value in checks.items()
            if value == 0 or value is False
        ]

        if failed_tests:
            return (
                f"  - FAILED TESTS: {', '.join(failed_tests)}\n"
                "  - FIX: Ensure your new netlist resolves these violations.\n"
            )

        return "  - Note: Circuit failed basic structural/syntax checks.\n"

    def _aggregate_previous_batch_data(
        self,
        previous_batch_id: str,
        label: str,
        candidate_idx: int,
    ) -> str:
        """Build feedback string for a single candidate (legacy method).

        New grouped GRPO should use _aggregate_previous_batch_data_by_id().

        Args:
            previous_batch_id (str): Batch ID of the previous generation.
            label (str): Filename label slug for the constraint.
            candidate_idx (int): 1-based candidate index within the batch.

        Returns:
            str: Formatted feedback string, or empty string when unavailable.
        """

        if not previous_batch_id:
            return ""

        match = re.search(r"batch_(\d+)", str(previous_batch_id))
        prev_b_suffix = f"_b{match.group(1)}" if match else ""

        target_circuit_name = f"{label}_cand{candidate_idx}{prev_b_suffix}"

        return self._aggregate_previous_batch_data_by_id(
            previous_batch_id=previous_batch_id,
            target_circuit_name=target_circuit_name,
        )


# ── Convenience singleton (lazy-loaded) ────────────────────────────────

_singleton: TopologyLLM | None = None


def get_llm(**kwargs) -> TopologyLLM:
    """Return a process-wide singleton TopologyLLM."""

    global _singleton

    if _singleton is None:
        _singleton = TopologyLLM(**kwargs)

    return _singleton


if __name__ == "__main__":
    # Smoke test.
    print("llm_api.py import OK. Instantiate TopologyLLM only on HPC or with local model cache.")
    llm = TopologyLLM()
    out = llm.generate_from_constraint(
        {"vin": 12, "vout_target": 5, "efficiency_target": 0.9},
        n=1,
    )
    print(out[0][:200])