"""
generate_sft_candidates.py
==========================

Uses your local Qwen2.5-Coder-7B (via TopologyLLM) to bootstrap more SFT
candidates.  For each constraint in constraints.json, generates N candidates
with high temperature, then keeps only those that pass `validator()`.

Since this Windows box has neither LTspice nor working ngspice, we can NOT
compute fitness_score here — that's left to a later HPC pass.  But
validator-passed candidates are still usable SFT data: they're at least
syntactically and structurally correct.

Outputs:
    1. data/sft/gen_<modelname>/batch_<idx>/<LLM_output, validation_results.json>
       (project-standard layout, so you can re-score them later via HPC pipeline)
    2. data/sft/sft_from_generation.jsonl
       (validator-passed (prompt, completion) pairs ready for SFT training)

CLI:
    python scripts/generate_sft_candidates.py \
        --indices 0 1 2 3 4 \
        --n-per-constraint 24 \
        --temperature 0.9 \
        --top-p 0.95 \
        --max-new-tokens 768 \
        --model-id "<defaults to project's local Qwen2.5-Coder-7B>"

Memory: Qwen2.5-Coder-7B + 4-bit NF4 ≈ 4-5 GB VRAM (fits on most laptops).
Speed:  ~5 s/candidate on RTX 5070, so 5 constraints × 24 ≈ 10 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.llm_topology_generation.prompt_input import (
    load_constraints, make_prompt,
)
from pipeline.llm_topology_generation.net_writer import (
    write_netlists, get_llm_output_dir,
)


def _load_fewshot_examples(path: Path, k: int = 2) -> list[dict]:
    """Load top-k high-fitness samples for use as in-context examples.

    Args:
        path (Path): Path to ``sft_from_existing.jsonl`` or any JSONL with
            a ``"fitness"`` field per line.
        k (int): Number of top-fitness samples to return.

    Returns:
        list[dict]: Up to ``k`` rows sorted by descending fitness; empty
            list if the file is missing.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    # Keep rows that have a real fitness score, sort desc, take top-k
    scored = [r for r in rows if r.get("fitness") is not None]
    scored.sort(key=lambda r: r["fitness"], reverse=True)
    return scored[:k]


def _build_fewshot_prompt(constraint: dict, fewshots: list[dict]) -> str:
    """Build a prompt that injects worked examples before the target constraint.

    Falls back to the bare ``make_prompt`` output when ``fewshots`` is empty.

    Args:
        constraint (dict): Target constraint for which a netlist is requested.
        fewshots (list[dict]): In-context examples; each dict must have
            ``"completion"`` and optionally ``"constraint"`` keys.

    Returns:
        str: Full prompt string ready for the model.
    """
    if not fewshots:
        return make_prompt(constraint)

    # Re-use NAMING_RULES preamble via make_prompt of the first fewshot's
    # constraint — that way NAMING_RULES stays the project's canonical text.
    # Then we append worked examples + the actual target constraint.
    from pipeline.llm_topology_generation.prompt_input import NAMING_RULES

    parts: list[str] = [NAMING_RULES.rstrip(), ""]

    parts.append("### Worked examples (your output must follow this exact format):")
    for i, fs in enumerate(fewshots, 1):
        c_short = {k: v for k, v in (fs.get("constraint") or {}).items()
                   if not k.startswith("_")}
        parts.append(f"")
        parts.append(f"# Example {i} — constraint: {json.dumps(c_short)}")
        parts.append(fs["completion"].strip())

    parts.append("")
    parts.append("### Now produce a netlist for the following constraint.")
    parts.append("Same format as the examples above (NO `key=value`, NO Markdown, NO Python).")
    parts.append("Pick DIFFERENT component values from the examples to add diversity.")
    parts.append("")
    parts.append("### Constraint:")
    payload = {k: v for k, v in constraint.items() if not k.startswith("_")}
    parts.append(json.dumps(payload, indent=2))
    parts.append("")
    parts.append("### SPICE Netlist:")
    return "\n".join(parts) + "\n"


def _run_validator(batch_id: str) -> dict:
    """Run the validator on a batch and return parsed validation results.

    Args:
        batch_id (str): Batch identifier; determines where the validator
            reads and writes files.

    Returns:
        dict: Parsed ``validation_results.json`` keyed by topology id;
            empty dict if the validator cannot be imported or the results
            file is not written.
    """
    try:
        from pipeline.netlist_validation.validator import validator
    except ImportError as e:
        print(f"  WARN: can't import validator ({e}); skipping validation, "
              f"will keep ALL non-empty candidates.")
        return {}

    val = validator()
    val.validate(batch_id)

    # Read back the file the validator wrote
    val_path = REPO_ROOT / "pipeline" / "data" / batch_id / "validation_results.json"
    if not val_path.exists():
        print(f"  WARN: validator ran but validation_results.json not found")
        return {}
    return json.loads(val_path.read_text(encoding="utf-8"))


def generate_for_constraint(llm, constraint_idx, constraint, n, batch_id_prefix,
                            fewshots):
    """Generate, validate, and collect SFT pairs for one constraint.

    Args:
        llm: TopologyLLM instance used for generation.
        constraint_idx (int): Index of the constraint in the constraints list.
        constraint (dict): Constraint specification.
        n (int): Number of candidate netlists to generate.
        batch_id_prefix (str): Prefix for the batch identifier written to
            disk (e.g. ``"batch_sft_gen"``).
        fewshots (list[dict]): In-context examples injected into the prompt.

    Returns:
        list[dict]: Validator-passing SFT pairs, each with keys ``"prompt"``,
            ``"completion"``, ``"fitness"``, ``"source"``, and
            ``"constraint"``.
    """
    print(f"\n")
    print(f" Constraint #{constraint_idx}: "
          f"vin={constraint.get('vin_min')}-{constraint.get('vin_max')}V → "
          f"vout={constraint.get('vout_target')}V, "
          f"P={constraint.get('power_in')}W")
    print(f"")

    batch_id = f"{batch_id_prefix}_idx{constraint_idx}"
    if fewshots:
        prompt = _build_fewshot_prompt(constraint, fewshots)
        print(f"[prompt] using {len(fewshots)} worked example(s); "
              f"prompt length = {len(prompt)} chars")
    else:
        prompt = make_prompt(constraint)
        print(f"[prompt] no fewshots; using bare make_prompt "
              f"(prompt length = {len(prompt)} chars)")

    # 1. Generate n raw candidates
    t0 = time.time()
    cands = llm.generate_from_text(prompt, n=n)
    dt = time.time() - t0
    nonempty = sum(1 for c in cands if c and c.strip())
    print(f"[gen] {nonempty}/{n} non-empty candidates in {dt:.1f}s "
          f"({dt/max(n,1):.1f}s each)")

    # 2. Write to disk in canonical batch layout
    written = write_netlists(netlists=cands, batchID=batch_id)
    print(f"[write] {len(written)} files → {get_llm_output_dir(batch_id)}")

    # 3. Run validator
    val_results = _run_validator(batch_id)
    n_passed = sum(1 for v in val_results.values() if v.get("passed"))
    print(f"[validate] {n_passed}/{len(val_results)} passed all 23 checks")

    # 4. Build SFT pairs from passed candidates
    pairs = []
    for path in written:
        # Validator writes results keyed by either "topN.net" or "topN"
        stem = path.stem
        info = val_results.get(stem) or val_results.get(path.name) or {}
        if not info.get("passed"):
            continue
        pairs.append({
            "prompt":     prompt,
            "completion": path.read_text(encoding="utf-8"),
            "fitness":    None,                # not scored yet — needs simulator
            "source":     f"{batch_id}/{stem}",
            "constraint": {k: v for k, v in constraint.items() if not k.startswith("_")},
        })
    print(f"[sft] kept {len(pairs)} validator-passing pair(s) from this constraint")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", type=Path,
                    default=REPO_ROOT / "pipeline" / "data" / "datasets" / "constraints.json")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Which constraint indices to generate for")
    ap.add_argument("--n-per-constraint", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--model-id", type=str, default=None,
                    help="HF id or local path; default = project's DEFAULT_MODEL_ID "
                         "(your local Qwen2.5-Coder-7B)")
    ap.add_argument("--quantization", type=str, default="4bit",
                    choices=["4bit", "fp16", None],
                    help="4bit (default, ~5GB VRAM) or fp16 (≥16GB)")
    ap.add_argument("--batch-prefix", type=str, default="batch_sft_gen",
                    help="Prefix for the per-constraint batch_id")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "sft" / "sft_from_generation.jsonl")
    ap.add_argument("--fewshot-from", type=Path,
                    default=REPO_ROOT / "data" / "sft" / "sft_from_existing.jsonl",
                    help="JSONL of known-good (prompt, completion, fitness) pairs "
                         "to inject as worked examples. Empty / missing → 0-shot.")
    ap.add_argument("--fewshot-k", type=int, default=2,
                    help="How many top-fitness samples to use as in-context examples")
    args = ap.parse_args()

    #  Sanity 
    if not args.constraints.exists():
        print(f"FATAL: {args.constraints} not found", file=sys.stderr)
        sys.exit(1)

    constraints = load_constraints(str(args.constraints))
    print(f"[setup] loaded {len(constraints)} constraints from {args.constraints}")
    print(f"[setup] will generate for indices {args.indices}")
    print(f"[setup] n_per_constraint={args.n_per_constraint}, "
          f"temperature={args.temperature}, top_p={args.top_p}")

    #  Load model once, reuse across all constraints 
    kwargs = {
        "quantization":   args.quantization,
        "max_new_tokens": args.max_new_tokens,
        "temperature":    args.temperature,
        "top_p":          args.top_p,
    }
    if args.model_id:
        kwargs["model_id"] = args.model_id

    print(f"\n[model] loading TopologyLLM({kwargs}) ...")
    t0 = time.time()
    llm = TopologyLLM(**kwargs)
    print(f"[model] ready in {time.time() - t0:.1f}s")

    #  Load few-shot examples 
    fewshots = _load_fewshot_examples(args.fewshot_from, k=args.fewshot_k)
    if fewshots:
        print(f"[fewshot] loaded {len(fewshots)} worked example(s) from "
              f"{args.fewshot_from}")
        for fs in fewshots:
            print(f"           - {fs['source']}  fitness={fs['fitness']:.2f}")
    else:
        print(f"[fewshot] {args.fewshot_from} missing or empty; using 0-shot")

    #  Generate per constraint 
    all_pairs: list[dict] = []
    for idx in args.indices:
        if idx >= len(constraints):
            print(f"WARN: index {idx} out of range ({len(constraints)} constraints)")
            continue
        pairs = generate_for_constraint(
            llm, idx, constraints[idx],
            n=args.n_per_constraint,
            batch_id_prefix=args.batch_prefix,
            fewshots=fewshots,
        )
        all_pairs.extend(pairs)

    #  Save 
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(all_pairs)} validator-passing SFT pair(s)")
    print(f"[done] written to {args.out}")
    if all_pairs:
        comp_lens = [len(p["completion"]) for p in all_pairs]
        print(f"[done] completion length: min={min(comp_lens)}, "
              f"max={max(comp_lens)}, avg={sum(comp_lens)//len(comp_lens)} chars")


if __name__ == "__main__":
    main()
