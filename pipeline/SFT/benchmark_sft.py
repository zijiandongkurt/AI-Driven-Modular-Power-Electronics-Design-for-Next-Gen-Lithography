"""
benchmark_sft.py
================

Head-to-head comparison of:
  (a) Base Qwen3-14B          (no adapter)
  (b) Qwen3-14B + SFT-LoRA    (the checkpoint we got after Stage 3)

For each model we:
  1. Pull N prompts from data/sft/sft_val.jsonl  (NEVER seen during gradient update)
  2. Generate K candidate netlists per prompt
  3. Run validator -> LTspice simulator -> RewardFunctionNorm on each candidate
  4. Aggregate three metrics per model:
       - validation pass rate     (well-formed SPICE)
       - simulation success rate  (LTspice ran and produced .raw)
       - mean normalized reward   (grpo_reward in [-1, +1])

Output:
  meeting/benchmark_results.csv   one row per (model, prompt_id, candidate_id)
  meeting/benchmark_summary.md    short table for the slide doc

Knobs (env vars):
  BENCH_N_PROMPTS   default 20
  BENCH_K_CANDS     default 4
  SFT_ADAPTER_PATH  default ./checkpoints/sft-lora/epoch-3
  BENCH_OUT_DIR     default meeting/
  GRPO_MODEL_ID     default Qwen/Qwen3-14B
  BENCH_TEMP        default 0.7

Run:
  bash scripts/benchmark_sft.sh
or:
  python scripts/benchmark_sft.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HF cache wiring (same trick as run_sft.py)
_cache = (os.environ.get("HF_HUB_CACHE")
          or os.environ.get("HF_CACHE_DIR")
          or os.environ.get("HF_HOME"))
if _cache:
    _flat   = glob.glob(os.path.join(_cache, "models--*"))
    _nested = glob.glob(os.path.join(_cache, "hub", "models--*"))
    _hub    = _cache if (_flat and not _nested) else os.path.join(_cache, "hub")
    os.environ["HF_HUB_CACHE"]       = _hub
    os.environ["TRANSFORMERS_CACHE"] = _hub
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.llm_topology_generation.net_writer import write_single_netlist


# 
# Helpers
# 

def load_val_constraints(jsonl_path: Path, n: int) -> List[Dict]:
    """Load the first n samples from a validation JSONL, keeping prompt and constraint.

    Args:
        jsonl_path (Path): Path to the validation JSONL file.
        n (int): Maximum number of samples to return.

    Returns:
        list[dict]: Each dict has keys ``"prompt_id"``, ``"prompt"``,
            ``"constraint"``, ``"tag"``, ``"source"``, and ``"gold"``.

    Raises:
        FileNotFoundError: If ``jsonl_path`` does not exist.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing val split: {jsonl_path}")
    out: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "prompt" not in d:
                continue
            out.append({
                "prompt_id": i,
                "prompt":    d["prompt"],
                "constraint": d.get("constraint", {}),
                "tag":       d.get("tag", ""),
                "source":    d.get("source", ""),
                "gold":      d.get("completion", ""),   # for reference, not used in scoring
            })
            if len(out) >= n:
                break
    return out


def generate_candidates(engine, prompt: str, k: int, temp: float) -> List[str]:
    """Call the model k times and return cleaned netlist strings.

    Takes the LLMEngine directly (not the TopologyLLM wrapper) so callers
    can pass ``llm.engine`` directly.

    Args:
        engine: LLMEngine instance with ``.model``, ``.tokenizer`` attributes.
        prompt (str): Full prompt string to feed to the model.
        k (int): Number of candidates to generate.
        temp (float): Sampling temperature.

    Returns:
        list[str]: Cleaned netlist strings, one per generation.
    """
    out: List[str] = []
    tok = engine.tokenizer
    model = engine.model
    device = model.device

    ids = tok(prompt, return_tensors="pt")
    ids = {k_: v.to(device) for k_, v in ids.items()}

    import torch
    for _ in range(k):
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[0][ids["input_ids"].shape[1]:]
        raw = tok.decode(new_tokens, skip_special_tokens=True)
        # Reuse the engine's cleaner so behaviour matches production.
        from pipeline.llm_topology_generation.llm_engine_minimal import LLMEngine
        out.append(LLMEngine._clean(raw))
    return out


def run_one_batch(
    batch_id: str,
    candidates: List[str],
    constraint: Dict,
    validator,
    simulator,
    reward_fn,
) -> List[Dict]:
    """Validate, simulate, and score one batch of candidate netlists.

    Args:
        batch_id (str): Unique batch identifier used for filesystem paths.
        candidates (list[str]): Raw netlist strings to evaluate.
        constraint (dict): Active constraint dict passed to the reward function.
        validator: Validator instance with a ``validate(batch_id)`` method.
        simulator: Simulator instance with a ``simulate(batch_id)`` method.
        reward_fn: Reward function instance with a ``process_batch(...)``
            method.

    Returns:
        list[dict]: One dict per candidate with keys ``"top"`` (str),
            ``"valid"`` (bool), ``"sim_ok"`` (bool), and ``"reward"``
            (float).
    """
    # 1. Write each candidate to top{1..K}.net in the canonical batch dir
    from pipeline.llm_topology_generation.net_writer import write_netlists
    written = write_netlists(candidates, batchID=batch_id)
    if not written:
        return []

    # 2. Validate (writes validation_results.json)
    val_report = validator.validate(batch_id)

    # 3. Simulate (writes simulation_results.csv)
    sim_report = simulator.simulate(batch_id)

    # 4. Reward (writes reward_results.json)
    reward_fn.process_batch(
        batch_id,
        constraint,
        weights={
            "v_out": 10.0, "efficiency": 20.0, "volume": 2.0,
            "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0,
                           "inductor": 1.0, "capacitor": 1.0},
        },
    )

    # 5. Read back the per-candidate verdict
    reward_path = REPO_ROOT / "pipeline" / "data" / batch_id / "reward_results.json"
    if not reward_path.exists():
        return [{"top": f"top{i+1}", "valid": False, "sim_ok": False,
                 "reward": -1.0} for i in range(len(candidates))]

    with reward_path.open() as f:
        rdata = json.load(f)

    rows: List[Dict] = []
    for i in range(len(candidates)):
        top = f"top{i+1}"
        info = rdata.get("circuits", {}).get(top, {})
        # validator passed?  check the per-top entry in validation_results.json
        val_path = REPO_ROOT / "pipeline" / "data" / batch_id / "validation_results.json"
        valid = False
        if val_path.exists():
            with val_path.open() as f:
                v = json.load(f)
            valid = bool(v.get(top, {}).get("valid", False))
        # sim succeeded? check .raw existence or reward presence
        sim_dir = REPO_ROOT / "pipeline" / "simulation" / "output" / batch_id
        sim_ok = (sim_dir / f"{top}.raw").exists() or (sim_dir / f"{top}.fail").exists() is False
        rows.append({
            "top":     top,
            "valid":   valid,
            "sim_ok":  sim_ok,
            "reward":  float(info.get("grpo_reward", info.get("fitness_score", -1.0))),
        })
    return rows


def aggregate(per_cand_rows: List[Dict]) -> Dict[str, float]:
    """Compute the three headline benchmark metrics from per-candidate rows.

    Args:
        per_cand_rows (list[dict]): Output of ``run_one_batch``; each dict
            must have ``"valid"`` (bool), ``"sim_ok"`` (bool), and
            ``"reward"`` (float) keys.

    Returns:
        dict: ``{"valid_pct": float, "sim_pct": float, "mean_reward": float}``
            where percentages are in the range [0, 100].
    """
    if not per_cand_rows:
        return {"valid_pct": 0.0, "sim_pct": 0.0, "mean_reward": -1.0}
    n = len(per_cand_rows)
    valid_pct = 100.0 * sum(1 for r in per_cand_rows if r["valid"]) / n
    sim_pct   = 100.0 * sum(1 for r in per_cand_rows if r["sim_ok"]) / n
    mean_r    = sum(r["reward"] for r in per_cand_rows) / n
    return {"valid_pct": valid_pct, "sim_pct": sim_pct, "mean_reward": mean_r}


# 
# Main
# 

def main():
    sft_adapter   = os.environ.get("SFT_ADAPTER_PATH", "./checkpoints/sft-lora/epoch-3")
    n_prompts     = int(os.environ.get("BENCH_N_PROMPTS", "20"))
    k_cands       = int(os.environ.get("BENCH_K_CANDS",   "4"))
    out_dir       = Path(os.environ.get("BENCH_OUT_DIR",  str(REPO_ROOT / "meeting")))
    model_id      = os.environ.get("GRPO_MODEL_ID",       "Qwen/Qwen3-14B")
    temp          = float(os.environ.get("BENCH_TEMP",    "0.7"))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_results.csv"
    md_path  = out_dir / "benchmark_summary.md"

    print("=" * 72)
    print(" SFT vs Base — benchmark on held-out val prompts")
    print("=" * 72)
    print(f"  model_id     : {model_id}")
    print(f"  SFT adapter  : {sft_adapter}")
    print(f"  n_prompts    : {n_prompts}")
    print(f"  k_cands      : {k_cands}")
    print(f"  temp         : {temp}")
    print(f"  output dir   : {out_dir}")
    print()

    #  Load held-out prompts 
    val_path = REPO_ROOT / "data" / "sft" / "sft_val.jsonl"
    samples = load_val_constraints(val_path, n_prompts)
    print(f"[bench] loaded {len(samples)} held-out prompts")
    if len(samples) < n_prompts:
        print(f"[bench] WARNING: only {len(samples)} samples available — running on what we have")

    #  Lazy imports — we don't need the simulator until generation is done 
    from pipeline.netlist_validation.validator import validator as ValidatorCls
    from pipeline.simulation.ltspice_runner_snellius import LTSpiceSimulator
    from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm

    val_inst = ValidatorCls()
    sim_inst = LTSpiceSimulator()
    rew_inst = RewardFunctionNorm()

    #  Open CSV 
    csv_fp = csv_path.open("w", encoding="utf-8", newline="")
    csv_w  = csv.writer(csv_fp)
    csv_w.writerow(["model", "prompt_id", "tag", "candidate", "valid", "sim_ok", "reward"])
    csv_fp.flush()

    #  Outer loop: one model at a time so we don't pay 2× the load cost 
    all_rows: Dict[str, List[Dict]] = {"base": [], "sft": []}

    for model_label, adapter_path in [("base", None), ("sft", sft_adapter)]:
        print()
        print(f" Model: {model_label} ")
        print(f"[bench] loading TopologyLLM (adapter={adapter_path})")
        llm = TopologyLLM(model_id=model_id, temperature=temp, top_p=0.9)
        if adapter_path is not None:
            if not Path(adapter_path).exists():
                print(f"[FATAL] adapter not found at {adapter_path}")
                sys.exit(2)
            llm.engine.load_adapter("sft", adapter_path, trainable=False)

        t0_model = time.time()
        for s in samples:
            batch_id = f"bench_{model_label}_p{s['prompt_id']}"
            print(f"  [{model_label}] prompt_id={s['prompt_id']} tag={s.get('tag','')[:40]}")
            t0 = time.time()
            cands = generate_candidates(llm.engine, s["prompt"], k_cands, temp)
            t1 = time.time()
            print(f"    gen: {t1-t0:.1f}s, lengths={[len(c) for c in cands]}")

            rows = run_one_batch(
                batch_id=batch_id,
                candidates=cands,
                constraint=s.get("constraint") or {},
                validator=val_inst,
                simulator=sim_inst,
                reward_fn=rew_inst,
            )
            for r in rows:
                csv_w.writerow([
                    model_label, s["prompt_id"], s.get("tag",""),
                    r["top"], int(r["valid"]), int(r["sim_ok"]),
                    f"{r['reward']:.4f}",
                ])
                all_rows[model_label].append(r)
            csv_fp.flush()
            print(f"    valid={sum(r['valid'] for r in rows)}/{len(rows)}, "
                  f"sim_ok={sum(r['sim_ok'] for r in rows)}/{len(rows)}, "
                  f"reward_mean={sum(r['reward'] for r in rows)/max(len(rows),1):+.3f}, "
                  f"total={time.time()-t0:.1f}s")

        print(f"[bench] {model_label} done in {time.time()-t0_model:.1f}s")

        # Tear down llm to free GPU memory before loading the next variant.
        # (Both variants are the SAME base model, but we keep it simple —
        # the alternative is to load the base once and then `set_adapter`
        # back and forth, which is faster but harder to get right when
        # PEFT's adapter state interleaves with engine state.)
        del llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_fp.close()
    print(f"\n[bench] CSV written → {csv_path}")

    #  Summary 
    base_agg = aggregate(all_rows["base"])
    sft_agg  = aggregate(all_rows["sft"])
    delta = {k: sft_agg[k] - base_agg[k] for k in base_agg}

    md_lines = [
        "# SFT vs Base benchmark",
        "",
        f"- prompts: {len(samples)} (held-out from `data/sft/sft_val.jsonl`)",
        f"- candidates per prompt: {k_cands}",
        f"- total candidates per model: {len(all_rows['base'])}",
        f"- temperature: {temp}",
        "",
        "| Model | Valid % | Sim OK % | Mean reward |",
        "|---|---|---|---|",
        f"| Base Qwen3-14B          | {base_agg['valid_pct']:5.1f} % | {base_agg['sim_pct']:5.1f} % | {base_agg['mean_reward']:+.3f} |",
        f"| **+ SFT LoRA (epoch-3)** | **{sft_agg['valid_pct']:5.1f} %** | **{sft_agg['sim_pct']:5.1f} %** | **{sft_agg['mean_reward']:+.3f}** |",
        f"| Δ (SFT − base) | {delta['valid_pct']:+5.1f} pp | {delta['sim_pct']:+5.1f} pp | {delta['mean_reward']:+.3f} |",
        "",
        "Raw rows: `meeting/benchmark_results.csv` (one row per candidate).",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "\n".join(md_lines))
    print(f"\n[bench] summary written → {md_path}")


if __name__ == "__main__":
    main()
