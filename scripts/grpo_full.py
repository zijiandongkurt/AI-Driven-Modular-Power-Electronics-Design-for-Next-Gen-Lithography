"""
grpo_full.py
============

Full GRPO training loop:
    generate (n=4 SFT-LoRA samples) → validate → simulate → reward → RL update

Requires Atakan's containerized LTspice setup:
    ~/ltspice-files/                       (bind-mounted to /sim)
    ~/run_ltspice_snellius.sh              (apptainer launcher)
    <his.sif> with Wine + LTspice XVIIx64.exe

If any of those are missing, this script will fail in the simulate stage.
Use scripts/grpo_smoke.py first to verify the RL pipeline before booking
a long full-loop run.

Knobs (env vars):
    SFT_ADAPTER_PATH    default: ./checkpoints/sft-lora/epoch-3
    GRPO_BATCH_PREFIX   default: batch_grpo_run
    GRPO_N_STEPS        default: 5     (how many GRPO iterations to do)
    GRPO_N_PER_STEP     default: 4     (group size for GRPO)
    GRPO_CONSTRAINT_IDX default: 0     (which constraint to optimize for)
    GRPO_MODEL_ID       default: Qwen/Qwen3-14B
    GRPO_TEMPERATURE    default: 0.5
    GRPO_TOP_P          default: 0.9

Run:
    python scripts/grpo_full.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HF cache wiring
_cache = (os.environ.get("HF_HUB_CACHE")
          or os.environ.get("HF_CACHE_DIR")
          or os.environ.get("HF_HOME"))
if _cache:
    import glob
    _flat   = glob.glob(os.path.join(_cache, "models--*"))
    _nested = glob.glob(os.path.join(_cache, "hub", "models--*"))
    _hub    = _cache if (_flat and not _nested) else os.path.join(_cache, "hub")
    os.environ["HF_HUB_CACHE"]       = _hub
    os.environ["TRANSFORMERS_CACHE"] = _hub
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.llm_topology_generation.prompt_input import load_constraint


def _import_downstream():
    """Lazy: only import these when we actually need them (so missing-on-LLM_Syh
    stubs only fail when full-loop training is requested)."""
    from pipeline.netlist_validation.validator import validator
    from pipeline.simulation.ltspice_runner_snellius import LTSpiceSimulator
    from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
    from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
    return validator, LTSpiceSimulator, RewardFunctionNorm, GRPOTrainer


def main():
    sft_adapter   = os.environ.get("SFT_ADAPTER_PATH", "./checkpoints/sft-lora/epoch-3")
    batch_prefix  = os.environ.get("GRPO_BATCH_PREFIX", "batch_grpo_run")
    n_steps       = int(os.environ.get("GRPO_N_STEPS",         "5"))
    n_per_step    = int(os.environ.get("GRPO_N_PER_STEP",      "4"))
    constraint_idx = int(os.environ.get("GRPO_CONSTRAINT_IDX", "0"))
    model_id      = os.environ.get("GRPO_MODEL_ID",   "Qwen/Qwen3-14B")
    temperature   = float(os.environ.get("GRPO_TEMPERATURE",   "0.5"))
    top_p         = float(os.environ.get("GRPO_TOP_P",         "0.9"))

    print("=" * 72)
    print(" GRPO full-loop training")
    print("=" * 72)
    print(f"  model_id        : {model_id}")
    print(f"  SFT adapter     : {sft_adapter}")
    print(f"  constraint idx  : {constraint_idx}")
    print(f"  steps           : {n_steps}")
    print(f"  n per step      : {n_per_step}")
    print(f"  batch prefix    : {batch_prefix}")
    print(f"  temp / top_p    : {temperature} / {top_p}")
    print()

    # 1. Load TopologyLLM + SFT adapter (trainable)
    print("[grpo_full] loading TopologyLLM ...")
    llm = TopologyLLM(model_id=model_id, temperature=temperature, top_p=top_p)
    print(f"[grpo_full] loading SFT LoRA adapter (trainable=True)")
    llm.engine.load_adapter("sft", sft_adapter, trainable=True)
    trainable = sum(p.numel() for p in llm.engine.model.parameters() if p.requires_grad)
    print(f"[grpo_full] trainable params: {trainable:,}")

    # 2. Build the downstream pipeline (raises if anything is missing)
    print("[grpo_full] loading validator / simulator / reward ...")
    validator_cls, LTSpiceSimulator, RewardFunctionNorm, GRPOTrainer = _import_downstream()
    val = validator_cls()
    sim = LTSpiceSimulator()
    rew = RewardFunctionNorm()

    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=constraint_idx)
    grpo = GRPOTrainer(
        llm=llm,
        validator=val, simulator=sim, reward_fn=rew,
        constraint=constraint,
    )

    # 3. Iterate
    history = []
    for step in range(1, n_steps + 1):
        batch_id = f"{batch_prefix}_step{step}"
        print(f"\n{'#'*72}\n# Step {step}/{n_steps}  batch={batch_id}\n{'#'*72}")
        try:
            metrics = grpo.train(batch_id=batch_id, n=n_per_step)
        except Exception as e:
            print(f"[FATAL] step {step} crashed: {e!r}")
            import traceback; traceback.print_exc()
            break
        history.append({"step": step, "batch_id": batch_id, "metrics": metrics})

    # 4. Save history
    out = REPO_ROOT / "checkpoints" / "grpo-lora" / "history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\n[grpo_full] history saved to {out}")
    print(f"[grpo_full] completed {len(history)} / {n_steps} steps")


if __name__ == "__main__":
    main()
