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
    GRPO_N_STEPS        default: 50    (was 5; bumped after 20-step diag
                                        showed learning needs more steps)
    GRPO_N_PER_STEP     default: 4     (group size for GRPO)
    GRPO_CONSTRAINT_IDX default: 0     (which constraint to optimize for)
    GRPO_MODEL_ID       default: Qwen/Qwen3-14B
    GRPO_TEMPERATURE    default: 0.7   (was 0.5; raised to fight mode
                                        collapse — diag run showed 3/4
                                        identical candidates by step 11)
    GRPO_TOP_P          default: 0.9

    GRPO_LR             default: 1e-5  (v2 — per-token mean loss scale)
    GRPO_KL_BETA        default: 0.05  (v2 NEW — KL(π || π_base) anchor)
    GRPO_ENTROPY_BETA   default: 0.0   (v2 NEW — bump to 0.01 if still collapsing)
    GRPO_MAX_GRAD_NORM  default: 1.0
    GRPO_SAVE_EVERY     default: 5

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
from pipeline.reinforcement_algorithm.new_rl_updater import RLConfig


def _import_downstream():
    """Lazy: only import these when we actually need them (so missing-on-LLM_Syh
    stubs only fail when full-loop training is requested)."""
    from pipeline.netlist_validation.validator import validator
    from pipeline.simulation.ltspice_runner_snellius import LTSpiceSimulator
    from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
    from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
    return validator, LTSpiceSimulator, RewardFunctionNorm, GRPOTrainer


def _build_rl_config() -> RLConfig:
    """Start from the H100 defaults baked into RLConfig and apply env-var
    overrides on top.  Keeps grpo_full.py one consistent surface for
    every dial we exposed in new_rl_updater.py v2."""
    cfg = RLConfig()        # ← H100 defaults (lr=1e-5, kl_beta=0.05, ...)

    if "GRPO_LR" in os.environ:
        cfg.learning_rate = float(os.environ["GRPO_LR"])
    if "GRPO_KL_BETA" in os.environ:
        cfg.kl_beta = float(os.environ["GRPO_KL_BETA"])
    if "GRPO_ENTROPY_BETA" in os.environ:
        cfg.entropy_beta = float(os.environ["GRPO_ENTROPY_BETA"])
    if "GRPO_MAX_GRAD_NORM" in os.environ:
        cfg.max_grad_norm = float(os.environ["GRPO_MAX_GRAD_NORM"])
    if "GRPO_SAVE_EVERY" in os.environ:
        cfg.save_every = int(os.environ["GRPO_SAVE_EVERY"])
    if "GRPO_LORA_R" in os.environ:
        cfg.lora_r = int(os.environ["GRPO_LORA_R"])
    if "GRPO_LORA_ALPHA" in os.environ:
        cfg.lora_alpha = int(os.environ["GRPO_LORA_ALPHA"])
    if "GRPO_MAX_LENGTH" in os.environ:
        L = int(os.environ["GRPO_MAX_LENGTH"])
        cfg.max_length = L
        cfg.max_prompt_length = min(cfg.max_prompt_length, max(L - 256, 128))
        cfg.max_completion_length = min(cfg.max_completion_length, L)

    return cfg


def main():
    sft_adapter   = os.environ.get("SFT_ADAPTER_PATH", "./checkpoints/sft-lora/epoch-3")
    batch_prefix  = os.environ.get("GRPO_BATCH_PREFIX", "batch_grpo_run")
    n_steps       = int(os.environ.get("GRPO_N_STEPS",         "50"))
    n_per_step    = int(os.environ.get("GRPO_N_PER_STEP",      "4"))
    constraint_idx = int(os.environ.get("GRPO_CONSTRAINT_IDX", "0"))
    model_id      = os.environ.get("GRPO_MODEL_ID",   "Qwen/Qwen3-14B")
    temperature   = float(os.environ.get("GRPO_TEMPERATURE",   "0.7"))
    top_p         = float(os.environ.get("GRPO_TOP_P",         "0.9"))

    rl_config = _build_rl_config()

    print("=" * 72)
    print(" GRPO full-loop training  (v2: KL + low-LR anchored)")
    print("=" * 72)
    print(f"  model_id        : {model_id}")
    print(f"  SFT adapter     : {sft_adapter}")
    print(f"  constraint idx  : {constraint_idx}")
    print(f"  steps           : {n_steps}")
    print(f"  n per step      : {n_per_step}")
    print(f"  batch prefix    : {batch_prefix}")
    print(f"  temp / top_p    : {temperature} / {top_p}")
    print(f"  learning_rate   : {rl_config.learning_rate}")
    print(f"  kl_beta         : {rl_config.kl_beta}")
    print(f"  entropy_beta    : {rl_config.entropy_beta}")
    print(f"  max_grad_norm   : {rl_config.max_grad_norm}")
    print(f"  save_every      : {rl_config.save_every}")
    print(f"  max_length      : {rl_config.max_length}")
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
        rl_config=rl_config,
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
