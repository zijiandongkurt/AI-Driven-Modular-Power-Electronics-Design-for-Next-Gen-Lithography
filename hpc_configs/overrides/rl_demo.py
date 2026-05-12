"""
rl_demo.py — HPC-tunable drop-in replacement.

This version reads SLURM-side environment variables so the same Python
file can power any of the three SLURM scripts (single 80GB, quad 40GB,
single 40GB) without code edits.

Knobs (set in the SLURM script via `export GRPO_*`):

    GRPO_MODEL_ID         default "Qwen/Qwen3-14B"
    GRPO_QUANTIZATION     "" (full bf16) or "4bit"           ← required on 40GB
    GRPO_BATCH_ID         default "batch_1"
    GRPO_N_SAMPLES        default unset  (use full batch)    ← OOM escape hatch
    GRPO_MAX_LENGTH       default unset  (use override defaults)
    GRPO_FULL_TRAIN       "0" / "1"  — call train() vs train_from_existing_batch()
"""

import os

# Wire our nonstandard cache knob (HF_CACHE_DIR) into the standard
# HF env vars BEFORE importing anything that touches transformers.
_cache = os.environ.get("HF_CACHE_DIR") or os.environ.get("HF_HOME")
if _cache:
    os.environ["HF_HOME"]              = _cache
    os.environ["HF_HUB_CACHE"]         = os.path.join(_cache, "hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache, "hub"))

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
from pipeline.reinforcement_algorithm.new_rl_updater import RLConfig


# ── Read SLURM-side env vars ────────────────────────────────────────────
MODEL_ID      = os.environ.get("GRPO_MODEL_ID",     "Qwen/Qwen3-14B")
QUANTIZATION  = os.environ.get("GRPO_QUANTIZATION", "") or None
BATCH_ID      = os.environ.get("GRPO_BATCH_ID",     "batch_1")
N_SAMPLES_CAP = os.environ.get("GRPO_N_SAMPLES")             # str or None
MAX_LENGTH    = os.environ.get("GRPO_MAX_LENGTH")            # str or None
FULL_TRAIN    = os.environ.get("GRPO_FULL_TRAIN", "0") == "1"


def _build_rl_config() -> RLConfig:
    """Build an RLConfig, applying env-var overrides on top of the override
    file's HPC defaults."""
    cfg = RLConfig(
        learning_rate=1e-5,
        kl_beta=0.0,
        save_every=5,
        lora_r=8,
        lora_alpha=16,
    )
    if MAX_LENGTH:
        L = int(MAX_LENGTH)
        cfg.max_length = L
        cfg.max_prompt_length = min(cfg.max_prompt_length, max(L - 256, 128))
        cfg.max_completion_length = min(cfg.max_completion_length, L)
        print(f"[rl_demo] GRPO_MAX_LENGTH override → max_length={cfg.max_length}, "
              f"max_prompt={cfg.max_prompt_length}, max_completion={cfg.max_completion_length}")
    return cfg


def main():
    print(f"[rl_demo] MODEL_ID={MODEL_ID}")
    print(f"[rl_demo] QUANTIZATION={QUANTIZATION or 'bf16 (full)'}")
    print(f"[rl_demo] BATCH_ID={BATCH_ID}")
    print(f"[rl_demo] N_SAMPLES_CAP={N_SAMPLES_CAP}")
    print(f"[rl_demo] FULL_TRAIN={FULL_TRAIN}")

    llm        = TopologyLLM(model_id=MODEL_ID, quantization=QUANTIZATION)
    val        = validator()
    simulator  = LTSpiceSimulator()
    reward_fn  = RewardFunctionNorm()
    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=0)

    grpo = GRPOTrainer(
        llm=llm,
        validator=val,
        simulator=simulator,
        reward_fn=reward_fn,
        constraint=constraint,
        rl_config=_build_rl_config(),
    )

    # OOM escape hatch — if GRPO_N_SAMPLES is set we monkey-patch the
    # builder to truncate the batch.  Loses some GRPO variance but lets
    # you fall back to 2 samples on a really tight 40GB card.
    if N_SAMPLES_CAP:
        cap = int(N_SAMPLES_CAP)
        _orig = grpo._build_training_batch

        def _capped(batch_id):
            p, c, r = _orig(batch_id)
            print(f"[rl_demo] GRPO_N_SAMPLES={cap} → trimming {len(r)} → {cap}")
            return p[:cap], c[:cap], r[:cap]

        grpo._build_training_batch = _capped

    if FULL_TRAIN:
        grpo.train(batch_id=BATCH_ID, n=4)
    else:
        grpo.train_from_existing_batch(batch_id=BATCH_ID)


if __name__ == "__main__":
    main()
