"""
rl_demo.py — HPC-tunable drop-in replacement.

This version reads SLURM-side environment variables so the same Python
file can power any of the three SLURM scripts (single 80GB, quad 40GB,
single 40GB) without code edits.

Defaults match the H100 80GB recommended config in
`hpc_configs/overrides/new_rl_updater.py`.  Tighter cards (40GB) scale
these down via env vars set in their SLURM script.

Knobs (set in the SLURM script via `export GRPO_*`):

    Model                                                            ← H100 default
    ─────────────────────────────────────────────────────────────────
    GRPO_MODEL_ID         "Qwen/Qwen3-14B"                            ← Qwen3-14B
    GRPO_QUANTIZATION     "" (full bf16) or "4bit"                    ← bf16 on H100, 4bit on 40GB
    GRPO_BATCH_ID         "batch_1"
    GRPO_FULL_TRAIN       "0" / "1"  — train() vs train_from_existing_batch()

    Sequence  (defaults from new_rl_updater.py RLConfig)              ← H100 default
    ─────────────────────────────────────────────────────────────────
    GRPO_MAX_LENGTH       unset → 2048                                ← 2048
    GRPO_N_SAMPLES        unset → full batch (no cap)                 ← OOM escape hatch

    LoRA  (defaults from new_rl_updater.py RLConfig)                  ← H100 default
    ─────────────────────────────────────────────────────────────────
    GRPO_LORA_R           unset → 16                                  ← 16
    GRPO_LORA_ALPHA       unset → 32                                  ← 32 (= 2 × r)

    Training  (defaults from new_rl_updater.py RLConfig)              ← H100 default
    ─────────────────────────────────────────────────────────────────
    GRPO_LR               unset → 2e-5                                ← 2e-5
"""

import os
import glob

# Wire our nonstandard cache knobs into the standard HF env vars BEFORE
# importing anything that touches transformers.  Auto-detect flat vs
# nested cache layout (Snellius is flat: $CACHE/models--<org>--<name>).
_cache = (os.environ.get("HF_HUB_CACHE")
          or os.environ.get("HF_CACHE_DIR")
          or os.environ.get("HF_HOME"))
if _cache:
    _flat   = glob.glob(os.path.join(_cache, "models--*"))
    _nested = glob.glob(os.path.join(_cache, "hub", "models--*"))
    if _flat and not _nested:
        _hub = _cache
    elif _nested:
        _hub = os.path.join(_cache, "hub")
    else:
        _hub = os.path.join(_cache, "hub")
    os.environ["HF_HOME"]            = _cache
    os.environ["HF_HUB_CACHE"]       = _hub
    os.environ["TRANSFORMERS_CACHE"] = _hub

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
    """Build an RLConfig, starting from the H100-tuned defaults in
    new_rl_updater.py's RLConfig() dataclass, then applying env-var
    overrides on top (used by 40GB SLURM to scale things down)."""
    cfg = RLConfig()        # ← H100 defaults (lora_r=16, max_length=2048, lr=2e-5)

    # LoRA scaling (env-var override)
    if "GRPO_LORA_R" in os.environ:
        cfg.lora_r = int(os.environ["GRPO_LORA_R"])
        print(f"[rl_demo] GRPO_LORA_R override → lora_r={cfg.lora_r}")
    if "GRPO_LORA_ALPHA" in os.environ:
        cfg.lora_alpha = int(os.environ["GRPO_LORA_ALPHA"])
        print(f"[rl_demo] GRPO_LORA_ALPHA override → lora_alpha={cfg.lora_alpha}")

    # Learning rate (env-var override)
    if "GRPO_LR" in os.environ:
        cfg.learning_rate = float(os.environ["GRPO_LR"])
        print(f"[rl_demo] GRPO_LR override → learning_rate={cfg.learning_rate}")

    # Sequence-length (env-var override)
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
