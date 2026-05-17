"""
grpo_smoke.py
=============

GRPO single-step smoke test.  Loads the SFT-trained LoRA adapter (as
trainable!) and runs ONE policy-gradient update on the pre-computed
reward_results.json of an existing batch.  **No simulator required.**

Use this to verify the SFT → GRPO hand-off pipeline before booking a
long full-loop run that needs LTspice.

Knobs (env vars):
    SFT_ADAPTER_PATH    default: ./checkpoints/sft-lora/epoch-3
    GRPO_BATCH_ID       default: batch_2
    GRPO_MODEL_ID       default: Qwen/Qwen3-14B
    GRPO_TEMPERATURE    default: 0.5
    GRPO_TOP_P          default: 0.9

Run:
    python scripts/grpo_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HF cache wiring (same trick as run_sft.py — don't touch HF_HOME)
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
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
from pipeline.reinforcement_algorithm.new_rl_updater import RLConfig


def _build_rl_config() -> RLConfig:
    """Same env-var override surface as grpo_full.py — keep them in sync."""
    cfg = RLConfig()
    if "GRPO_LR" in os.environ:           cfg.learning_rate   = float(os.environ["GRPO_LR"])
    if "GRPO_KL_BETA" in os.environ:      cfg.kl_beta         = float(os.environ["GRPO_KL_BETA"])
    if "GRPO_ENTROPY_BETA" in os.environ: cfg.entropy_beta    = float(os.environ["GRPO_ENTROPY_BETA"])
    if "GRPO_MAX_GRAD_NORM" in os.environ: cfg.max_grad_norm  = float(os.environ["GRPO_MAX_GRAD_NORM"])
    if "GRPO_SAVE_EVERY" in os.environ:   cfg.save_every      = int(os.environ["GRPO_SAVE_EVERY"])
    if "GRPO_LORA_R" in os.environ:       cfg.lora_r          = int(os.environ["GRPO_LORA_R"])
    if "GRPO_LORA_ALPHA" in os.environ:   cfg.lora_alpha      = int(os.environ["GRPO_LORA_ALPHA"])
    return cfg


def main():
    sft_adapter   = os.environ.get("SFT_ADAPTER_PATH", "./checkpoints/sft-lora/epoch-3")
    batch_id      = os.environ.get("GRPO_BATCH_ID",   "batch_2")
    model_id      = os.environ.get("GRPO_MODEL_ID",   "Qwen/Qwen3-14B")
    temperature   = float(os.environ.get("GRPO_TEMPERATURE", "0.5"))
    top_p         = float(os.environ.get("GRPO_TOP_P",       "0.9"))

    print("=" * 72)
    print(" GRPO smoke test — single RL step on pre-computed rewards")
    print("=" * 72)
    print(f"  model_id    : {model_id}")
    print(f"  SFT adapter : {sft_adapter}")
    print(f"  batch_id    : {batch_id}")
    print(f"  temp / top_p: {temperature} / {top_p}")
    print()

    # 1. Load base model
    print("[grpo_smoke] loading TopologyLLM ...")
    llm = TopologyLLM(
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
    )

    # 2. Attach SFT LoRA in TRAINABLE mode — this is the critical bit
    print(f"[grpo_smoke] loading SFT LoRA adapter (trainable=True) from {sft_adapter}")
    if not Path(sft_adapter).exists():
        print(f"[FATAL] adapter not found at {sft_adapter}")
        print(f"        looked here: {Path(sft_adapter).resolve()}")
        sys.exit(2)
    llm.engine.load_adapter("sft", sft_adapter, trainable=True)

    # 3. Sanity: confirm trainable params count
    trainable = sum(
        p.numel() for p in llm.engine.model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in llm.engine.model.parameters())
    print(f"[grpo_smoke] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")
    if trainable == 0:
        print("[FATAL] no trainable params — load_adapter trainable=True may not have taken effect")
        sys.exit(3)

    # 4. Build GRPOTrainer (no simulator/validator/reward — we use the
    #    pre-computed reward_results.json from disk)
    constraint = load_constraint(
        "pipeline/data/datasets/constraints.json", idx=0,
    )
    grpo = GRPOTrainer(
        llm=llm,
        validator=None, simulator=None, reward_fn=None,
        constraint=constraint,
        rl_config=_build_rl_config(),
    )

    # 5. Run one RL step
    print(f"\n[grpo_smoke] running train_from_existing_batch('{batch_id}') ...")
    metrics = grpo.train_from_existing_batch(batch_id)

    # 6. Verdict
    print("\n" + "=" * 72)
    print(" RESULT")
    print("=" * 72)
    print(f"  num_valid_samples : {metrics.get('num_valid_samples')}")
    print(f"  mean_reward       : {metrics.get('mean_reward')}")
    print(f"  policy_loss       : {metrics.get('policy_loss')}")
    print(f"  advantages        : {metrics.get('advantages')}")

    ok = (
        metrics.get("num_valid_samples", 0) >= 2
        and metrics.get("policy_loss") is not None
        and not (isinstance(metrics.get("policy_loss"), float)
                 and (metrics["policy_loss"] != metrics["policy_loss"]))  # NaN check
    )
    print("\n  Verdict: " + ("✅ PASS" if ok else "❌ FAIL"))
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
