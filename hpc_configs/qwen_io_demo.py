"""
qwen_io_demo.py — minimal standalone Qwen call: prompt in → text out.

Purpose:
  Smoke-test that the model loads and generates anything at all, WITHOUT
  involving the GRPO trainer, validator, simulator, or reward function.
  Use this as the first thing you run on a new HPC node — if Qwen can't
  load here, GRPO certainly can't.

What it does:
  1. Loads Qwen via the project's `TopologyLLM` wrapper (same code path
     GRPO would use, so it exercises the real loader / quantization /
     HF-cache plumbing).
  2. Generates N candidates for a single prompt.
  3. Prints them to stdout, plus VRAM peak.

Knobs (read from environment, all optional):
    GRPO_MODEL_ID      default "Qwen/Qwen3-14B"
    GRPO_QUANTIZATION  ""  / "4bit"          ← required on a 40GB card
    QWEN_DEMO_N        default 2             ← number of candidates
    QWEN_DEMO_PROMPT   default: idx-0 constraint from constraints.json
    QWEN_DEMO_MAX_NEW  default 512           ← max_new_tokens

Run locally / interactively:
    python hpc_configs/qwen_io_demo.py

Run on HPC:
    sbatch hpc_configs/slurm/smoke_qwen.slurm
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make sure `pipeline.*` imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch

from pipeline.llm_topology_generation.llm_api import TopologyLLM


def _build_prompt() -> str:
    """Use the project's normal constraint-prompt template by default,
    so the demo exercises the same prompt path as GRPO would."""
    custom = os.environ.get("QWEN_DEMO_PROMPT")
    if custom:
        return custom

    # Default: pull constraint idx=0 and run it through make_prompt
    try:
        from pipeline.llm_topology_generation.prompt_input import (
            load_constraint, make_prompt,
        )
        constraint = load_constraint(
            str(_REPO_ROOT / "pipeline" / "data" / "datasets" / "constraints.json"),
            idx=0,
        )
        return make_prompt(constraint)
    except Exception as e:
        print(f"[qwen_demo] couldn't load constraint ({e}), falling back to "
              f"a hard-coded prompt.")
        return (
            "You are a SPICE expert. Produce a 12V→5V buck converter netlist.\n"
            "Output ONLY the SPICE netlist — no commentary.\n\n"
            "### SPICE Netlist:\n"
        )


def _gpu_summary(stage: str):
    if not torch.cuda.is_available():
        print(f"[{stage}] CUDA not available — running on CPU (will be SLOW).")
        return
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    print(f"[{stage}] CUDA: device={torch.cuda.get_device_name(0)} "
          f"used={used:.2f} GB / total={total/1024**3:.2f} GB")


def _wire_hf_cache_env():
    """Convert our nonstandard GRPO_*/HF_CACHE_DIR knobs into the standard
    HF_HUB_CACHE / TRANSFORMERS_CACHE env vars that transformers +
    huggingface_hub actually read.  Must be called BEFORE any
    `from transformers import ...`.

    Auto-detects whether the cache layout is:
        $CACHE/hub/models--<org>--<name>/    (HF default)   → hub_dir = $CACHE/hub
        $CACHE/models--<org>--<name>/        (Snellius flat) → hub_dir = $CACHE
    """
    import glob

    cache = (os.environ.get("HF_HUB_CACHE")
             or os.environ.get("HF_CACHE_DIR")
             or os.environ.get("HF_HOME"))
    if not cache:
        print("[cache] No HF_HUB_CACHE / HF_CACHE_DIR / HF_HOME set — using HF default "
              "(~/.cache/huggingface).")
        return

    # Detect which layout this cache uses
    flat_layout   = glob.glob(os.path.join(cache, "models--*"))
    nested_layout = glob.glob(os.path.join(cache, "hub", "models--*"))

    if flat_layout and not nested_layout:
        hub_dir = cache
        layout = "flat ($CACHE/models--<org>--<name>)"
    elif nested_layout:
        hub_dir = os.path.join(cache, "hub")
        layout = "nested ($CACHE/hub/models--<org>--<name>)"
    else:
        # Fallback to HF's default expectation
        hub_dir = os.path.join(cache, "hub")
        layout = "unknown (defaulting to $CACHE/hub)"

    os.environ["HF_HOME"]              = cache
    os.environ["HF_HUB_CACHE"]         = hub_dir
    os.environ["TRANSFORMERS_CACHE"]   = hub_dir
    print(f"[cache] layout       = {layout}")
    print(f"[cache] HF_HOME      = {os.environ['HF_HOME']}")
    print(f"[cache] HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")


def main():
    _wire_hf_cache_env()

    model_id = os.environ.get("GRPO_MODEL_ID", "Qwen/Qwen3-14B")
    quant_env = os.environ.get("GRPO_QUANTIZATION", "")
    quantization = quant_env or None  # "" → None (full bf16)
    n_candidates = int(os.environ.get("QWEN_DEMO_N", "2"))
    max_new = int(os.environ.get("QWEN_DEMO_MAX_NEW", "512"))

    print("==================================================================")
    print(" Qwen I/O demo (no GRPO, no validator, no simulator)")
    print("==================================================================")
    print(f"  model_id     : {model_id}")
    print(f"  quantization : {quantization or 'bf16 (full)'}")
    print(f"  n_candidates : {n_candidates}")
    print(f"  max_new_tok  : {max_new}")
    print()

    _gpu_summary("before-load")

    t0 = time.time()
    llm = TopologyLLM(
        model_id=model_id,
        quantization=quantization,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
    )
    print(f"[load] OK in {time.time() - t0:.1f}s")
    _gpu_summary("after-load")

    prompt = _build_prompt()
    print(f"\n[prompt] {len(prompt)} chars")
    print("─" * 66)
    print(prompt)
    print("─" * 66)

    t0 = time.time()
    outputs = llm.generate_from_text(prompt, n=n_candidates)
    dt = time.time() - t0
    print(f"\n[generate] OK in {dt:.1f}s "
          f"({dt / max(n_candidates, 1):.1f}s per candidate)")
    _gpu_summary("after-gen")

    for i, text in enumerate(outputs, 1):
        print(f"\n══════ candidate {i}/{len(outputs)} "
              f"({len(text)} chars) ══════")
        print(text if text.strip() else "(empty output)")

    print("\n[done]")


if __name__ == "__main__":
    main()
