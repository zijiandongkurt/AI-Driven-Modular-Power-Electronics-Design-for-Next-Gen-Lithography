"""
demo_constrained_gen.py — End-to-end demo.

Pipeline:
    sample_constraints.json
        │
        ▼  prompt_input.load_constraints + make_prompt
    list of (constraint, prompt)
        │
        ▼  LLMEngine.generate (batch_size = 4 per constraint)
    4 candidate netlists per constraint
        │
        ▼  net_writer.write_net_file
    output_netlists/<idx>_<slug>.net   (one file per constraint, 4 candidates each)

No naming-convention rules are injected. This baseline demonstrates the
output format expected after pre-training/fine-tuning would normalise the
LLM's output style.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from llm_engine_minimal import LLMEngine
from prompt_input import load_constraints, make_prompt, slug
from net_writer import write_net_file


def generate_batch(engine: LLMEngine, prompt: str, n: int = 4) -> list[str]:
    """Generate n candidate netlists for a single prompt.

    Uses the engine's tokenizer + model directly so we can pass our own
    prompt (bypassing engine.generate's built-in prompt builder).
    """
    tok = engine.tokenizer
    model = engine.model
    ids = tok(prompt, return_tensors="pt").to(model.device)

    candidates: list[str] = []
    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=engine._max_new_tokens,
                do_sample=True,
                temperature=engine._temperature,
                top_p=engine._top_p,
                pad_token_id=tok.pad_token_id,
            )
        gen_ids = out[0][ids["input_ids"].shape[1]:]
        raw = tok.decode(gen_ids, skip_special_tokens=True)
        candidates.append(engine._clean(raw))
    return candidates


def run(
    json_path: str,
    out_dir: str,
    model_id: str,
    n: int,
    limit: int = 0,
):
    print(f"=== Demo run ===")
    print(f"Constraints file: {json_path}")
    print(f"Output dir:       {out_dir}")
    print(f"Model:            {model_id}")
    print(f"Batch per file:   {n}\n")

    constraints = load_constraints(json_path)
    if limit > 0:
        constraints = constraints[:limit]
    print(f"Processing {len(constraints)} constraints\n")

    print("Loading model in 4-bit (this may take a while on first run)...")
    t0 = time.time()
    engine = LLMEngine(model_id, quantization="4bit")
    print(f"Model ready in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")
    else:
        print()

    out_root = Path(out_dir)
    for i, c in enumerate(constraints):
        label = slug(c, i)
        prompt = make_prompt(c)
        print(f"[{i+1}/{len(constraints)}] {label}")

        t1 = time.time()
        candidates = generate_batch(engine, prompt, n=n)
        print(f"  generated {len(candidates)} candidates in {time.time()-t1:.1f}s")

        out_path = out_root / f"{label}.net"
        write_net_file(out_path, candidates, c)
        print(f"  wrote {out_path}\n")

    print(f"=== Done. {len(constraints)} .net files written to {out_root} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=str(Path(__file__).parent / "sample_constraints.json"),
        help="Path to constraint list JSON",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "output_netlists"),
        help="Output directory for .net files",
    )
    ap.add_argument(
        "--model",
        default="../../../models/qwen25-coder-7b",
        help="HF repo id or local path of the base model",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=4,
        help="Candidates per constraint (default 4)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N constraints (0 = all)",
    )
    args = ap.parse_args()
    run(args.json, args.out, args.model, args.n, args.limit)


if __name__ == "__main__":
    main()
