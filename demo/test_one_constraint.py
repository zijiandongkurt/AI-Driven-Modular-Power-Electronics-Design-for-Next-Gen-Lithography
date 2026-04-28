"""
test_one_constraint.py — Instrumented single-constraint test.

Captures detailed metrics for one constraint (4 candidates):
  - LLM behaviour:  prompt/completion token counts, finish reason, output content
  - Compute usage:  load time, per-candidate latency, throughput,
                    VRAM peak, GPU utilization, CPU/RAM snapshot

Outputs:
  - test_report/metrics.json     (raw numbers)
  - test_report/report.md        (human-readable summary)
  - test_report/<slug>.net       (the .net file)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import psutil

from llm_engine_minimal import LLMEngine
from prompt_input import load_constraints, make_prompt, slug
from net_writer import write_net_file


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": p.name,
        "vram_total_gb": round(p.total_memory / 1e9, 2),
        "compute_capability": f"sm_{p.major}{p.minor}",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def system_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }


def run_test(
    json_path: str,
    model_id: str,
    out_dir: str,
    constraint_idx: int = 0,
    n: int = 4,
    max_new_tokens: int = 512,
):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "system": system_info(),
        "gpu": gpu_info(),
        "config": {
            "model_id": model_id,
            "constraint_idx": constraint_idx,
            "n_candidates": n,
            "max_new_tokens": max_new_tokens,
            "quantization": "4bit_nf4",
        },
    }

    # ── Load constraint ──────────────────────────────────────────────
    constraints = load_constraints(json_path)
    constraint = constraints[constraint_idx]
    label = slug(constraint, constraint_idx)
    metrics["constraint"] = constraint
    metrics["label"] = label

    # ── Reset CUDA stats baseline ────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1e9

    # ── Load model ───────────────────────────────────────────────────
    print(f"[1/3] Loading model in 4-bit NF4...")
    t0 = time.perf_counter()
    engine = LLMEngine(model_id, quantization="4bit", max_new_tokens=max_new_tokens)
    load_time = time.perf_counter() - t0
    vram_after_load = (
        torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    )
    metrics["load"] = {
        "time_sec": round(load_time, 2),
        "vram_after_load_gb": round(vram_after_load, 3),
        "ram_after_load_gb": round(proc.memory_info().rss / 1e9, 3),
        "ram_delta_gb": round(proc.memory_info().rss / 1e9 - ram_before, 3),
    }
    print(f"      loaded in {load_time:.2f}s, VRAM {vram_after_load:.2f} GB")

    # ── Build prompt ─────────────────────────────────────────────────
    prompt = make_prompt(constraint)
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt").to(engine.model.device)
    n_prompt_tokens = int(prompt_ids["input_ids"].shape[1])
    metrics["prompt"] = {
        "text": prompt,
        "n_tokens": n_prompt_tokens,
    }
    print(f"      prompt: {n_prompt_tokens} tokens")

    # ── Generate N candidates (fresh prompt re-tokenize each time) ───
    print(f"[2/3] Generating {n} candidates...")
    candidates: list[str] = []
    cand_metrics: list[dict] = []

    for i in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = engine.model.generate(
                **prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=engine._temperature,
                top_p=engine._top_p,
                pad_token_id=engine.tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t1

        gen_ids = out[0][prompt_ids["input_ids"].shape[1]:]
        n_gen = int(gen_ids.shape[0])
        raw_text = engine.tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned = engine._clean(raw_text)

        # Detect finish reason
        eos_id = engine.tokenizer.eos_token_id
        hit_eos = bool((gen_ids == eos_id).any().item()) if eos_id is not None else False
        finish_reason = "eos" if hit_eos else (
            "length" if n_gen >= max_new_tokens else "early_stop"
        )

        cand_metrics.append({
            "candidate": i + 1,
            "gen_time_sec": round(gen_time, 3),
            "n_tokens_generated": n_gen,
            "throughput_tok_per_sec": round(n_gen / gen_time, 2),
            "finish_reason": finish_reason,
            "n_chars_cleaned": len(cleaned),
            "n_lines_cleaned": cleaned.count("\n") + 1,
            "ends_with_dot_end": cleaned.lower().rstrip().endswith(".end"),
        })
        candidates.append(cleaned)
        print(
            f"      [{i+1}/{n}] {gen_time:.2f}s | {n_gen} tok | "
            f"{n_gen/gen_time:.1f} tok/s | finish={finish_reason}"
        )

    # ── Aggregate compute metrics ────────────────────────────────────
    total_gen_time = sum(m["gen_time_sec"] for m in cand_metrics)
    total_gen_tokens = sum(m["n_tokens_generated"] for m in cand_metrics)
    metrics["candidates"] = cand_metrics
    metrics["aggregate"] = {
        "total_gen_time_sec": round(total_gen_time, 2),
        "total_gen_tokens": total_gen_tokens,
        "avg_throughput_tok_per_sec": round(total_gen_tokens / total_gen_time, 2),
        "avg_latency_per_candidate_sec": round(total_gen_time / n, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
            if torch.cuda.is_available() else 0,
        "vram_at_end_gb": round(torch.cuda.memory_allocated() / 1e9, 3)
            if torch.cuda.is_available() else 0,
        "ram_at_end_gb": round(proc.memory_info().rss / 1e9, 3),
    }

    # ── Behaviour analysis ───────────────────────────────────────────
    has_vin   = sum(1 for c in candidates if "vin" in c.lower())
    has_rload = sum(1 for c in candidates if "rload" in c.lower())
    has_vgate = sum(1 for c in candidates if "vgate" in c.lower())
    has_dotend = sum(1 for c in candidates if c.lower().rstrip().endswith(".end"))
    has_tran  = sum(1 for c in candidates if ".tran" in c.lower())
    has_model = sum(1 for c in candidates if ".model" in c.lower())

    metrics["behaviour"] = {
        "candidates_with_Vin":     has_vin,
        "candidates_with_Rload":   has_rload,
        "candidates_with_Vgate":   has_vgate,
        "candidates_with_dot_end": has_dotend,
        "candidates_with_tran":    has_tran,
        "candidates_with_model":   has_model,
        "naming_compliance_ratio": round(
            (has_vin + has_rload + has_vgate + has_dotend + has_tran + has_model)
            / (6 * n), 3
        ),
    }

    # ── Write outputs ────────────────────────────────────────────────
    print(f"[3/3] Writing outputs to {out_root}...")
    net_path = write_net_file(out_root / f"{label}.net", candidates, constraint)
    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = out_root / "report.md"
    report_path.write_text(_render_markdown(metrics), encoding="utf-8")

    print(f"\n=== Test complete ===")
    print(f"  .net file:  {net_path}")
    print(f"  metrics:    {metrics_path}")
    print(f"  report:     {report_path}")


# ── Markdown report rendering ──────────────────────────────────────────

def _render_markdown(m: dict) -> str:
    sys_, gpu, cfg = m["system"], m["gpu"], m["config"]
    load, agg, beh = m["load"], m["aggregate"], m["behaviour"]
    cands = m["candidates"]

    lines = []
    lines.append(f"# Test Report — Single Constraint Generation\n")
    lines.append(f"**Constraint:** {m['constraint'].get('_comment', m['label'])}")
    lines.append(f"**Generated at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. Environment\n")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Platform | {sys_['platform']} |")
    lines.append(f"| Python | {sys_['python']} |")
    lines.append(f"| CPU logical cores | {sys_['cpu_count_logical']} |")
    lines.append(f"| RAM total | {sys_['ram_total_gb']} GB |")
    if gpu.get("available"):
        lines.append(f"| GPU | {gpu['name']} |")
        lines.append(f"| VRAM total | {gpu['vram_total_gb']} GB |")
        lines.append(f"| Compute capability | {gpu['compute_capability']} |")
        lines.append(f"| CUDA | {gpu['cuda']} |")
        lines.append(f"| PyTorch | {gpu['torch']} |")
    lines.append("")

    lines.append("## 2. Configuration\n")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    for k, v in cfg.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 3. Compute Consumption\n")
    lines.append("### Model Loading")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Load time | {load['time_sec']} s |")
    lines.append(f"| VRAM after load | {load['vram_after_load_gb']} GB |")
    lines.append(f"| RAM after load | {load['ram_after_load_gb']} GB |")
    lines.append(f"| RAM delta (loading) | {load['ram_delta_gb']} GB |")
    lines.append("")

    lines.append("### Generation (per candidate)")
    lines.append("| # | Time (s) | Tokens | Throughput (tok/s) | Finish | Lines | `.end`? |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in cands:
        lines.append(
            f"| {c['candidate']} | {c['gen_time_sec']} | {c['n_tokens_generated']} | "
            f"{c['throughput_tok_per_sec']} | {c['finish_reason']} | "
            f"{c['n_lines_cleaned']} | {'✓' if c['ends_with_dot_end'] else '✗'} |"
        )
    lines.append("")

    lines.append("### Aggregate")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total generation time | {agg['total_gen_time_sec']} s |")
    lines.append(f"| Total generated tokens | {agg['total_gen_tokens']} |")
    lines.append(f"| Avg throughput | {agg['avg_throughput_tok_per_sec']} tok/s |")
    lines.append(f"| Avg latency / candidate | {agg['avg_latency_per_candidate_sec']} s |")
    lines.append(f"| VRAM peak | {agg['vram_peak_gb']} GB |")
    lines.append(f"| VRAM at end | {agg['vram_at_end_gb']} GB |")
    lines.append(f"| RAM at end | {agg['ram_at_end_gb']} GB |")
    lines.append("")

    lines.append("## 4. LLM Behaviour\n")
    n = cfg["n_candidates"]
    lines.append(f"Naming-convention compliance across {n} candidates:\n")
    lines.append("| Token / Directive | Present in N candidates |")
    lines.append("|---|---|")
    lines.append(f"| `Vin` | {beh['candidates_with_Vin']} / {n} |")
    lines.append(f"| `Rload` | {beh['candidates_with_Rload']} / {n} |")
    lines.append(f"| `Vgate` | {beh['candidates_with_Vgate']} / {n} |")
    lines.append(f"| `.tran` | {beh['candidates_with_tran']} / {n} |")
    lines.append(f"| `.model` | {beh['candidates_with_model']} / {n} |")
    lines.append(f"| ends with `.end` | {beh['candidates_with_dot_end']} / {n} |")
    lines.append(f"| **Overall compliance** | **{beh['naming_compliance_ratio']*100:.1f}%** |")
    lines.append("")
    lines.append("> Without SFT/RL, the base Qwen2.5-Coder-7B is **not** expected to follow")
    lines.append("> the SPICE naming convention. This score establishes the pre-training baseline.\n")

    lines.append("## 5. Notes\n")
    lines.append("- Quantization: 4-bit NF4 with double quant, bf16 compute.")
    lines.append("- Sampling: temperature=0.7, top_p=0.9, do_sample=True.")
    lines.append("- All 4 candidates use the same prompt (variance comes from sampling).")
    lines.append("- Output `.net` file groups all 4 candidates into a single file with banners.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=str(Path(__file__).parent / "sample_constraints.json"),
    )
    ap.add_argument(
        "--model",
        default=r"D:\Document\Course\Team_intership\LLM\models\qwen25-coder-7b",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "test_report"),
    )
    ap.add_argument("--idx", type=int, default=0, help="Constraint index")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    run_test(
        json_path=args.json,
        model_id=args.model,
        out_dir=args.out,
        constraint_idx=args.idx,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
