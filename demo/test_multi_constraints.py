"""
test_multi_constraints.py — Instrumented multi-constraint test.

Loads the model once, runs N constraints (default: 4 representative ones
spanning Step-Down / Step-Up / Up&Down), generates 4 candidates per
constraint, and writes:
  - test_report_multi/<slug>.net           (one .net file per constraint)
  - test_report_multi/metrics.json         (raw metrics for all constraints)
  - test_report_multi/report.md            (English aggregated report)
  - test_report_multi/报告.md               (Chinese aggregated report)
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


# ── Environment helpers ──────────────────────────────────────────────────

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


# ── Per-constraint generation ────────────────────────────────────────────

def run_one(engine: LLMEngine, constraint: dict, idx: int, n: int,
            max_new_tokens: int, out_root: Path) -> dict:
    label = slug(constraint, idx)
    prompt = make_prompt(constraint)
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt").to(engine.model.device)
    n_prompt_tokens = int(prompt_ids["input_ids"].shape[1])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    candidates: list[str] = []
    cand_metrics: list[dict] = []
    eos_id = engine.tokenizer.eos_token_id

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
        raw = engine.tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned = engine._clean(raw)

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

    # Behaviour analysis
    has_vin    = sum(1 for c in candidates if "vin" in c.lower())
    has_rload  = sum(1 for c in candidates if "rload" in c.lower())
    has_vgate  = sum(1 for c in candidates if "vgate" in c.lower())
    has_dotend = sum(1 for c in candidates if c.lower().rstrip().endswith(".end"))
    has_tran   = sum(1 for c in candidates if ".tran" in c.lower())
    has_model  = sum(1 for c in candidates if ".model" in c.lower())

    total_gen_time = sum(m["gen_time_sec"] for m in cand_metrics)
    total_gen_tokens = sum(m["n_tokens_generated"] for m in cand_metrics)

    # Write .net file
    net_path = write_net_file(out_root / f"{label}.net", candidates, constraint)

    return {
        "idx": idx,
        "label": label,
        "constraint": constraint,
        "prompt_n_tokens": n_prompt_tokens,
        "candidates": cand_metrics,
        "aggregate": {
            "total_gen_time_sec": round(total_gen_time, 2),
            "total_gen_tokens": total_gen_tokens,
            "avg_throughput_tok_per_sec": round(total_gen_tokens / total_gen_time, 2),
            "avg_latency_per_candidate_sec": round(total_gen_time / n, 2),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
                if torch.cuda.is_available() else 0,
        },
        "behaviour": {
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
        },
        "net_file": str(net_path),
    }


# ── Main run ─────────────────────────────────────────────────────────────

def run_test(
    json_path: str,
    model_id: str,
    out_dir: str,
    indices: list[int],
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
            "constraint_indices": indices,
            "n_candidates_per_constraint": n,
            "max_new_tokens": max_new_tokens,
            "quantization": "4bit_nf4",
        },
    }

    constraints_all = load_constraints(json_path)
    selected = [(i, constraints_all[i]) for i in indices]

    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1e9

    print(f"[1/3] Loading model in 4-bit NF4 (one-time)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
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

    print(f"[2/3] Running {len(selected)} constraints x {n} candidates each...")
    per_constraint: list[dict] = []
    t_run0 = time.perf_counter()
    for k, (idx, c) in enumerate(selected, 1):
        print(f"\n  ── ({k}/{len(selected)}) idx={idx}: {c.get('_comment', '')}")
        result = run_one(engine, c, idx, n, max_new_tokens, out_root)
        per_constraint.append(result)
    total_run_time = time.perf_counter() - t_run0

    metrics["per_constraint"] = per_constraint

    # Cross-constraint aggregate
    all_gen_time = sum(p["aggregate"]["total_gen_time_sec"] for p in per_constraint)
    all_gen_tok  = sum(p["aggregate"]["total_gen_tokens"]   for p in per_constraint)
    metrics["overall"] = {
        "n_constraints": len(per_constraint),
        "total_candidates": len(per_constraint) * n,
        "total_gen_time_sec": round(all_gen_time, 2),
        "wall_time_sec": round(total_run_time, 2),
        "total_gen_tokens": all_gen_tok,
        "avg_throughput_tok_per_sec": round(all_gen_tok / all_gen_time, 2),
        "avg_latency_per_candidate_sec": round(all_gen_time / (len(per_constraint) * n), 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
            if torch.cuda.is_available() else 0,
        "ram_at_end_gb": round(proc.memory_info().rss / 1e9, 3),
        "avg_naming_compliance_ratio": round(
            sum(p["behaviour"]["naming_compliance_ratio"] for p in per_constraint)
            / len(per_constraint), 3
        ),
    }

    # ── Write outputs ────────────────────────────────────────────────
    print(f"\n[3/3] Writing reports to {out_root}...")
    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_en_path = out_root / "report.md"
    report_en_path.write_text(_render_markdown_en(metrics), encoding="utf-8")
    report_zh_path = out_root / "报告.md"
    report_zh_path.write_text(_render_markdown_zh(metrics), encoding="utf-8")

    print(f"\n=== Test complete ===")
    print(f"  metrics:    {metrics_path}")
    print(f"  EN report:  {report_en_path}")
    print(f"  ZH report:  {report_zh_path}")
    print(f"  .net files: {len(per_constraint)} files in {out_root}")


# ── English markdown rendering ─────────────────────────────────────────

def _render_markdown_en(m: dict) -> str:
    sys_, gpu, cfg = m["system"], m["gpu"], m["config"]
    load, ovr = m["load"], m["overall"]
    rows = m["per_constraint"]

    out = []
    out.append("# Test Report — Multi-Constraint Generation\n")
    out.append(f"**Generated at:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"**Constraints tested:** {ovr['n_constraints']}  ")
    out.append(f"**Candidates per constraint:** {cfg['n_candidates_per_constraint']}  ")
    out.append(f"**Total candidates:** {ovr['total_candidates']}\n")

    out.append("## 1. Environment\n")
    out.append("| Item | Value |")
    out.append("|---|---|")
    out.append(f"| Platform | {sys_['platform']} |")
    out.append(f"| Python | {sys_['python']} |")
    out.append(f"| CPU logical cores | {sys_['cpu_count_logical']} |")
    out.append(f"| RAM total | {sys_['ram_total_gb']} GB |")
    if gpu.get("available"):
        out.append(f"| GPU | {gpu['name']} |")
        out.append(f"| VRAM total | {gpu['vram_total_gb']} GB |")
        out.append(f"| Compute capability | {gpu['compute_capability']} |")
        out.append(f"| CUDA | {gpu['cuda']} |")
        out.append(f"| PyTorch | {gpu['torch']} |")
    out.append("")

    out.append("## 2. Configuration\n")
    out.append("| Item | Value |")
    out.append("|---|---|")
    out.append(f"| model_id | {cfg['model_id']} |")
    out.append(f"| constraint_indices | {cfg['constraint_indices']} |")
    out.append(f"| n_candidates_per_constraint | {cfg['n_candidates_per_constraint']} |")
    out.append(f"| max_new_tokens | {cfg['max_new_tokens']} |")
    out.append(f"| quantization | {cfg['quantization']} |")
    out.append("")

    out.append("## 3. Compute Consumption\n")
    out.append("### Model Loading (one-time)")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Load time | {load['time_sec']} s |")
    out.append(f"| VRAM after load | {load['vram_after_load_gb']} GB |")
    out.append(f"| RAM after load | {load['ram_after_load_gb']} GB |")
    out.append(f"| RAM delta (loading) | {load['ram_delta_gb']} GB |")
    out.append("")

    out.append("### Per-constraint summary")
    out.append("| # | Constraint | Avg lat. (s) | Throughput (tok/s) | Tokens | Compliance |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        c = r["constraint"]
        a = r["aggregate"]
        b = r["behaviour"]
        comment = c.get("_comment", r["label"])
        out.append(
            f"| {r['idx']} | {comment} | "
            f"{a['avg_latency_per_candidate_sec']} | "
            f"{a['avg_throughput_tok_per_sec']} | "
            f"{a['total_gen_tokens']} | "
            f"{b['naming_compliance_ratio']*100:.1f}% |"
        )
    out.append("")

    out.append("### Overall aggregate")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Wall time (all constraints) | {ovr['wall_time_sec']} s |")
    out.append(f"| Pure generation time | {ovr['total_gen_time_sec']} s |")
    out.append(f"| Total generated tokens | {ovr['total_gen_tokens']} |")
    out.append(f"| Avg throughput | {ovr['avg_throughput_tok_per_sec']} tok/s |")
    out.append(f"| Avg latency / candidate | {ovr['avg_latency_per_candidate_sec']} s |")
    out.append(f"| VRAM peak | {ovr['vram_peak_gb']} GB |")
    out.append(f"| RAM at end | {ovr['ram_at_end_gb']} GB |")
    out.append("")

    out.append("## 4. LLM Behaviour\n")
    out.append("Naming-convention compliance per constraint (counts out of 4 candidates):\n")
    out.append("| # | Constraint | Vin | Rload | Vgate | .tran | .model | .end | Compliance |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    n = cfg["n_candidates_per_constraint"]
    for r in rows:
        b = r["behaviour"]
        comment = r["constraint"].get("_comment", r["label"])
        out.append(
            f"| {r['idx']} | {comment} | "
            f"{b['candidates_with_Vin']}/{n} | "
            f"{b['candidates_with_Rload']}/{n} | "
            f"{b['candidates_with_Vgate']}/{n} | "
            f"{b['candidates_with_tran']}/{n} | "
            f"{b['candidates_with_model']}/{n} | "
            f"{b['candidates_with_dot_end']}/{n} | "
            f"**{b['naming_compliance_ratio']*100:.1f}%** |"
        )
    out.append("")
    out.append(f"**Average compliance across {ovr['n_constraints']} constraints: "
               f"{ovr['avg_naming_compliance_ratio']*100:.1f}%**\n")
    out.append("> Without SFT/RL, the base Qwen2.5-Coder-7B is **not** expected to follow")
    out.append("> the SPICE naming convention. This score establishes the pre-training baseline.\n")

    out.append("## 5. Notes\n")
    out.append("- Quantization: 4-bit NF4 with double quant, bf16 compute.")
    out.append("- Sampling: temperature=0.7, top_p=0.9, do_sample=True.")
    out.append("- All candidates within a constraint share the same prompt; variance comes from sampling.")
    out.append("- One `.net` file per constraint, each containing all candidates separated by banners.")
    return "\n".join(out) + "\n"


# ── Chinese markdown rendering ─────────────────────────────────────────

def _render_markdown_zh(m: dict) -> str:
    sys_, gpu, cfg = m["system"], m["gpu"], m["config"]
    load, ovr = m["load"], m["overall"]
    rows = m["per_constraint"]

    out = []
    out.append("# 测试报告 — 多约束批量生成\n")
    out.append(f"**生成时间：** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"**测试约束数：** {ovr['n_constraints']}  ")
    out.append(f"**每条约束候选数：** {cfg['n_candidates_per_constraint']}  ")
    out.append(f"**总候选数：** {ovr['total_candidates']}\n")

    out.append("## 1. 运行环境\n")
    out.append("| 项目 | 取值 |")
    out.append("|---|---|")
    out.append(f"| 操作系统 | {sys_['platform']} |")
    out.append(f"| Python 版本 | {sys_['python']} |")
    out.append(f"| CPU 逻辑核心数 | {sys_['cpu_count_logical']} |")
    out.append(f"| 总内存 (RAM) | {sys_['ram_total_gb']} GB |")
    if gpu.get("available"):
        out.append(f"| GPU | {gpu['name']} |")
        out.append(f"| 显存总量 (VRAM) | {gpu['vram_total_gb']} GB |")
        out.append(f"| 计算能力 | {gpu['compute_capability']} |")
        out.append(f"| CUDA 版本 | {gpu['cuda']} |")
        out.append(f"| PyTorch 版本 | {gpu['torch']} |")
    out.append("")

    out.append("## 2. 配置\n")
    out.append("| 项目 | 取值 |")
    out.append("|---|---|")
    out.append(f"| 模型路径 | {cfg['model_id']} |")
    out.append(f"| 选用约束索引 | {cfg['constraint_indices']} |")
    out.append(f"| 每条约束候选数 | {cfg['n_candidates_per_constraint']} |")
    out.append(f"| 单候选最大 token 数 | {cfg['max_new_tokens']} |")
    out.append(f"| 量化方案 | {cfg['quantization']} |")
    out.append("")

    out.append("## 3. 算力消耗\n")
    out.append("### 模型加载（一次性）")
    out.append("| 指标 | 取值 |")
    out.append("|---|---|")
    out.append(f"| 加载耗时 | {load['time_sec']} 秒 |")
    out.append(f"| 加载后显存占用 | {load['vram_after_load_gb']} GB |")
    out.append(f"| 加载后内存占用 | {load['ram_after_load_gb']} GB |")
    out.append(f"| 加载内存增量 | {load['ram_delta_gb']} GB |")
    out.append("")

    out.append("### 各约束生成性能")
    out.append("| 序号 | 约束描述 | 平均延迟 (秒/候选) | 吞吐 (tok/s) | 总 token | 命名合规率 |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        c = r["constraint"]; a = r["aggregate"]; b = r["behaviour"]
        comment = c.get("_comment", r["label"])
        out.append(
            f"| {r['idx']} | {comment} | "
            f"{a['avg_latency_per_candidate_sec']} | "
            f"{a['avg_throughput_tok_per_sec']} | "
            f"{a['total_gen_tokens']} | "
            f"{b['naming_compliance_ratio']*100:.1f}% |"
        )
    out.append("")

    out.append("### 总体汇总")
    out.append("| 指标 | 取值 |")
    out.append("|---|---|")
    out.append(f"| 总挂钟耗时 | {ovr['wall_time_sec']} 秒 |")
    out.append(f"| 纯生成耗时 | {ovr['total_gen_time_sec']} 秒 |")
    out.append(f"| 累计生成 token | {ovr['total_gen_tokens']} |")
    out.append(f"| 平均吞吐 | {ovr['avg_throughput_tok_per_sec']} tok/s |")
    out.append(f"| 平均每候选延迟 | {ovr['avg_latency_per_candidate_sec']} 秒 |")
    out.append(f"| 显存峰值 | {ovr['vram_peak_gb']} GB |")
    out.append(f"| 末尾内存占用 | {ovr['ram_at_end_gb']} GB |")
    out.append("")

    out.append("## 4. LLM 行为监测\n")
    out.append("各约束在 4 个候选中包含命名约定 token 的数量：\n")
    out.append("| 序号 | 约束描述 | Vin | Rload | Vgate | .tran | .model | .end | 合规率 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    n = cfg["n_candidates_per_constraint"]
    for r in rows:
        b = r["behaviour"]
        comment = r["constraint"].get("_comment", r["label"])
        out.append(
            f"| {r['idx']} | {comment} | "
            f"{b['candidates_with_Vin']}/{n} | "
            f"{b['candidates_with_Rload']}/{n} | "
            f"{b['candidates_with_Vgate']}/{n} | "
            f"{b['candidates_with_tran']}/{n} | "
            f"{b['candidates_with_model']}/{n} | "
            f"{b['candidates_with_dot_end']}/{n} | "
            f"**{b['naming_compliance_ratio']*100:.1f}%** |"
        )
    out.append("")
    out.append(f"**全部 {ovr['n_constraints']} 条约束平均合规率："
               f"{ovr['avg_naming_compliance_ratio']*100:.1f}%**\n")
    out.append("> 未经 SFT / RL 微调时，基座 Qwen2.5-Coder-7B 并不会主动遵守 SPICE 命名约定，")
    out.append("> 此分数代表 **预训练阶段基线**，是后续微调效果对比的起点。\n")

    out.append("### 各约束完成原因分布\n")
    out.append("| 序号 | 约束描述 | eos | length | early_stop |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        comment = r["constraint"].get("_comment", r["label"])
        eos_cnt = sum(1 for c in r["candidates"] if c["finish_reason"] == "eos")
        len_cnt = sum(1 for c in r["candidates"] if c["finish_reason"] == "length")
        early   = sum(1 for c in r["candidates"] if c["finish_reason"] == "early_stop")
        out.append(f"| {r['idx']} | {comment} | {eos_cnt}/{n} | {len_cnt}/{n} | {early}/{n} |")
    out.append("")
    out.append("> `eos` 表示模型自然停止；`length` 表示截断到 max_new_tokens 上限；")
    out.append("> `early_stop` 表示其他终止原因。`length` 比例越高，说明模型倾向于无限重复。\n")

    out.append("## 5. 备注\n")
    out.append("- 量化方案：4-bit NF4，启用 double-quant，计算精度 bf16。")
    out.append("- 采样参数：temperature=0.7，top_p=0.9，do_sample=True。")
    out.append("- 同一约束的多个候选共享同一 prompt，差异仅来自采样随机性。")
    out.append("- 每条约束输出一个 `.net` 文件，文件中以 banner 分隔多个候选。")
    out.append("- 模型只加载一次，多约束循环复用，因此每条约束的实际生成耗时可视为稳态性能。")
    return "\n".join(out) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────

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
        default=str(Path(__file__).parent / "test_report_multi"),
    )
    ap.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 4, 8],
        help="Which constraint indices to run (default: 0 1 4 8 — one per category + extra)",
    )
    ap.add_argument("--n", type=int, default=4, help="Candidates per constraint")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    run_test(
        json_path=args.json,
        model_id=args.model,
        out_dir=args.out,
        indices=args.indices,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
