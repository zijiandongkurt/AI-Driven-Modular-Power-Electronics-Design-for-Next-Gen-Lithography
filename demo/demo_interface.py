"""
demo_interface.py — Gradio web UI for the topology-generation LLM.

Two tabs:
  1. JSON 约束输入 — upload a constraints JSON file. Each constraint is
     fed through the make_prompt template, the LLM generates N candidates,
     and one .net file per constraint is written to disk.
  2. 直接文本输入  — type a raw prompt. The LLM generates and the result
     is displayed inline (no file is written).

The model is loaded ONCE on the first request (lazy) and reused.

Launch:
    python demo_interface.py
    → opens http://127.0.0.1:7860
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import gradio as gr

from llm_api import get_llm, DEFAULT_MODEL_ID
from prompt_input import load_constraints, slug
from net_writer import write_net_file


HERE = Path(__file__).parent
DEFAULT_OUT_DIR = HERE / "interface_outputs"


# ── Tab 1 handler: JSON file → .net files ───────────────────────────────

def handle_json(file_obj, n_candidates: int, model_path: str, out_dir: str):
    """Process a JSON constraint file end-to-end."""
    if file_obj is None:
        return "❌ 请先上传 .json 约束文件。", None

    src_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    if not src_path.exists():
        return f"❌ 文件不存在：{src_path}", None

    out_root = Path(out_dir or DEFAULT_OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    log.append(f"📄 输入文件：{src_path.name}")
    log.append(f"📂 输出目录：{out_root}")
    log.append(f"🔢 每条约束候选数：{n_candidates}")

    try:
        constraints = load_constraints(src_path)
    except Exception as e:
        return f"❌ 解析 JSON 失败：{e}", None
    log.append(f"✅ 解析成功，共 {len(constraints)} 条约束\n")

    log.append("⏳ 加载模型（首次约 12-20 秒，后续复用）...")
    t0 = time.perf_counter()
    llm = get_llm(model_id=model_path or DEFAULT_MODEL_ID)
    log.append(f"✅ 模型就绪，耗时 {time.perf_counter()-t0:.1f}s\n")

    written: list[str] = []
    for i, c in enumerate(constraints):
        comment = c.get("_comment", f"constraint_{i:02d}")
        log.append(f"  [{i+1}/{len(constraints)}] {comment}")
        t1 = time.perf_counter()
        try:
            cands = llm.generate_from_constraint(c, n=int(n_candidates))
            label = slug(c, i)
            net_path = write_net_file(out_root / f"{label}.net", cands, c)
            written.append(str(net_path))
            log.append(f"      ✓ {time.perf_counter()-t1:.1f}s → {net_path.name}")
        except Exception as e:
            log.append(f"      ✗ 失败：{e}")

    log.append("")
    log.append(f"📦 已写入 {len(written)} 个 .net 文件：")
    for p in written:
        log.append(f"   - {p}")

    # Return ZIP of all .net files for download convenience
    zip_path = out_root.parent / f"{out_root.name}.zip"
    if written:
        try:
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", out_root)
            log.append(f"\n📥 已打包为：{zip_path}")
        except Exception as e:
            log.append(f"\n⚠ 打包失败：{e}")
            zip_path = None

    return "\n".join(log), str(zip_path) if zip_path and zip_path.exists() else None


# ── Tab 2 handler: raw text → text output ───────────────────────────────

def handle_text(prompt: str, n_candidates: int, model_path: str):
    """Generate from a raw text prompt; no file output."""
    if not prompt or not prompt.strip():
        return "❌ 请输入提示文本。"

    log: list[str] = []
    log.append("⏳ 加载模型（首次约 12-20 秒，后续复用）...")
    t0 = time.perf_counter()
    llm = get_llm(model_id=model_path or DEFAULT_MODEL_ID)
    log.append(f"✅ 模型就绪，耗时 {time.perf_counter()-t0:.1f}s\n")

    log.append(f"⏳ 生成 {n_candidates} 个候选...")
    t1 = time.perf_counter()
    try:
        outs = llm.generate_from_text(prompt, n=n_candidates)
    except Exception as e:
        return f"❌ 生成失败：{e}"
    log.append(f"✅ 生成完毕，耗时 {time.perf_counter()-t1:.1f}s\n")

    for i, o in enumerate(outs, 1):
        log.append(f"\n{'='*60}")
        log.append(f"=== 候选 {i} ===")
        log.append("="*60)
        log.append(o)

    return "\n".join(log)


# ── UI layout ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LLM Topology Generation Demo") as ui:
        gr.Markdown(
            "# LLM SPICE 拓扑生成 Demo\n"
            "基座模型：Qwen2.5-Coder-7B（4-bit NF4 量化）  \n"
            "模型加载约 12-20 秒，仅首次请求触发，之后所有请求复用。"
        )

        with gr.Row():
            model_box = gr.Textbox(
                label="模型路径（HuggingFace repo id 或本地绝对路径）",
                value=DEFAULT_MODEL_ID,
                scale=3,
            )

        with gr.Tabs():
            # Tab 1 — JSON
            with gr.Tab("📁 JSON 约束输入（写文件）"):
                gr.Markdown(
                    "上传 `sample_constraints.json` 格式的 JSON 文件。"
                    "对每条约束自动套用提示词模板，调用模型生成 N 个候选，"
                    "并写入 `.net` 文件到输出目录。"
                )
                with gr.Row():
                    json_file = gr.File(
                        label="约束 JSON 文件",
                        file_types=[".json"],
                        scale=2,
                    )
                    with gr.Column(scale=1):
                        n_json = gr.Slider(
                            1, 8, value=4, step=1,
                            label="每条约束候选数 N",
                        )
                        out_dir_box = gr.Textbox(
                            label="输出目录",
                            value=str(DEFAULT_OUT_DIR),
                        )
                        btn_json = gr.Button(
                            "▶ 生成 .net 文件",
                            variant="primary",
                        )
                json_log = gr.Textbox(
                    label="日志",
                    lines=20,
                    interactive=False,
                )
                json_zip = gr.File(
                    label="打包下载（.zip）",
                    interactive=False,
                )
                btn_json.click(
                    fn=handle_json,
                    inputs=[json_file, n_json, model_box, out_dir_box],
                    outputs=[json_log, json_zip],
                )

            # Tab 2 — Raw text
            with gr.Tab("⌨ 直接文本输入（仅显示）"):
                gr.Markdown(
                    "直接输入任意提示词。**不会** 套用约束模板，**不会** 写文件，"
                    "结果直接在下方显示。"
                )
                with gr.Row():
                    prompt_box = gr.Textbox(
                        label="提示词",
                        placeholder="### Constraint:\n{ \"vin_min\": 12, ... }\n\n### SPICE Netlist:\n",
                        lines=8,
                        scale=3,
                    )
                    with gr.Column(scale=1):
                        n_text = gr.Slider(
                            1, 4, value=1, step=1,
                            label="候选数 N",
                        )
                        btn_text = gr.Button(
                            "▶ 生成", variant="primary"
                        )
                text_log = gr.Textbox(
                    label="生成结果",
                    lines=25,
                    interactive=False,
                )
                btn_text.click(
                    fn=handle_text,
                    inputs=[prompt_box, n_text, model_box],
                    outputs=[text_log],
                )

        gr.Markdown(
            "---\n"
            "**说明：** 模型为单实例懒加载。首次请求加载耗时较长，"
            "之后任意 Tab 的请求都会复用同一模型权重。"
        )
    return ui


def main():
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
