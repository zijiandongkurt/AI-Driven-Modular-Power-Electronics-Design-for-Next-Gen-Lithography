"""
run_gui.py — Interactive Gradio GUI for the Power Electronics Topology Generator.

Usage (local):
    python run_gui.py

Usage (Snellius, public link):
    sbatch run_gui.slurm
    # The public URL will appear in ./logs/slurm-gui-<JOBID>.out
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.llm_topology_generation.prompt_input import load_constraints

# ── Lazy model cache ──────────────────────────────────────────────────────────

_llm: TopologyLLM | None = None
_loaded_model_id: str | None = None
_loaded_lora_path: str | None = None


def _get_llm(model_id: str, lora_path: str, max_tokens: int) -> TopologyLLM:
    global _llm, _loaded_model_id, _loaded_lora_path

    lora_path = lora_path.strip() or None

    if (
        _llm is None
        or _loaded_model_id != model_id
        or _loaded_lora_path != lora_path
    ):
        _llm = TopologyLLM(
            model_id=model_id,
            max_new_tokens=int(max_tokens),
            lora_path=lora_path,
        )
        _loaded_model_id = model_id
        _loaded_lora_path = lora_path

    return _llm


# ── Quick structural check (no LTSpice needed) ────────────────────────────────

_REQUIRED_KEYWORDS = [".end", "Vin ", "Rload ", ".tran", ".model NMOS", ".model DIODE"]

def _validate_quick(netlist: str) -> str:
    if not netlist.strip():
        return ""
    missing = [kw for kw in _REQUIRED_KEYWORDS if kw not in netlist]
    if missing:
        return f"⚠️  Missing: {', '.join(missing)}"
    return "✅ Structurally valid"


# ── Preset loader ─────────────────────────────────────────────────────────────

_PRESET_FILES = {
    "Easy":   "pipeline/data/datasets/constraints_easy.json",
    "Medium": "pipeline/data/datasets/constraints_medium.json",
    "Hard":   "pipeline/data/datasets/constraints_hard.json",
}

def _load_presets() -> dict[str, dict]:
    presets = {"— select a preset —": None}
    for difficulty, path in _PRESET_FILES.items():
        try:
            constraints = load_constraints(path)
            for i, c in enumerate(constraints):
                label = f"[{difficulty}] {c.get('description', f'idx {i}')}"
                presets[label] = c
        except Exception:
            pass
    return presets

_PRESETS = _load_presets()


def apply_preset(preset_label):
    c = _PRESETS.get(preset_label)
    if c is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    return (
        c.get("description", ""),
        c.get("vin_min", 0),
        c.get("vin_max", 0),
        c.get("vout_target", 0),
        c.get("efficiency_target", 0.85),
        c.get("power_in", 0),
    )


# ── Generation function ───────────────────────────────────────────────────────

def generate_netlists(
    description, vin_min, vin_max, vout_target, efficiency_target, power_in,
    n_candidates, model_id, lora_path, max_tokens,
):
    try:
        constraint = {
            "description":       description,
            "vin_min":           float(vin_min),
            "vin_max":           float(vin_max),
            "vout_target":       float(vout_target),
            "efficiency_target": float(efficiency_target),
            "power_in":          float(power_in),
        }

        llm = _get_llm(model_id, lora_path, int(max_tokens))
        netlists = llm.generate_from_constraint(constraint, n=int(n_candidates))

        # Pad to 4 outputs
        padded = (netlists + [""] * 4)[:4]
        validations = [_validate_quick(n) for n in padded]

        return (
            padded[0], validations[0],
            padded[1], validations[1],
            padded[2], validations[2],
            padded[3], validations[3],
            "✅ Done",
        )

    except Exception as e:
        empty = ("", "") * 4
        return *empty, f"❌ {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Power Electronics Topology Generator") as demo:

    gr.Markdown("# ⚡ AI Power Electronics Topology Generator")
    gr.Markdown(
        "Define your DC/DC converter constraints, choose a model checkpoint, "
        "and generate candidate SPICE netlists."
    )

    with gr.Row():

        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### Preset Constraints")
            preset_dd = gr.Dropdown(
                choices=list(_PRESETS.keys()),
                value="— select a preset —",
                label="Load from dataset",
            )

            gr.Markdown("### Design Constraints")
            description    = gr.Textbox(label="Description",            value="12V to 5V Buck Converter")
            with gr.Row():
                vin_min    = gr.Number(label="Vin min (V)",             value=10.0)
                vin_max    = gr.Number(label="Vin max (V)",             value=14.0)
            with gr.Row():
                vout_target = gr.Number(label="Vout target (V)",        value=5.0)
                power_in    = gr.Number(label="Power in (W)",           value=20.0)
            efficiency_target = gr.Slider(
                label="Efficiency target",
                minimum=0.50, maximum=0.99, step=0.01, value=0.85,
            )

            gr.Markdown("### Model Settings")
            model_id   = gr.Textbox(label="Model ID",                  value="Qwen/Qwen3-14B")
            lora_path  = gr.Textbox(
                label="LoRA path (optional)",
                placeholder="e.g. checkpoints/sft/sft_002/final",
            )
            with gr.Row():
                n_candidates = gr.Slider(label="Candidates", minimum=1, maximum=4, step=1, value=2)
                max_tokens   = gr.Slider(label="Max tokens",  minimum=256, maximum=2048, step=128, value=1536)

            generate_btn = gr.Button("⚡ Generate Netlists", variant="primary", size="lg")
            status_box   = gr.Textbox(label="Status", interactive=False)

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Generated Netlists")

            outputs = []
            with gr.Tabs():
                for i in range(1, 5):
                    with gr.TabItem(f"Candidate {i}"):
                        val_box = gr.Textbox(label="Validation", interactive=False, lines=1)
                        net_box = gr.Textbox(label="SPICE Netlist", lines=28, max_lines=40)
                        outputs.extend([net_box, val_box])

    # ── Wire preset loader ────────────────────────────────────────────────────
    preset_dd.change(
        fn=apply_preset,
        inputs=[preset_dd],
        outputs=[description, vin_min, vin_max, vout_target, efficiency_target, power_in],
    )

    # ── Wire generate button ──────────────────────────────────────────────────
    # outputs order: net1, val1, net2, val2, net3, val3, net4, val4, status
    generate_btn.click(
        fn=generate_netlists,
        inputs=[
            description, vin_min, vin_max, vout_target, efficiency_target, power_in,
            n_candidates, model_id, lora_path, max_tokens,
        ],
        outputs=[
            outputs[0], outputs[1],
            outputs[2], outputs[3],
            outputs[4], outputs[5],
            outputs[6], outputs[7],
            status_box,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
