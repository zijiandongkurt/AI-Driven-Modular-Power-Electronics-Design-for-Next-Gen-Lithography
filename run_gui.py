"""
run_gui.py — Zycos Testing Environment

Usage (local):
    python run_gui.py

Usage (Snellius):
    sbatch scripts/run_gui.slurm
    # URL appears in ./logs/slurm-gui-<JOBID>.out
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
import tempfile
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
import gradio as gr
from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.llm_topology_generation.prompt_input import load_constraints

#  Lazy model cache 

_llm: TopologyLLM | None = None
_loaded_key: tuple | None = None


def _get_llm(checkpoint_label: str, max_tokens: int, quantization: str) -> TopologyLLM:
    global _llm, _loaded_key

    lora_path = _CHECKPOINTS.get(checkpoint_label)
    quant     = quantization if quantization != "None" else None
    key       = (checkpoint_label, max_tokens, quant)

    if _llm is None or _loaded_key != key:
        _llm = TopologyLLM(
            model_id="Qwen/Qwen3-14B",
            max_new_tokens=int(max_tokens),
            lora_path=lora_path,
            quantization=quant,
        )
        _loaded_key = key

    return _llm


#  Checkpoint scanner 

def _scan_checkpoints() -> dict[str, str | None]:
    result: dict[str, str | None] = {"Base model (no adapter)": None}
    root = Path("checkpoints")
    if not root.exists():
        return result

    for run_dir in sorted(root.glob("zycos_*"), reverse=True):
        num   = run_dir.name.split("_", 1)[-1].lstrip("0") or "0"
        label = f"Zycos {num}"
        adapter = run_dir / "grpo-lora" / "final" / "grpo"
        result[label] = str(adapter) if adapter.exists() else str(run_dir)

    return result

_CHECKPOINTS = _scan_checkpoints()


#  Validation 

_REQUIRED = [".end", "Vin ", "Rload ", ".tran", ".model NMOS", ".model DIODE"]

def _validate_html(netlist: str) -> str:
    if not netlist.strip():
        return '<p class="val-empty">No output generated.</p>'
    missing = [kw for kw in _REQUIRED if kw not in netlist]
    if missing:
        return (
            f'<p class="val-fail">Validation failed &mdash; '
            f'missing: <code>{", ".join(missing)}</code></p>'
        )
    return '<p class="val-pass">Structurally valid</p>'


#  Preset loader 

_PRESET_FILES = {
    "Easy":   "pipeline/data/datasets/constraints_easy.json",
    "Medium": "pipeline/data/datasets/constraints_medium.json",
    "Hard":   "pipeline/data/datasets/constraints_hard.json",
}

def _load_presets() -> dict[str, dict | None]:
    presets: dict[str, dict | None] = {"— none —": None}
    for difficulty, path in _PRESET_FILES.items():
        try:
            for i, c in enumerate(load_constraints(path)):
                label = f"[{difficulty}]  {c.get('_comment', f'#{i}')}"
                presets[label] = c
        except Exception:
            pass
    return presets

_PRESETS = _load_presets()


def download_netlist(netlist: str, candidate: int):
    if not netlist.strip():
        return None
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".net",
        prefix=f"zycos_candidate{candidate}_",
        mode="w",
    )
    tmp.write(netlist)
    tmp.close()
    return tmp.name


def apply_preset(label):
    c = _PRESETS.get(label)
    if c is None:
        return "", 0, 0, 0, 0.85, 0
    return (
        c.get("_comment", ""),
        c.get("vin_min", 0),
        c.get("vin_max", 0),
        c.get("vout_target", 0),
        c.get("efficiency_target", 0.85),
        c.get("power_in", 0),
    )


#  Generation 

def generate_netlists(
    description, vin_min, vin_max, vout_target, efficiency_target, power_in,
    n_candidates, checkpoint_label, max_tokens, quantization,
):
    try:
        constraint = {
            "_comment":          description,
            "vin_min":           float(vin_min),
            "vin_max":           float(vin_max),
            "vout_target":       float(vout_target),
            "efficiency_target": float(efficiency_target),
            "power_in":          float(power_in),
        }
        llm      = _get_llm(checkpoint_label, int(max_tokens), quantization)
        netlists = llm.generate_from_constraint(constraint, n=int(n_candidates))
        padded   = (netlists + [""] * 4)[:4]
        tab_visibility = [gr.update(visible=(i < int(n_candidates))) for i in range(4)]
        return (
            padded[0], _validate_html(padded[0]),
            padded[1], _validate_html(padded[1]),
            padded[2], _validate_html(padded[2]),
            padded[3], _validate_html(padded[3]),
            "Generation complete.",
            gr.update(visible=False),
            gr.update(visible=True),
            *tab_visibility,
        )
    except Exception as e:
        return (
            "", _validate_html(""), "", _validate_html(""),
            "", _validate_html(""), "", _validate_html(""),
            f"Error: {e}",
            gr.update(visible=True),
            gr.update(visible=False),
            *[gr.update(visible=(i < 2)) for i in range(4)],
        )


#  CSS 

CSS = """
footer { display: none !important; }

.zycos-header { border-bottom: 2px solid #cbd5e1; padding-bottom: 10px; margin-bottom: 4px; }
.zycos-header h1 { font-size: 1.35rem !important; font-weight: 700; margin: 0; }
.zycos-header p  { font-size: 0.8rem; color: #64748b; margin: 2px 0 0; }

.panel-divider { border-right: 1px solid #e2e8f0; padding-right: 16px; }

.section-head {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    padding: 12px 0 4px;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 6px;
}

.idle-panel {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 420px;
    border: 1px dashed #cbd5e1;
    border-radius: 6px;
    color: #94a3b8;
    font-size: 0.85rem;
    text-align: center;
    line-height: 2;
}

.val-pass {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin: 4px 0 8px;
}

.val-fail {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin: 4px 0 8px;
}

.val-empty {
    color: #cbd5e1;
    font-size: 0.78rem;
    margin: 4px 0 8px;
}
"""


#  UI 

with gr.Blocks(title="Zycos Testing Environment", css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML("""
        <div class="zycos-header">
            <h1>Zycos Testing Environment</h1>
            <p>AI-Driven Power Electronics Topology Generator &mdash; TU/e &times; ASML</p>
        </div>
    """)

    with gr.Row():

        #  Left panel 
        with gr.Column(scale=1, elem_classes="panel-divider"):

            gr.HTML('<div class="section-head">Preset</div>')
            preset_dd = gr.Dropdown(
                choices=list(_PRESETS.keys()),
                value="— none —",
                label="Constraint preset",
                container=False,
            )

            gr.HTML('<div class="section-head">Design Constraints</div>')
            description = gr.Textbox(label="Description", value="12V to 5V Buck Converter")
            with gr.Row():
                vin_min = gr.Number(label="Vin min (V)", value=10.0)
                vin_max = gr.Number(label="Vin max (V)", value=14.0)
            with gr.Row():
                vout_target = gr.Number(label="Vout target (V)", value=5.0)
                power_in    = gr.Number(label="Power in (W)",    value=20.0)
            efficiency_target = gr.Slider(
                label="Efficiency target", minimum=0.50, maximum=0.99, step=0.01, value=0.85,
            )

            gr.HTML('<div class="section-head">Model</div>')
            checkpoint_dd = gr.Dropdown(
                choices=list(_CHECKPOINTS.keys()),
                value=list(_CHECKPOINTS.keys())[0],
                label="Checkpoint",
            )
            quantization = gr.Radio(
                choices=["None", "4bit"],
                value="4bit",
                label="Quantization",
                info="4bit recommended for GPUs with less than 24 GB VRAM",
            )
            with gr.Row():
                n_candidates = gr.Slider(label="Candidates", minimum=1, maximum=4, step=1, value=2)
                max_tokens   = gr.Slider(label="Max tokens",  minimum=256, maximum=2048, step=128, value=1536)

            gr.HTML('<div style="height:8px"></div>')
            generate_btn = gr.Button("Generate", variant="primary", size="lg")
            status_box   = gr.Textbox(label="Status", interactive=False, lines=1)

        #  Right panel 
        with gr.Column(scale=2):

            gr.HTML('<div class="section-head">Output</div>')

            # Idle state
            idle_html = gr.HTML(
                '<div class="idle-panel">No results yet.<br>'
                'Configure constraints on the left and click Generate.</div>',
                visible=True,
            )

            # Results (hidden until first generation)
            with gr.Column(visible=False) as results_col:
                outputs = []
                dl_btns = []
                tabs    = []
                with gr.Tabs():
                    for i in range(1, 5):
                        with gr.Tab(f"Candidate {i}", visible=(i <= 2)) as tab:
                            val_html = gr.HTML("")
                            net_box  = gr.Textbox(
                                label="SPICE netlist", lines=30, max_lines=50,
                            )
                            with gr.Row():
                                dl_btn  = gr.Button("Download .net", size="sm", variant="secondary")
                                dl_file = gr.File(label="", visible=False)
                            outputs.extend([net_box, val_html])
                            dl_btns.append((dl_btn, dl_file, i))
                            tabs.append(tab)

    #  Wiring 
    preset_dd.change(
        fn=apply_preset,
        inputs=[preset_dd],
        outputs=[description, vin_min, vin_max, vout_target, efficiency_target, power_in],
    )

    generate_btn.click(
        fn=generate_netlists,
        inputs=[
            description, vin_min, vin_max, vout_target, efficiency_target, power_in,
            n_candidates, checkpoint_dd, max_tokens, quantization,
        ],
        outputs=[
            outputs[0], outputs[1], outputs[2], outputs[3],
            outputs[4], outputs[5], outputs[6], outputs[7],
            status_box, idle_html, results_col,
            tabs[0], tabs[1], tabs[2], tabs[3],
        ],
    )

    for dl_btn, dl_file, idx in dl_btns:
        net_box = outputs[(idx - 1) * 2]
        dl_btn.click(
            fn=lambda txt, i=idx: (download_netlist(txt, i), gr.update(visible=True)),
            inputs=[net_box],
            outputs=[dl_file, dl_file],
        )


if __name__ == "__main__":
    import os
    host = "127.0.0.1" if os.environ.get("SLURM_JOB_ID") is None else "0.0.0.0"
    demo.launch(server_name=host, server_port=7860, share=False)