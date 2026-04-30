# LLM Topology Generation — Core Module

This folder ships the **core LLM topology-generation pipeline**.
Only **four** files are needed to load the model and produce SPICE
netlists from constraint dicts:

| File | Role |
|---|---|
| `llm_engine_minimal.py` | Low-level engine — loads the quantized model, runs `model.generate(...)`, cleans the raw output. |
| `prompt_input.py`       | Constraint → prompt template. Injects the **naming-convention rules** the downstream evaluator/simulator depend on. |
| `net_writer.py`         | Persists each generated topology to its own `.net` file (one topology per file) under the canonical batch layout. |
| `llm_api.py`            | High-level façade. Wraps the three modules above into a single `TopologyLLM` class with four public methods. |

> Web UI, demo scripts, tests, sample constraint files and launch scripts
> are **not** part of this core module — only the four files above are
> required for `main.py` to drive the pipeline end-to-end.

---

## 1. Environment

| Item | Value |
|---|---|
| Python (Windows dev) | `D:\Document\Course\Team_intership\LLM\.venv-gpu\Scripts\python.exe` |
| Required packages | `torch>=2.11`, `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece` |
| GPU | NVIDIA (≥ 6 GB VRAM, sm_120 / Blackwell supported by `cu128`) |
| Model weights | `D:\Document\Course\Team_intership\LLM\models\qwen25-coder-7b` (~15 GB on disk, ~5.5 GB VRAM with 4-bit NF4) |

The `DEFAULT_MODEL_ID` constant in `llm_api.py` points at the path above —
override via the `model_id=` kwarg if your weights live elsewhere.

> **Cross-platform note**: all directory names emitted by this module are
> lowercase (`pipeline/data/<batchID>/llm_output/`) to stay portable
> across Windows and Linux.

---

## 2. Loading the Model

The model is loaded the first time you instantiate `TopologyLLM`
(roughly 12 – 20 s + a one-shot CUDA JIT). After that, generation runs at
about **19 tok/s** on an RTX 5070 Laptop with 4-bit NF4.

```python
from pipeline.llm_topology_generation.llm_api import TopologyLLM

# 4-bit NF4 (default). Pass quantization="fp16" if you have ≥ 16 GB VRAM.
llm = TopologyLLM()
```

For programs with several modules that all need the LLM, use the
**process-wide, thread-safe singleton** — it loads exactly once:

```python
from pipeline.llm_topology_generation.llm_api import get_llm

llm  = get_llm()    # first call: loads model
llm2 = get_llm()    # subsequent calls: reuse, no reload
assert llm is llm2
```

---

## 3. Public API (`llm_api.TopologyLLM`)

| Method | Input | Template applied? | Writes files? | Returns |
|---|---|---|---|---|
| `generate_from_constraint(constraint, n=4)` | constraint `dict` | ✅ via `make_prompt` | ❌ | `list[str]` — one cleaned netlist per candidate |
| `generate_for_batch(constraint, batchID, n=4)` | constraint `dict` + batch identifier | ✅ via `make_prompt` | ✅ → `data/<batchID>/llm_output/topN.net` | `list[Path]` — written file paths |
| `generate_from_json(json_path, batchID, n=4)` | path to a JSON list of constraints + batch id | ✅ per item | ✅ contiguous numbering across all constraints | `list[Path]` — flat list of all written files |
| `generate_from_text(prompt, n=4)` | raw prompt string | ❌ (your prompt is fed verbatim) | ❌ | `list[str]` |

`generate_for_batch` is the standard pipeline entry point — `main.py`
calls exactly this.

### 3.1 Single constraint, in-memory

```python
from pipeline.llm_topology_generation.llm_api import TopologyLLM

llm = TopologyLLM()
constraint = {
    "_comment": "Buck 12-24V to 5V, 50W",
    "vin_min": 12, "vin_max": 24,
    "vout_target": 5,
    "efficiency_target": 0.9,
    "power_in": 50,
}
candidates = llm.generate_from_constraint(constraint, n=4)
for i, net in enumerate(candidates, 1):
    print(f"=== Candidate {i} ===\n{net}\n")
```

### 3.2 Single constraint, written to the canonical batch folder

```python
from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.llm_topology_generation.prompt_input import load_constraint

llm = TopologyLLM()
constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=0)

paths = llm.generate_for_batch(constraint, batchID="batch_3", n=4)
# Files written:
#   pipeline/data/batch_3/llm_output/top1.net
#   pipeline/data/batch_3/llm_output/top2.net
#   pipeline/data/batch_3/llm_output/top3.net
#   pipeline/data/batch_3/llm_output/top4.net
```

### 3.3 Whole JSON dataset → one batch folder

```python
llm.generate_from_json(
    json_path="pipeline/data/datasets/constraints.json",
    batchID="batch_3",
    n=4,
)
# With K constraints and n=4, you get K*4 files top1.net..top{K*4}.net
# (numbering is global to the batch — nothing gets overwritten).
```

### 3.4 Free-form prompt (bypass the template)

```python
prompt = """Write a SPICE netlist for a flyback converter.
### SPICE Netlist:
"""
outs = llm.generate_from_text(prompt, n=2)
```

### 3.5 Sampling overrides

```python
llm = TopologyLLM(
    model_id="/path/to/weights",
    quantization="4bit",     # or "fp16"
    max_new_tokens=1024,
    temperature=0.5,
    top_p=0.85,
)
```

`temperature` and `top_p` are forwarded to the underlying `LLMEngine`
constructor — no hidden private-attribute mutation.

---

## 4. Constraint JSON Format

A list of dicts. Five numeric fields are expected; `_comment` (and any
other key starting with `_`) is metadata only and is **not** sent to the
model.

```json
[
  {
    "_comment": "Buck 12-24V to 5V, 50W",
    "vin_min": 12,
    "vin_max": 24,
    "vout_target": 5,
    "efficiency_target": 0.9,
    "power_in": 50
  }
]
```

Use `prompt_input.load_constraint(path, idx=0)` to grab one entry by
index, or `load_constraints(path)` for the whole list.

---

## 5. Output `.net` File Format

Each call to `generate_for_batch` (or `generate_from_json`) creates the
batch folder if it does not exist and writes **one topology per file**:

```
pipeline/data/batch_3/
├── llm_output/
│   ├── top1.net
│   ├── top2.net
│   ├── top3.net
│   └── top4.net
└── (validation_results.json, simulation_results.csv, reward_results.json
   are produced by downstream stages)
```

File contents are **pure SPICE — no comment header, no Markdown fences**:

```
Vin in 0 12
M1 in gate sw 0 NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out 100u
C1 out 0 470u
Rload out 0 0.278
Vgate gate 0 PULSE(0 12 0 1n 1n 4.16667e-06 1e-05)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end
```

Empty / whitespace-only candidates from the LLM are **dropped with a
`RuntimeWarning`** so they never produce a 0-byte `.net` that would
crash the validator.

If you need to manage file paths yourself:

```python
from pipeline.llm_topology_generation.net_writer import (
    write_netlists, write_single_netlist, get_llm_output_dir,
)

# Resolve the canonical folder for a batch (does NOT create it):
get_llm_output_dir("batch_3")
# → pipeline/data/batch_3/llm_output

# Append more candidates to an existing batch (no overwrite):
write_netlists(more_nets, batchID="batch_3", start_index=5)
# → top5.net, top6.net, ...

# Write a single explicit file:
write_single_netlist("/abs/path/my.net", netlist_str)
```

---

## 6. Naming Convention (enforced in the prompt)

`prompt_input.NAMING_RULES` is prepended to **every** prompt built by
`make_prompt`. It is the contract between this module and the
downstream validator (`pipeline/netlist_validation/validator.py`) and
the reward functions:

- Input voltage source must be `Vin`
- Output load resistor must be `Rload`
- Gate drive must be `Vgate` (or `Vgate1..N`), as a 7-parameter `PULSE`
- MOSFET pin order: `M<name> drain gate source bulk model`
  - high-side : `drain=in, gate=gate, source=sw, bulk=0`
  - low-side  : `drain=sw, gate=gate2, source=0, bulk=0`
- Required model cards: `.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)` and
  `.model DIODE D` (the latter whenever any `D*` is used)
- Component prefix universe: `V` (sources), `M` (MOSFETs), `D` (diodes),
  `L` (inductors), `C` (capacitors), `R` (resistors — `Rload` for load,
  `Rbleed*` for floating-node tie-down)
- Required nodes: `in`, `out`, `sw`/`sw1..N`, `gate`/`gate1..N`, `0` (GND)
- Floating nodes need a `Rbleed <node> 0 1Meg`
- Output is plain SPICE only — no Markdown, no commentary

> **Important:** if you rename a token in `NAMING_RULES`, also update
> `pipeline/netlist_validation/validator.py` and any reward functions
> that grep for that token, otherwise rewards will silently mis-score.

---

## 7. Module Boundaries

```
                       ┌────────────────────────┐
   constraint dict ─►  │      llm_api.py        │  ◄── public entry point
                       │   class TopologyLLM    │
                       └─────────┬──────────────┘
                                 │
              ┌──────────────────┼─────────────────────┐
              ▼                  ▼                     ▼
   prompt_input.make_prompt   LLMEngine.generate    net_writer.write_netlists
   (naming rules + JSON)      (load + sample)       (top<N>.net, no header)
```

- **`llm_engine_minimal.py`** is the only file that imports `transformers`,
  `torch`, `bitsandbytes`. Anything that doesn't need the model can stay
  off the GPU dependency stack.
- **`prompt_input.py`** is pure-Python (stdlib only). Safe to import from
  the RL trainer for prompt construction with no GPU side-effects.
- **`net_writer.py`** is pure-Python (stdlib only). Resolves
  `pipeline/data/<batchID>/llm_output/` from `__file__`, so it works
  regardless of cwd.
- **`llm_api.py`** is the one facade that wires the three together and
  owns the singleton.

---

## 8. Quick CLI Smoke Tests

```bash
# From the repo root
python -m pipeline.llm_topology_generation.prompt_input
# ↳ prints the full prompt for the first 2 constraints in
#   pipeline/data/datasets/constraints.json — verify NAMING_RULES is
#   present at the top.

python -m pipeline.llm_topology_generation.net_writer
# ↳ writes 2 dummy top<N>.net files into ./demo_out/ and prints them
#   back. Verifies pure-SPICE format (no header).
```

For the full pipeline (model load + generate + validate + simulate +
reward), see the project root `main.py`.
