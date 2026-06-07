# LLM Topology Generation — Core Module

This folder ships the **core LLM topology-generation pipeline**.
Only **four** files are needed to load the model and produce SPICE
netlists from constraint dicts:

| File | Role |
|---|---|
| `llm_engine_minimal.py` | Low-level engine — loads the quantized model, runs `model.generate(...)`, cleans the raw output. |
| `prompt_input.py`       | Constraint → prompt template. Injects the **naming-convention rules** the downstream evaluator depends on. |
| `net_writer.py`         | Persists each generated topology to its own `.net` file (one topology per file). |
| `llm_api.py`            | High-level façade. Wraps the three modules above into a single `TopologyLLM` class with three public methods. |

> Web UI (`demo_interface.py`), tests, sample data, and launch scripts are
> **not** part of this core module — they live elsewhere (in `demo/` or
> outside the pipeline). All those tools depend on, but are not required
> by, the four files above.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | `D:\Document\Course\Team_intership\LLM\.venv-gpu\Scripts\python.exe` |
| Required packages | `torch>=2.11+cu128`, `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece` |
| GPU | NVIDIA (≥ 6 GB VRAM, sm_120 supported) |
| Model weights | `D:\Document\Course\Team_intership\LLM\models\qwen25-coder-7b` (~15 GB on disk, ~5.5 GB VRAM with 4-bit NF4) |

The `DEFAULT_MODEL_ID` constant in `llm_api.py` points at the path above —
override via the `model_id=` kwarg if your weights live elsewhere.

---

## 2. Loading the Model

The model is loaded the first time you instantiate `TopologyLLM`
(roughly 12 – 20 s + a one-shot CUDA JIT). After that, generation runs at
about **19 tok/s** on an RTX 5070 Laptop with 4-bit NF4.

```python
from llm_api import TopologyLLM

# 4-bit NF4 (default). Pass quantization="fp16" if you have ≥ 16 GB VRAM.
llm = TopologyLLM()
```

For programs with several modules that all need the LLM, use the
process-wide singleton — it loads exactly once:

```python
from llm_api import get_llm
llm = get_llm()       # first call: loads model
llm2 = get_llm()      # subsequent calls: reuse
assert llm is llm2
```

---

## 3. Public API (`llm_api.TopologyLLM`)

| Method | Input | Template applied? | Writes files? | Returns |
|---|---|---|---|---|
| `generate_from_constraint(constraint, n=4)` | constraint `dict` | ✅ via `make_prompt` | ❌ | `list[str]` — one cleaned netlist per candidate |
| `generate_from_json(json_path, out_dir, n=4)` | path to a JSON list of constraints + output dir | ✅ per item | ✅ **one `.net` file per candidate** | `list[Path]` — flat list of every written file |
| `generate_from_text(prompt, n=1)` | raw prompt string | ❌ (your prompt is fed verbatim) | ❌ | `list[str]` |

### 3.1 In-memory generation from a single constraint

```python
from llm_api import TopologyLLM

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

### 3.2 Batch from JSON, writing one `.net` per topology

```python
from llm_api import TopologyLLM

llm = TopologyLLM()
written = llm.generate_from_json(
    json_path="sample_constraints.json",   # list[dict]
    out_dir="output_netlists",
    n=4,                                   # 4 candidates per constraint
)
print(f"Wrote {len(written)} .net files")
# Each file holds ONE topology, named e.g.
#   00_Buck_12_24V_to_5V_50W_cand1.net
#   00_Buck_12_24V_to_5V_50W_cand2.net
#   ...
```

### 3.3 Free-form prompt (bypass the template)

```python
from llm_api import TopologyLLM

llm = TopologyLLM()
prompt = """Write a SPICE netlist for a flyback converter.
### SPICE Netlist:
"""
outs = llm.generate_from_text(prompt, n=2)
```

### 3.4 Sampling overrides

```python
llm = TopologyLLM(
    model_id=r"D:\path\to\weights",
    quantization="4bit",     # or "fp16"
    max_new_tokens=1024,
    temperature=0.5,
    top_p=0.85,
)
```

---

## 4. Constraint JSON Format

A list of dicts. Five numeric fields are expected; `_comment` (and any
other key starting with `_`) is metadata only and is **not** sent to the
model.

```json
[
  {
    "_comment": "Buck 12-24V to 5V, 50W",
    "vin_min": 12, "vin_max": 24,
    "vout_target": 5,
    "efficiency_target": 0.9,
    "power_in": 50
  }
]
```

---

## 5. Output `.net` File Format

`net_writer.write_netlists(...)` produces **one file per candidate**:

```
output_netlists/
├── 00_Buck_12_24V_to_5V_50W_cand1.net
├── 00_Buck_12_24V_to_5V_50W_cand2.net
├── 00_Buck_12_24V_to_5V_50W_cand3.net
└── 00_Buck_12_24V_to_5V_50W_cand4.net
```

Each file has a comment header followed by the netlist body:

```
* Generated: 2026-04-27T14:05:11
* Constraint: Buck 12-24V to 5V, 50W
*   vin_min = 12
*   vin_max = 24
*   vout_target = 5
*   efficiency_target = 0.9
*   power_in = 50
* Candidate: 1 of 4
*
Vin in 0 12
L1 in sw 100u
M1 sw gnd gnd gnd NMOS
...
.tran 1u 1m
.end
```

The single-candidate writer is also exposed in case you want to drive
the file naming yourself:

```python
from net_writer import write_single_netlist
write_single_netlist(
    path="my_topology.net",
    netlist=netlist_str,
    constraint=constraint_dict,
    candidate_idx=1,
    total_candidates=1,
)
```

---

## 6. Naming Convention (enforced in the prompt)

`prompt_input.NAMING_RULES` is prepended to **every** prompt produced by
`make_prompt`. It targets the regexes used by `netlist_filter.py` and
the SFT/GRPO reward function downstream:

- Input voltage source must be `Vin`
- Output load resistor must be `Rload`
- MOSFET gate driver must be `Vgate`
- Inductors `L*`, capacitors `C*`, diodes `D*`
- `.model NMOS NMOS` / `.model PMOS PMOS` cards required when MOSFETs are used
- Exactly one `.tran <step> <stop>` directive
- Netlist ends with `.end`
- Plain SPICE only — no Markdown fences, no commentary

> **Important:** if you change a token here, also update `netlist_filter.py`
> and the RL reward functions, otherwise rewards will silently mis-score.

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
   (naming rules + JSON)      (load + sample)       (one .net per topology)
```

- **`llm_engine_minimal.py`** is the only file that imports `transformers`,
  `torch`, `bitsandbytes`. Anything that doesn't need the model can stay
  off the GPU dependency stack.
- **`prompt_input.py`** is pure-Python (stdlib only). Safe to import from
  the RL trainer for prompt construction with no GPU side-effects.
- **`net_writer.py`** is pure-Python (stdlib only).
- **`llm_api.py`** is the one facade that wires the three together.

---

## 8. Quick CLI Smoke Test

```powershell
$PY = "D:\Document\Course\Team_intership\LLM\.venv-gpu\Scripts\python.exe"
& $PY llm_api.py
# ↳ instantiates TopologyLLM and runs `generate_from_text("### Hello world:\n", n=1)`
```

```powershell
& $PY prompt_input.py sample_constraints.json
# ↳ prints the slug + full prompt for the first 2 constraints — verify
#   the naming-convention block is present at the top of each prompt.
```

```powershell
& $PY net_writer.py
# ↳ writes 3 dummy candidate files into ./demo_out/ and prints them back.
```
