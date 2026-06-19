# Zycos — AI-Driven Power Electronics Design

![Zycos](zycos.png)

**TU/e × ASML Internship Project**

Zycos is a reinforcement learning framework that trains a large language model to autonomously generate, simulate, and improve SPICE netlists for non-isolated DC/DC converters. Given a set of electrical constraints (input voltage, output voltage, efficiency target, power rating), the model proposes circuit topologies, which are validated, simulated in LTSpice, and scored — closing a feedback loop that progressively improves design quality.

---

## How it works

```
Constraints JSON
      │
      ▼
 Qwen3-14B + LoRA  ──generates──►  SPICE Netlist(s)
                                          │
                                    Validator (23 checks)
                                          │
                              ┌───────────┴──────────┐
                           invalid                 valid
                              │                      │
                    validation penalty         LTSpice simulation
                    (reward ∈ [-1, -0.5])     (Snellius → Windows PC)
                                                      │
                                              RawExtractor
                                           (volume model)
                                                      │
                                           RewardFunctionNorm
                                           (reward ∈ [0.5, 1.0])
                                                      │
                                              GRPO update
                                         (LoRA weights updated)
                                                      │
                                         History + parent selection
                                    (ε-greedy Top-K softmax sampling)
                                                      │
                                              next batch ◄──┘
```

**Training curriculum:** SFT (supervised warm-up) → GRPO Phase 1 (easy) → Phase 2 (medium) → Phase 3 (hard constraints).

---

## Repository Structure

```
├── pipeline/
│   ├── llm_topology_generation/   # LLM engine, prompt builder, .net file writer
│   ├── netlist_validation/        # 23-check structural validator
│   ├── simulation/
│   │   ├── local/                 # Windows PC: LTSpice runner, raw extractor
│   │   └── snellius/              # HPC: simulation job server
│   ├── reward_evaluation/         # RewardFunctionNorm (GRPO) + legacy
│   ├── reinforcement_algorithm/   # GRPO trainer, RL updater
│   ├── utility/                   # Topology hasher (WL graph hash), summary logger
│   ├── graphs_and_visualizations/ # Training plots
│   ├── SFT/                       # SFT dataset builders and trainer
│   └── data/                      # Generated netlists, simulation results, SFT data
├── experiments/                   # Benchmark runner and results
├── test/                          # 179-test suite (pytest)
├── docs/                          # Project documentation
├── meetings/                      # Meeting notes
├── checkpoints/                   # Saved LoRA adapters
├── training_loop_random.py        # Main GRPO training loop (runs on Snellius)
├── configs/                       # JSON configuration files
│   ├── training_config.json       # Hyperparameters for the training run
│   ├── simulation_config.json     # SSH + LTSpice paths (machine-specific)
│   ├── sft_config.json            # SFT trainer settings
│   ├── benchmark_config.json      # Benchmark run settings
│   └── test_config.json           # Quick inference test settings
└── scripts/                       # SLURM job scripts for Snellius
    ├── run_training.slurm         # GRPO training run
    ├── run_sft.slurm              # SFT warm-up
    ├── run_inference.slurm        # Batch inference
    ├── run_benchmark.slurm        # Benchmark evaluation
    └── run_gui.slurm              # GUI/visualisation server
```

---

## Setup

### Requirements

- Python 3.11
- CUDA-capable GPU (training: H100 on Snellius; inference: ≥ 6 GB VRAM)
- LTSpice installed on Windows (simulation client only)

### Install

```bash
git lfs install
git clone https://github.com/zijiandongkurt/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
cd AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
pip install -r requirements.txt
```

> Checkpoints and presentations are stored with git LFS. Run `git lfs install` once per machine before cloning.

### Simulation bridge (Windows PC)

Edit `configs/simulation_config.json` with your machine paths:

```json
{
    "snellius_host": "snellius.surf.nl",
    "snellius_user": "<your_username>",
    "ssh_key_path": "~/.ssh/snellius_key",
    "snellius_repo": "AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography",
    "ltspice_exe": "C:\\Users\\...\\LTspice.exe",
    "poll_interval_s": 3
}
```

Before every training run, start the simulation client on your PC:

```powershell
.venv\Scripts\python pipeline\simulation\local\simulation_client.py
```

---

## Running a Training Run

All training runs on Snellius. Configure `configs/training_config.json` then submit:

```bash
sbatch scripts/run_training.slurm
```

Or run directly on an allocated node:

```bash
python training_loop_random.py
```

Key config fields:

```json
{
    "n_runs": 10,
    "n_batch": 10,
    "sft_lora_path": "checkpoints/zycos_009/grpo-lora/final/grpo",
    "constraint_path": "pipeline/data/datasets/constraints_hard.json",
    "seed_prompts": 4,
    "parents_per_batch": 4,
    "outputs_per_parent": 3,
    "top_k": 9,
    "epsilon": 0.3
}
```

Results are written to `pipeline/data/zycos_XXX/` after each batch.

---

## Testing Environment (GUI)

An interactive Gradio interface for generating and inspecting SPICE netlists without running the full training loop.

**On Snellius:**
```bash
sbatch scripts/run_gui.slurm
tail -f logs/slurm-gui-<JOBID>.out  # prints the SSH tunnel command
```

Then on your local machine run the printed tunnel command and open `http://localhost:7860`.

**Locally** (no model loading — layout preview only):
```bash
gradio run_gui.py
```

---

## Running Tests

```bash
python -m pytest test/ -v
```

179 tests covering: validator, topology hasher, reward functions, RL loop utilities, prompt/netlist generation, raw extractor physics, and benchmark infrastructure. The two tests in `test_simulation.py` require LTSpice to be installed locally.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Grouped GRPO | Candidates are grouped by parent; advantages normalized within groups — prevents high-fitness parents from dominating gradient signal |
| ε-greedy Top-K softmax parent selection | Balances exploitation of best topologies with exploration of the long tail |
| Topological hashing (WL graph hash) | Detects structurally duplicate circuits regardless of node naming or line order |
| Validation reward in [-1, -0.5] | Separated from simulation reward [0.5, 1.0] — any simulated topology always outranks any invalid one |
| Volume model | `V_total = V_fixed + K_L·ΣL + K_H/R_th` — physically grounded surrogate without running FEM |
| LTSpice offloaded to Windows PC | LTSpice cannot run on Snellius (Linux HPC); SSH job queue bridges the two machines |

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/simulation_process.md](docs/simulation_process.md) | Snellius ↔ PC simulation bridge — setup, config, architecture |
| [docs/llm_topology_generation.md](docs/llm_topology_generation.md) | LLM module API, prompt format, naming conventions |
| [docs/sft_data.md](docs/sft_data.md) | SFT dataset structure, rebuild instructions, design decisions |
| [docs/sft_briefing.md](docs/sft_briefing.md) | SFT setup, training data, loss curves, before/after example |

---

## Team

| Name | Role |
|------|------|
| Atakan Kumas | training setup, validation, simulation bridge |
| Zijian Dong | GRPO trainer, RL updater, documentation |
| Chris Hogendoorn| Reward function, test suite, benchmarking |
| Yuhao Shan | LLM generation module, SFT pipeline |
