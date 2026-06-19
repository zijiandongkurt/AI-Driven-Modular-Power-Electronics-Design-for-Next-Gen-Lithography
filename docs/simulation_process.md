# Simulation Process

## What it does

LTspice cannot run natively on Snellius (HPC). Instead, simulations are offloaded to a local Windows PC where LTspice is installed. Snellius sends netlists to the PC, the PC simulates them and sends results back.

---

## How it works

```
Snellius                          PC (Windows)
────────────────────────          ──────────────────────────
training_loop calls               simulation_client.py
simulate(batchID)                 runs continuously in background
     │                                    │
     │  creates job file                  │  detects job
     ▼                                    ▼
snellius/jobs/<batch>.json  ──►  downloads .net files
     │                                    │
     │  waits...                          │  runs LTspice (native, fast)
     │                                    │  extracts metrics
     │                                    │  deletes .raw files
     │                                    │
     │  CSV arrives          ◄──  uploads simulation_results.csv
     ▼
continues to reward function
```

---

## File Overview

| File | Location | Purpose |
|------|----------|---------|
| `simulation_server.py` | `pipeline/simulation/snellius/` | Snellius side — creates jobs, waits for results |
| `simulation_client.py` | `pipeline/simulation/local/` | PC side — polls Snellius, runs simulations |
| `ltspice_runner.py` | `pipeline/simulation/local/` | Runs LTspice via PyLTSpice SimRunner |
| `raw_extractor.py` | `pipeline/simulation/local/` | Parses `.raw` files → `simulation_results.csv` |
| `config.json` | repo root | Machine-specific settings (SSH key, LTspice path) |

---

## Setup (one time)

**1. SSH key** — already configured, no password needed between PC and Snellius.

**2. Config** — edit `config.json` at repo root with your machine's paths:
```json
{
    "snellius_host": "snellius.surf.nl",
    "snellius_user": "akumas",
    "ssh_key_path": "~/.ssh/snellius_key",
    "snellius_repo": "AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography",
    "ltspice_exe": "C:\\Users\\Pc\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe",
    "parallel_sims": 4,
    "poll_interval_s": 3
}
```

**3. Dependencies** — install in your PC venv:
```powershell
.venv\Scripts\pip install paramiko
```

---

## Running

**Before every training run**, start the client on your PC:
```powershell
.venv\Scripts\python pipeline\simulation\local\simulation_client.py
```

Keep it running for the duration of the training run. It will automatically pick up jobs as they arrive.

---

## Output

`simulation_results.csv` is written to `pipeline/data/<Run_XXX>/<batch_X>/` on Snellius after each batch. The reward function reads it from there directly.

---

## Training Loop Integration

```python
# Replace old import
from pipeline.simulation.snellius.simulation_server import SimulationServer as LTSpiceSimulator
```

No other changes needed — the interface is identical to the old runner.
