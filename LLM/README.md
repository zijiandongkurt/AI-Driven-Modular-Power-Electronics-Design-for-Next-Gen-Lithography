# LLM Module

This module provides the core LLM (Large Language Model) components for AI-driven SPICE netlist generation in the power electronics design pipeline.

## Overview

The module uses **Qwen2.5-Coder-7B** as the base model and follows a two-stage training strategy:

1. **SFT (Supervised Fine-Tuning)** — warm-start the model on (constraint, netlist) pairs
2. **GRPO (Group Relative Policy Optimization)** — refine the model using reward signals from simulation

## Files

| File | Description |
|------|-------------|
| `llm_engine_minimal.py` | Core inference engine. Loads the base model, manages LoRA adapters, and generates SPICE netlists from design constraints. |
| `sft_trainer.py` | SFT training module. Fine-tunes the base model on constraint-netlist pairs using LoRA, then saves/merges the adapter. |
| `rl_updater.py` | RL policy update module. Receives external reward scores and updates LoRA weights via GRPO algorithm. |


## Quick Start

### Inference

```python
from llm_engine_minimal import LLMEngine

engine = LLMEngine("Qwen/Qwen2.5-Coder-7B")
engine.load_adapter("sft", "./checkpoints/sft-lora/final")

output = engine.generate({"vin": 12, "vout_target": 5})
print(output.netlist)
```

### SFT Training

```python
from sft_trainer import SFTWarmStart, SFTConfig

config = SFTConfig(data_path="./data/sft_pairs.jsonl")
trainer = SFTWarmStart("Qwen/Qwen2.5-Coder-7B", config)
adapter_path = trainer.train()
```

### RL Update

```python
from rl_updater import RLUpdater, RLConfig

engine = LLMEngine("./checkpoints/sft-merged")
updater = RLUpdater(engine, RLConfig())

metrics = updater.update(prompts, completions, rewards)
```

## Dependencies

- `torch`
- `transformers`
- `peft`
- `trl`
- `datasets`
- `bitsandbytes` (optional, for 4-bit quantization)
