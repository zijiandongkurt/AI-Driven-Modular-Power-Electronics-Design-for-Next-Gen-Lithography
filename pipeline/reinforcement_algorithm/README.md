# Reinforcement Learning Module for Topology Optimization

This module implements a reinforcement learning (RL) loop that connects simulation-based evaluation with language model training.

The goal is to convert circuit simulation results (fitness scores) into training signals and use them to guide updates of a language model.

---

## Overview

The current implementation focuses on the RL loop between the reward function and the language model.

It follows a simplified policy-gradient approach:

1. Compute relative rewards (advantages) from fitness scores
2. Construct training samples (prompt + topology + advantage)
3. Compute log-probabilities using a language model
4. Build a policy-gradient-style loss
5. Perform backpropagation and update model parameters

---

## Project Structure
```
pipeline/
├── reinforcement_algorithm/
│   ├── grpo_loop.py
│   ├── reward_normalizer.py
│   ├── policy_update.py
│   ├── policy_update_batch.json
│   ├── group_summaries.json
│   ├── batch_summary.json
│   └── sample_batch.json
│
├── model_training/
│   ├── trainer_demo.py
│   ├── trainer_demo_output.json
│   ├── trainer_demo_summary.json
```
---

## Module Description

### 1. reinforcement_algorithm

Handles RL-related processing.

- Takes simulation outputs (fitness scores)
- Normalizes rewards within each constraint group
- Computes advantages
- Prepares training data for the model

**Key files:**

- `grpo_loop.py`  
  Main script for batch processing and advantage computation

- `reward_normalizer.py`  
  Computes normalized rewards (advantages)

- `policy_update.py`  
  Converts rewards into training-ready samples

- `policy_update_batch.json`  
  Output data used as input for training

---

### 2. model_training

Performs the actual model training step using RL signals.

- Loads training samples from RL module
- Computes log-probabilities using a language model
- Builds a policy-gradient-style loss
- Performs backpropagation and optimizer update

**Model used:**

- `Qwen2.5-0.5B-Instruct` (used for local testing due to hardware constraints)

**Key files:**

- `trainer_demo.py`  
  Main training script

- `trainer_demo_output.json`  
  Per-sample results

- `trainer_demo_summary.json`  
  Batch-level statistics

---

## LLM Component

The LLM component is implemented by Yuhao.

- Code location: `<LLM>` folder under branch `"LLM_Syh"`
- Detailed explanations are provided in the README.md in that branch

---

## How to Run

### Step 1: Run RL processing

```bash
python pipeline/reinforcement_algorithm/grpo_loop.py