# Reinforcement Algorithm Module

## Current Version
This module implements a **multi-constraint offline GRPO prototype**.

Instead of processing only one fixed constraint set, it now supports:
- multiple constraint sets per batch
- multiple topologies for each constraint set
- per-group reward normalization
- batch-level reward aggregation

## Input
`sample_batch.json`

Structure:
- `constraint_id`
- `prompt_text`
- `topologies`
    - `topology_path`
    - `fitness_score`
    - `valid`

## Output
The module generates three output files:

### 1. `policy_update_batch.json`
Per-topology update signals:
- constraint_id
- prompt_text
- topology_path
- fitness_score
- valid
- relative_reward
- preference

### 2. `group_summaries.json`
Per-constraint statistics:
- mean fitness
- mean relative reward
- best topology
- best fitness

### 3. `batch_summary.json`
Batch-level statistics:
- number of constraint sets
- total number of topologies
- batch mean fitness
- batch mean relative reward
- batch objective
- global best topology

## Current Logic
1. Load multiple constraint sets
2. Apply invalid penalty to invalid topologies
3. Normalize rewards within each constraint group
4. Build per-topology policy update signals
5. Aggregate group-level and batch-level summaries

## Next Step
Use the aggregated batch objective as the training signal for:
- LoRA-based fine-tuning
- future GRPO / RL backpropagation