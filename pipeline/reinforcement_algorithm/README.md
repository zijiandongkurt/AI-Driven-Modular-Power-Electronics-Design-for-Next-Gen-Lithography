# Reinforcement Algorithm Module

## Current Version
This module implements a multi-constraint offline GRPO-style prototype.

It currently supports:
- multiple constraint sets per batch
- multiple topologies for each constraint set
- invalid topology penalty
- per-group reward normalization
- batch-level reward aggregation
- pseudo policy-gradient structure

## Input
`sample_batch.json`

Each constraint set contains:
- `constraint_id`
- `prompt_text`
- `topologies`
  - `topology_path`
  - `fitness_score`
  - `valid`

## Output

### `policy_update_batch.json`
Per-topology training signals:
- fitness_score
- advantage
- log_prob (pseudo)
- loss (pseudo)
- preference

### `group_summaries.json`
Per-constraint summaries:
- mean_fitness
- mean_advantage
- group_objective
- best_topology
- best_fitness

### `batch_summary.json`
Batch-level summaries:
- batch_mean_fitness
- batch_mean_advantage
- batch_mean_loss
- batch_objective
- best_topology
- best_fitness

## Current Logic
1. Load multiple constraint sets
2. Apply invalid penalty
3. Normalize rewards within each constraint group
4. Build policy update entries
5. Compute pseudo policy-gradient loss
6. Aggregate group-level and batch-level summaries

## Limitation
Current `log_prob` and `loss` are prototype placeholders and are not yet connected to a real LLM.

## Next Step
- Replace pseudo `log_prob` with real LLM log-probabilities
- Use LoRA-based fine-tuning
- Connect batch objective to actual backpropagation