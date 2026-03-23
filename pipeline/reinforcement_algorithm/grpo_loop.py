import json
from pathlib import Path
from typing import List, Dict, Tuple

from grpo_config import GRPOConfig
from reward_normalizer import normalize_relative_rewards
from policy_update import build_policy_update_entries


def load_constraint_batch(batch_path: Path) -> List[Dict]:
    with batch_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_invalid_penalty_to_topologies(topologies: List[Dict], invalid_penalty: float) -> List[Dict]:
    processed = []

    for item in topologies:
        item_copy = dict(item)
        if not item_copy.get("valid", False):
            item_copy["fitness_score"] = invalid_penalty
        processed.append(item_copy)

    return processed


def compute_group_summary(
        constraint_id: str,
        prompt_text: str,
        topologies: List[Dict],
        relative_rewards: List[float],
) -> Dict:
    fitness_scores = [item["fitness_score"] for item in topologies]
    best_item = max(topologies, key=lambda x: x["fitness_score"])

    mean_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
    mean_advantage = sum(relative_rewards) / len(relative_rewards) if relative_rewards else 0.0
    group_objective = best_item["fitness_score"] - mean_fitness

    return {
        "constraint_id": constraint_id,
        "prompt_text": prompt_text,
        "num_topologies": len(topologies),
        "mean_fitness": mean_fitness,
        "mean_advantage": mean_advantage,
        "group_objective": group_objective,
        "best_topology": best_item["topology_path"],
        "best_fitness": best_item["fitness_score"],
    }

def compute_batch_summary(
        all_update_entries: List[Dict],
        group_summaries: List[Dict],
) -> Dict:
    if not all_update_entries:
        return {
            "num_constraints": 0,
            "num_total_topologies": 0,
            "batch_mean_fitness": 0.0,
            "batch_mean_advantage": 0.0,
            "batch_mean_loss": 0.0,
            "batch_objective": 0.0,
            "best_topology": None,
            "best_fitness": None,
        }

    all_fitness = [item["fitness_score"] for item in all_update_entries]
    all_advantages = [item["advantage"] for item in all_update_entries]
    all_losses = [item["loss"] for item in all_update_entries]

    best_item = max(all_update_entries, key=lambda x: x["fitness_score"])

    batch_mean_fitness = sum(all_fitness) / len(all_fitness)
    batch_mean_advantage = sum(all_advantages) / len(all_advantages)
    batch_mean_loss = sum(all_losses) / len(all_losses)

    # non-zero batch objective for optimization signal
    batch_objective = best_item["fitness_score"] - batch_mean_fitness

    return {
        "num_constraints": len(group_summaries),
        "num_total_topologies": len(all_update_entries),
        "batch_mean_fitness": batch_mean_fitness,
        "batch_mean_advantage": batch_mean_advantage,
        "batch_mean_loss": batch_mean_loss,
        "batch_objective": batch_objective,
        "best_topology": best_item["topology_path"],
        "best_fitness": best_item["fitness_score"],
    }


def process_single_constraint_group(
        constraint_group: Dict,
        invalid_penalty: float,
        epsilon: float,
) -> Tuple[List[Dict], Dict]:
    constraint_id = constraint_group["constraint_id"]
    prompt_text = constraint_group["prompt_text"]
    topologies = constraint_group["topologies"]

    processed_topologies = apply_invalid_penalty_to_topologies(topologies, invalid_penalty)

    fitness_scores = [item["fitness_score"] for item in processed_topologies]
    relative_rewards = normalize_relative_rewards(fitness_scores, epsilon)

    update_entries = build_policy_update_entries(
        constraint_id=constraint_id,
        prompt_text=prompt_text,
        topologies=processed_topologies,
        relative_rewards=relative_rewards,
    )

    group_summary = compute_group_summary(
        constraint_id=constraint_id,
        prompt_text=prompt_text,
        topologies=processed_topologies,
        relative_rewards=relative_rewards,
    )

    return update_entries, group_summary


def main():
    config = GRPOConfig()
    batch_path = Path(__file__).parent / "sample_batch.json"

    constraint_batch = load_constraint_batch(batch_path)

    all_update_entries: List[Dict] = []
    group_summaries: List[Dict] = []

    for constraint_group in constraint_batch:
        update_entries, group_summary = process_single_constraint_group(
            constraint_group=constraint_group,
            invalid_penalty=config.invalid_penalty,
            epsilon=config.epsilon,
        )

        all_update_entries.extend(update_entries)
        group_summaries.append(group_summary)

    batch_summary = compute_batch_summary(
        all_update_entries=all_update_entries,
        group_summaries=group_summaries,
    )

    out_dir = Path(__file__).parent
    update_batch_path = out_dir / "policy_update_batch.json"
    group_summary_path = out_dir / "group_summaries.json"
    batch_summary_path = out_dir / "batch_summary.json"

    with update_batch_path.open("w", encoding="utf-8") as f:
        json.dump(all_update_entries, f, indent=2)

    with group_summary_path.open("w", encoding="utf-8") as f:
        json.dump(group_summaries, f, indent=2)

    with batch_summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    print("=== Multi-Constraint Offline GRPO Prototype ===")
    print(f"Constraint sets: {batch_summary['num_constraints']}")
    print(f"Total topologies: {batch_summary['num_total_topologies']}")
    print(f"Best topology: {batch_summary['best_topology']}")
    print(f"Best fitness: {batch_summary['best_fitness']:.4f}")
    print(f"Batch objective: {batch_summary['batch_objective']:.4f}")
    print(f"Batch mean loss: {batch_summary['batch_mean_loss']:.4f}")
    print(f"Saved update batch to: {update_batch_path}")
    print(f"Saved group summaries to: {group_summary_path}")
    print(f"Saved batch summary to: {batch_summary_path}")


if __name__ == "__main__":
    main()