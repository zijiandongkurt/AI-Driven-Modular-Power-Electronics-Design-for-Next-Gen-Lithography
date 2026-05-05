import json
from pathlib import Path
from typing import List, Dict

from reward_normalizer import normalize_relative_rewards
from policy_update import build_policy_update_entries, update_policy


class GRPOTrainer:
    """
    Offline GRPO prototype.

    Current purpose:
    - Read sample_batch.json
    - Normalize fitness scores into advantages
    - Build policy update entries
    - Save summary JSON files

    NOTE:
    This file does NOT update the LLM yet.
    Real model update is handled by:
        pipeline/llm_topology_generation/rl_updater.py
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def load_sample_batch(self, batch_path: Path) -> Dict:
        """Load sample_batch.json."""
        with batch_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def parse_simple_sample_batch(self, raw_batch: Dict) -> List[Dict]:
        """
        Convert raw sample batch format:

            {
              "topology_1": {"fitness_score": 0.8},
              "topology_2": {"fitness_score": 0.5}
            }

        into:

            [
              {"topology_id": "topology_1", "fitness_score": 0.8},
              {"topology_id": "topology_2", "fitness_score": 0.5}
            ]
        """
        topologies = []

        for topology_id, data in raw_batch.items():
            topologies.append({
                "topology_id": topology_id,
                "fitness_score": float(data["fitness_score"]),
            })

        return topologies

    def compute_batch_summary(self, update_entries: List[Dict]) -> Dict:
        """Compute simple batch-level summary."""
        if not update_entries:
            return {
                "num_total_topologies": 0,
                "batch_mean_fitness": 0.0,
                "batch_mean_advantage": 0.0,
                "batch_objective": 0.0,
                "best_topology": None,
                "best_fitness": None,
            }

        fitness_scores = [item["fitness_score"] for item in update_entries]
        advantages = [item["advantage"] for item in update_entries]

        best_item = max(update_entries, key=lambda x: x["fitness_score"])

        batch_mean_fitness = sum(fitness_scores) / len(fitness_scores)
        batch_mean_advantage = sum(advantages) / len(advantages)
        batch_objective = best_item["fitness_score"] - batch_mean_fitness

        return {
            "num_total_topologies": len(update_entries),
            "batch_mean_fitness": batch_mean_fitness,
            "batch_mean_advantage": batch_mean_advantage,
            "batch_objective": batch_objective,
            "best_topology": best_item["topology_id"],
            "best_fitness": best_item["fitness_score"],
        }

    def train(self):
        batch_path = Path(__file__).parent / "sample_batch.json"

        raw_batch = self.load_sample_batch(batch_path)
        topologies = self.parse_simple_sample_batch(raw_batch)

        fitness_scores = [item["fitness_score"] for item in topologies]
        advantages = normalize_relative_rewards(fitness_scores, self.epsilon)

        update_entries = build_policy_update_entries(
            topologies=topologies,
            relative_rewards=advantages,
        )

        update_result = update_policy(
            model=None,
            tokenizer=None,
            optimizer=None,
            update_entries=update_entries,
        )

        batch_summary = self.compute_batch_summary(update_entries)

        out_dir = Path(__file__).parent
        update_batch_path = out_dir / "policy_update_batch.json"
        batch_summary_path = out_dir / "batch_summary.json"
        update_status_path = out_dir / "policy_update_status.json"

        with update_batch_path.open("w", encoding="utf-8") as f:
            json.dump(update_entries, f, indent=2)

        with batch_summary_path.open("w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2)

        with update_status_path.open("w", encoding="utf-8") as f:
            json.dump(update_result, f, indent=2)

        print("=== Offline GRPO Prototype ===")
        print(f"Total topologies: {batch_summary['num_total_topologies']}")
        print(f"Best topology: {batch_summary['best_topology']}")
        print(f"Best fitness: {batch_summary['best_fitness']:.4f}")
        print(f"Batch objective: {batch_summary['batch_objective']:.4f}")
        print(f"Batch mean advantage: {batch_summary['batch_mean_advantage']:.4f}")
        print(f"Policy updated: {update_result['updated']}")
        print(f"Update reason: {update_result['reason']}")
        print(f"Saved update batch to: {update_batch_path}")
        print(f"Saved batch summary to: {batch_summary_path}")
        print(f"Saved update status to: {update_status_path}")


if __name__ == "__main__":
    grpo = GRPOTrainer()
    grpo.train()