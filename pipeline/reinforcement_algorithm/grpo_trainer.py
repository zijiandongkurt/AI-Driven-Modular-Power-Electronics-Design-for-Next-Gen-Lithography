import json
from pathlib import Path
from typing import List, Dict
from reward_normalizer import normalize_relative_rewards
from policy_update import build_policy_update_entries, update_policy
from rl_demo import update_model

class GRPOTrainer:

    def __init__(self, 
                 batch_size: int = 8, 
                 k_iterations: int = 50, 
                 kl_beta: float = 0.1, 
                 early_stop_fitness: float = 0.92, 
                 invalid_penalty: float = -1.0, 
                 epsilon: float = 1e-8):
        self.batch_size: int = batch_size
        self.k_iterations: int = k_iterations
        self.kl_beta: float = kl_beta
        self.early_stop_fitness: float = early_stop_fitness
        self.invalid_penalty: float = invalid_penalty
        self.epsilon: float = epsilon

    def load_sample_batch(self, batch_path: Path) -> Dict:
        # Load the raw sample_batch.json file.
        # Current format: {topology_id: {"fitness_score": value}}
        with batch_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def parse_simple_sample_batch(self, raw_batch: Dict) -> List[Dict]:
        # Convert input JSON from dict format to list format.
        # This makes later reward normalization easier.
        topologies = []

        for topology_id, data in raw_batch.items():
            topologies.append({
                "topology_id": topology_id,
                "fitness_score": float(data["fitness_score"]),
            })
        return topologies



#     def apply_invalid_penalty_to_topologies(self, topologies: List[Dict], invalid_penalty: float) -> List[Dict]:
#         processed = []
#
#         for item in topologies:
#             item_copy = dict(item)
#             if not item_copy.get("valid", False):
#                 item_copy["fitness_score"] = invalid_penalty
#             processed.append(item_copy)
# #
#         return processed  #Comment out for now


    def compute_group_summary(
            self,
            constraint_id: str,
            prompt_text: str,
            topologies: List[Dict],
            advantages: List[float],
    ) -> Dict:
        fitness_scores = [item["fitness_score"] for item in topologies]
        best_item = max(topologies, key=lambda x: x["fitness_score"])

        mean_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0
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
            self,
            all_update_entries: List[Dict],
            group_summaries: List[Dict],
    ) -> Dict:
        if not all_update_entries:
            return {
                "num_constraints": 0,
                "num_total_topologies": 0,
                "batch_mean_fitness": 0.0,
                "batch_mean_advantage": 0.0,
                "batch_objective": 0.0,
                "best_topology": None,
                "best_fitness": None,
            }

        all_fitness = [item["fitness_score"] for item in all_update_entries]
        all_advantages = [item["advantage"] for item in all_update_entries]

        best_item = max(all_update_entries, key=lambda x: x["fitness_score"])

        batch_mean_fitness = sum(all_fitness) / len(all_fitness)
        batch_mean_advantage = sum(all_advantages) / len(all_advantages)

        # Batch-level optimization signal
        batch_objective = best_item["fitness_score"] - batch_mean_fitness

        return {
            "num_constraints": len(group_summaries),
            "num_total_topologies": len(all_update_entries),
            "batch_mean_fitness": batch_mean_fitness,
            "batch_mean_advantage": batch_mean_advantage,
            "batch_objective": batch_objective,
            "best_topology": best_item["topology_id"],
            "best_fitness": best_item["fitness_score"],
        }


    # def process_single_constraint_group(
    #         self,
    #         constraint_group: Dict,
    #         invalid_penalty: float,
    #         epsilon: float,
    # ) -> Tuple[List[Dict], Dict]:
    #     constraint_id = constraint_group["constraint_id"]
    #     prompt_text = constraint_group["prompt_text"]
    #     topologies = constraint_group["topologies"]
    #
    #     processed_topologies = self.apply_invalid_penalty_to_topologies(topologies, invalid_penalty)
    #
    #     fitness_scores = [item["fitness_score"] for item in processed_topologies]
    #     advantages = normalize_relative_rewards(fitness_scores, epsilon)
    #
    #     update_entries = build_policy_update_entries(
    #         constraint_id=constraint_id,
    #         prompt_text=prompt_text,
    #         topologies=processed_topologies,
    #         relative_rewards=advantages,
    #     )
    #
    #     group_summary = self.compute_group_summary(
    #         constraint_id=constraint_id,
    #         prompt_text=prompt_text,
    #         topologies=processed_topologies,
    #         advantages=advantages,
    #     )
    #
    #     return update_entries, group_summary

    def train(self):
        batch_path = Path(__file__).parent / "sample_batch.json"

        # 1. Load raw input JSON.
        raw_batch = self.load_sample_batch(batch_path)

        # 2. Convert raw dict format to list of topology records.
        topologies = self.parse_simple_sample_batch(raw_batch)

        # 3. Extract raw fitness scores.
        fitness_scores = [item["fitness_score"] for item in topologies]

        # 4. Compute advantages inside RL module.
        advantages = normalize_relative_rewards(fitness_scores, self.epsilon)

        # 5. Build policy update entries.
        all_update_entries = build_policy_update_entries(
            topologies=topologies,
            relative_rewards=advantages,
        )
        # Try to connect to policy update stage.
        # Currently this is only a placeholder because model/tokenizer/optimizer are not available yet.
        update_result = update_policy(
            model=None,
            tokenizer=None,
            optimizer=None,
            update_entries=all_update_entries,
)

        update_result = update_model(
            model=None,
            tokenizer=None,
            optimizer=None,
            update_entries=all_update_entries,
        )


        # 6. No group summaries for now because there is no constraint_id.
        group_summaries: List[Dict] = []

        # 7. Compute batch-level summary.
        batch_summary = self.compute_batch_summary(
            all_update_entries=all_update_entries,
            group_summaries=group_summaries,
        )

        out_dir = Path(__file__).parent
        update_batch_path = out_dir / "policy_update_batch.json"
        batch_summary_path = out_dir / "batch_summary.json"
        update_status_path = out_dir / "policy_update_status.json"


        with update_batch_path.open("w", encoding="utf-8") as f:
            json.dump(all_update_entries, f, indent=2)

        with update_status_path.open("w", encoding="utf-8") as f:
            json.dump(update_result, f, indent=2)

        with batch_summary_path.open("w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2)

        print("=== Simple Offline GRPO Prototype ===")
        print(f"Total topologies: {batch_summary['num_total_topologies']}")
        print(f"Best topology: {batch_summary['best_topology']}")
        print(f"Best fitness: {batch_summary['best_fitness']:.4f}")
        print(f"Batch objective: {batch_summary['batch_objective']:.4f}")
        print(f"Batch mean advantage: {batch_summary['batch_mean_advantage']:.4f}")
        print(f"Saved update batch to: {update_batch_path}")
        print(f"Saved batch summary to: {batch_summary_path}")

        print(f"Policy updated: {update_result['updated']}")
        print(f"Update reason: {update_result['reason']}")
        print(f"Saved update status to: {update_status_path}")

        print(f"Model updated: {update_result['updated']}")
        print(f"Update reason: {update_result['reason']}")


if __name__ == "__main__":
    grpo = GRPOTrainer()
    grpo.train()