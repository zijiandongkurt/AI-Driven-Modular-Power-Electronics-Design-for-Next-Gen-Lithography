from typing import List, Dict


def build_policy_update_entries(
        constraint_id: str,
        prompt_text: str,
        topologies: List[Dict],
        relative_rewards: List[float],
) -> List[Dict]:
    update_entries = []

    for item, rel_reward in zip(topologies, relative_rewards):
        update_entries.append({
            "constraint_id": constraint_id,
            "prompt_text": prompt_text,
            "topology_path": item["topology_path"],
            "fitness_score": item["fitness_score"],
            "valid": item.get("valid", False),
            "relative_reward": rel_reward,
            "preference": "increase" if rel_reward > 0 else "decrease",
        })

    return update_entries