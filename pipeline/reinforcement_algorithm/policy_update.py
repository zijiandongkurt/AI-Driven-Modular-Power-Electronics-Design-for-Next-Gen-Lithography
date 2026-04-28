from typing import List, Dict, Optional


def build_policy_update_entries(
        topologies: List[Dict],
        relative_rewards: List[float],
) -> List[Dict]:
    """
    Build training entries for RL policy update.

    Current stage:
    - Input only contains topology_id and fitness_score
    - relative_rewards = normalized rewards (advantage)

    Output:
    - A list of entries that represent training signals
    - These entries will later be used for real RL updates

    NOTE:
    This function does NOT update any model.
    It only prepares structured data for future training.
    """

    update_entries = []

    for item, rel_reward in zip(topologies, relative_rewards):
        update_entries.append({

            # Unique identifier of the generated topology
            "topology_id": item["topology_id"],

            # Raw score from simulator / evaluator
            "fitness_score": item["fitness_score"],

            # Advantage computed by RL (normalized reward)
            # Positive → better than average
            # Negative → worse than average
            "advantage": rel_reward,

            # Direction of update signal
            # increase → model should generate similar outputs more often
            # decrease → model should avoid similar outputs
            "preference": "increase" if rel_reward >= 0 else "decrease",

            # ===== Reserved fields for future RL integration =====

            # Input prompt to LLM (not available yet)
            "prompt_text": item.get("prompt_text"),

            # Generated topology content (not available yet)
            "topology_text": item.get("topology_text"),

            # Log probability of the generated output under current model
            # Required for real RL update (log_prob × advantage)
            "log_prob": item.get("log_prob"),
        })

    return update_entries


def update_policy(
        model: Optional[object],
        tokenizer: Optional[object],
        optimizer: Optional[object],
        update_entries: List[Dict],
) -> Dict:
    """
    Placeholder for real RL policy update.

    What a real RL update requires:
    - model (LLM)
    - tokenizer
    - optimizer
    - prompt_text
    - topology_text
    - log_prob (or enough info to compute it)

    Current limitation:
    - We only have fitness_score and advantage
    - Therefore, no real parameter update can be performed

    This function:
    - Checks missing components
    - Returns a structured status report
    """

    missing_fields = []

    for entry in update_entries:
        if entry.get("prompt_text") is None:
            missing_fields.append("prompt_text")
        if entry.get("topology_text") is None:
            missing_fields.append("topology_text")
        if entry.get("log_prob") is None:
            missing_fields.append("log_prob")

    # Case 1: No model components provided
    if model is None or tokenizer is None or optimizer is None:
        return {
            "updated": False,
            "reason": "Missing model/tokenizer/optimizer. Only data pipeline is connected.",
            "num_entries": len(update_entries),
            "missing_fields": sorted(set(missing_fields)),
        }

    # Case 2: Missing required training fields
    if missing_fields:
        return {
            "updated": False,
            "reason": "Missing required RL fields (prompt_text, topology_text, log_prob).",
            "num_entries": len(update_entries),
            "missing_fields": sorted(set(missing_fields)),
        }

    # ===== Future real RL update (NOT IMPLEMENTED YET) =====
    #
    # Example:
    # for entry in update_entries:
    #     log_prob = compute_log_prob(model, tokenizer, entry)
    #     loss = -log_prob * entry["advantage"]
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    return {
        "updated": False,
        "reason": "RL update logic not implemented yet.",
        "num_entries": len(update_entries),
    }