from typing import List
import math


def normalize_relative_rewards(fitness_scores: List[float], epsilon: float = 1e-8) -> List[float]:
    if not fitness_scores:
        return []

    mean_val = sum(fitness_scores) / len(fitness_scores)
    variance = sum((x - mean_val) ** 2 for x in fitness_scores) / len(fitness_scores)
    std_val = math.sqrt(variance)

    return [(x - mean_val) / (std_val + epsilon) for x in fitness_scores]