from dataclasses import dataclass


@dataclass
class GRPOConfig:
    batch_size: int = 8
    k_iterations: int = 50
    kl_beta: float = 0.1
    early_stop_fitness: float = 0.92
    invalid_penalty: float = -1.0
    epsilon: float = 1e-8
