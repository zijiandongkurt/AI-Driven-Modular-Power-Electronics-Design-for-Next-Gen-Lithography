import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.snellius.simulation_server import SimulationServer as LTSpiceSimulator  # CHANGED: use current main simulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
from pipeline.reinforcement_algorithm.new_rl_updater import RLConfig


def get_next_run_folder(data_dir: Path) -> str:
    """Find next Run_XXX folder."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    run_folders = [
        d.name for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"Run_\d+", d.name)
    ]

    if not run_folders:
        return "Run_001"

    run_numbers = [int(f.split("_")[1]) for f in run_folders]
    return f"Run_{max(run_numbers) + 1:03d}"


def load_reward_data(batch_id: str) -> Dict:
    """Load reward_results.json for a batch."""
    path = Path("pipeline") / "data" / batch_id / "reward_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing reward file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_group_id(netlist_id: str) -> Optional[str]:
    """Extract group id from names like xxx_g1_cand3_b2."""
    match = re.search(r"_(g\d+)_cand\d+", netlist_id)
    if match:
        return match.group(1)
    return None


def add_batch_to_history(
    history: List[Dict],
    batch_id: str,
    group_to_parent: Dict[str, Optional[Dict]],
    default_depth: int,
) -> None:
    """Add all evaluated netlists in current batch to history database."""
    reward_data = load_reward_data(batch_id)
    circuits = reward_data.get("circuits", {})

    for netlist_id, info in circuits.items():
        fitness = info.get("fitness_score", info.get("grpo_reward", None))
        if fitness is None:
            continue

        group_id = parse_group_id(netlist_id)
        parent = group_to_parent.get(group_id)

        if parent is None:
            depth = default_depth
            parent_id = None
        else:
            depth = int(parent["depth"]) + 1
            parent_id = parent["netlist_id"]

        history.append({
            "netlist_id": netlist_id,
            "batch_id": batch_id,
            "group_id": group_id,
            "parent_id": parent_id,
            "fitness": float(fitness),
            "depth": depth,
        })


def save_history(history: List[Dict], run_folder_path: Path) -> None:
    """Save historical state database."""
    path = run_folder_path / "history_db.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def softmax_sample_parents(
    history: List[Dict],
    k: int,
    alpha_depth: float,
    temperature: float,
    seed: int = 42,
) -> List[Dict]:
    """
    Select parents from all historical states using:
        score = fitness + alpha_depth * depth
    followed by temperature-scaled softmax sampling.
    """
    if len(history) < k:
        raise RuntimeError(f"Not enough history states to sample {k} parents.")

    rng = random.Random(seed)

    scores = [
        float(item["fitness"]) + alpha_depth * int(item["depth"])
        for item in history
    ]

    max_score = max(scores)
    exp_scores = [
        math.exp((s - max_score) / max(temperature, 1e-8))
        for s in scores
    ]

    selected = []
    available = list(range(len(history)))

    for _ in range(k):
        total = sum(exp_scores[i] for i in available)
        r = rng.random() * total

        acc = 0.0
        chosen_idx = available[-1]

        for idx in available:
            acc += exp_scores[idx]
            if acc >= r:
                chosen_idx = idx
                break

        selected.append(history[chosen_idx])
        available.remove(chosen_idx)

    return selected


def run_eval_pipeline(
    batch_id: str,
    val,
    simulator,
    reward_fn,
    constraint: Dict,
    weights: Dict,
) -> None:
    """Run validate -> simulate -> reward for one batch."""
    print("Validating netlists...")
    val.validate(batch_id)

    print("Running simulations...")
    simulator.simulate(batch_id)

    print("Evaluating fitness and formatting JSON...")
    reward_fn.process_batch(batch_id, constraint, weights=weights)


def main():
    # --- Tree-search GRPO configuration ---
    N_batch = 2
    SEED_PROMPTS = 2
    PARENTS_PER_BATCH = 2
    OUTPUTS_PER_PARENT = 4
    MAX_TOKENS = 1024

    # Selection hyperparameters.
    ALPHA_DEPTH = 0.05
    SOFTMAX_TEMPERATURE = 0.2
    RANDOM_SEED = 42

    weights = {
        "v_out": 10.0,
        "efficiency": 20.0,
        "volume": 2.0,
        "component_cost": 1.0,
        "components": {
            "mosfet": 1.0,
            "diode": 1.0,
            "inductor": 1.0,
            "capacitor": 1.0,
        },
    }

    llm = TopologyLLM(max_new_tokens=MAX_TOKENS)

    val = validator()
    simulator = LTSpiceSimulator()
    reward_fn = RewardFunctionNorm()
    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=2)

    grpo = GRPOTrainer(
        llm=llm,
        validator=val,
        simulator=simulator,
        reward_fn=reward_fn,
        constraint=constraint,
        rl_config=RLConfig(
            max_length=1024,
            max_prompt_length=256,
            max_completion_length=512,
            learning_rate=1e-5,
            save_every=1,
            lora_r=4,
            lora_alpha=8,
        ),
    )

    data_dir = Path("pipeline/data")
    run_folder_name = get_next_run_folder(data_dir)
    run_folder_path = data_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Tree-Search GRPO Run: {run_folder_name} ===")

    history: List[Dict] = []
    selected_parents: List[Dict] = []

    for batch_idx in range(1, N_batch + 1):
        current_batch_id = f"{run_folder_name}/batch_{batch_idx}"
        print(f"\n--- Processing {current_batch_id} ---")

        if batch_idx == 1:
            seed_parent_ids = [
                f"seed_prompt_{i}"
                for i in range(1, SEED_PROMPTS + 1)
            ]

            print("Generating seed groups...")
            written = llm.generate_grouped_for_batch(
                constraint=constraint,
                batchID=current_batch_id,
                parent_ids=seed_parent_ids,
                previous_batch_id=None,
                outputs_per_parent=OUTPUTS_PER_PARENT,
                DEMO=False,
            )

            group_to_parent = {
                f"g{i}": None
                for i in range(1, SEED_PROMPTS + 1)
            }

        else:
            print("Generating children from selected parents...")

            if hasattr(llm, "generate_grouped_for_parent_entries"):
                written = llm.generate_grouped_for_parent_entries(
                    constraint=constraint,
                    batchID=current_batch_id,
                    parent_entries=selected_parents,
                    outputs_per_parent=OUTPUTS_PER_PARENT,
                    DEMO=True,
                )
            else:
                parent_batches = {p["batch_id"] for p in selected_parents}
                if len(parent_batches) != 1:
                    raise RuntimeError(
                        "Selected parents come from different batches. "
                        "Please add llm.generate_grouped_for_parent_entries()."
                    )

                previous_batch_id = selected_parents[0]["batch_id"]

                written = llm.generate_grouped_for_batch(
                    constraint=constraint,
                    batchID=current_batch_id,
                    parent_ids=[p["netlist_id"] for p in selected_parents],
                    previous_batch_id=previous_batch_id,
                    outputs_per_parent=OUTPUTS_PER_PARENT,
                    DEMO=True,
                )

            group_to_parent = {
                f"g{i}": parent
                for i, parent in enumerate(selected_parents, start=1)
            }

        print(f"Generated {len(written) if written else 0} netlists.")

        run_eval_pipeline(
            batch_id=current_batch_id,
            val=val,
            simulator=simulator,
            reward_fn=reward_fn,
            constraint=constraint,
            weights=weights,
        )

        print("Running GRPO RL update...")
        grpo.update_from_batch(
            batch_id=current_batch_id,
            max_samples=None,
        )

        add_batch_to_history(
            history=history,
            batch_id=current_batch_id,
            group_to_parent=group_to_parent,
            default_depth=1,
        )

        save_history(history, run_folder_path)

        print(f"History size: {len(history)} evaluated netlists")

        # CHANGED: only select next parents if another batch remains.
        if batch_idx < N_batch:
            selected_parents = softmax_sample_parents(
                history=history,
                k=PARENTS_PER_BATCH,
                alpha_depth=ALPHA_DEPTH,
                temperature=SOFTMAX_TEMPERATURE,
                seed=RANDOM_SEED + batch_idx,
            )

            print("Selected parents for next batch:")
            for p in selected_parents:
                score = p["fitness"] + ALPHA_DEPTH * p["depth"]
                print(
                    f"  {p['netlist_id']} | "
                    f"fitness={p['fitness']:.4f}, "
                    f"depth={p['depth']}, "
                    f"score={score:.4f}, "
                    f"batch={p['batch_id']}"
                )

        print(f"--- Finished {current_batch_id} ---")

    print(f"\n=== Tree-Search GRPO Run {run_folder_name} Complete ===")


if __name__ == "__main__":
    main()