import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.snellius.simulation_server import SimulationServer as LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
from pipeline.reinforcement_algorithm.new_rl_updater import RLConfig
from pipeline.graphs_and_visualizations.Visualize_demo_results import plot_run_results
from pipeline.graphs_and_visualizations.plot_probabilities import plot_softmax_probabilities
from pipeline.graphs_and_visualizations.plot_cumulative_probabilities import plot_cumulative_probabilities
from pipeline.utility.summary_logger import SummaryLogger


def get_next_zycos_folder(data_dir: Path) -> str:
    """Find next zycos_XXX folder."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    folders = [
        d.name for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"zycos_\d+", d.name)
    ]

    if not folders:
        return "zycos_001"

    numbers = [int(f.split("_")[1]) for f in folders]
    return f"zycos_{max(numbers) + 1:03d}"


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


def _softmax_sample_from_pool(
    pool: List[Dict],
    temperature: float,
    rng: random.Random,
) -> Dict:
    """Sample one item from a pool using temperature-scaled softmax over fitness."""
    scores = [float(item["fitness"]) for item in pool]
    max_score = max(scores)

    exp_scores = [
        math.exp((score - max_score) / max(temperature, 1e-8))
        for score in scores
    ]

    total = sum(exp_scores)
    r = rng.random() * total

    acc = 0.0
    for item, weight in zip(pool, exp_scores):
        acc += weight
        if acc >= r:
            return item

    return pool[-1]


def epsilon_greedy_topk_sample_parents(
    history: List[Dict],
    k: int,
    top_k: int,
    epsilon: float,
    temperature: float,
    seed: int = 42,
) -> List[Dict]:
    """Select parents using epsilon-greedy Top-K softmax.

    With probability ``1 - epsilon`` (exploitation), samples from the Top-K
    candidates using temperature-scaled softmax over fitness. With probability
    ``epsilon`` (exploration), randomly samples from the long-tail candidates
    outside Top-K. Depth is used only for lineage tracking, not for selection.

    Args:
        history (List[Dict]): List of evaluated netlist records with fitness values.
        k (int): Number of parents to select.
        top_k (int): Size of the elite pool for softmax sampling.
        epsilon (float): Probability of selecting from the long-tail (exploration).
        temperature (float): Softmax temperature scaling factor.
        seed (int): Random seed for reproducibility.

    Returns:
        List[Dict]: Selected parent records, each annotated with a selection_mode key.

    Raises:
        RuntimeError: If history contains fewer than k entries, or if fewer than k
            parents could be selected.
    """
    if len(history) < k:
        raise RuntimeError(f"Not enough history states to sample {k} parents.")

    rng = random.Random(seed)

    sorted_history = sorted(
        history,
        key=lambda item: float(item["fitness"]),
        reverse=True,
    )
    # top-k netlists
    top_k_pool = sorted_history[:max(1, min(top_k, len(sorted_history)))]
    long_tail_pool = sorted_history[len(top_k_pool):]

    selected: List[Dict] = []
    selected_ids = set()

    for _ in range(k):
        available_top = [
            item for item in top_k_pool
            if item["netlist_id"] not in selected_ids
        ]

        available_tail = [
            item for item in long_tail_pool
            if item["netlist_id"] not in selected_ids
        ]
        # Epsilon-greedy 
        use_exploration = (
            bool(available_tail)
            and rng.random() < epsilon
        )

        if use_exploration:
            chosen = rng.choice(available_tail)
            selection_mode = "epsilon_random_long_tail"
        else:
            #exploitation / top-K softmax 
            candidate_pool = available_top if available_top else available_tail
            if not candidate_pool:
                break

            chosen = _softmax_sample_from_pool(
                pool=candidate_pool,
                temperature=temperature,
                rng=rng,
            )
            selection_mode = "topk_softmax"

        chosen = dict(chosen)
        chosen["selection_mode"] = selection_mode

        selected.append(chosen)
        selected_ids.add(chosen["netlist_id"])

    if len(selected) < k:
        raise RuntimeError(
            f"Only selected {len(selected)} parents, but required {k}."
        )

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


def run_single(
    run_idx: int,
    zycos_name: str,
    llm,
    val,
    simulator,
    reward_fn,
    grpo,
    data_dir: Path,
    config: Dict,
) -> None:
    """Run the full training loop for a single constraint."""

    N_batch = config["n_batch"]
    SEED_PROMPTS = config["seed_prompts"]
    PARENTS_PER_BATCH = config["parents_per_batch"]
    OUTPUTS_PER_PARENT = config["outputs_per_parent"]
    SOFTMAX_TEMPERATURE = config["softmax_temperature"]
    RANDOM_SEED = config["random_seed"]
    TOP_K = config.get("top_k", 8)
    EPSILON = config.get("epsilon", 0.15)
    weights = config["weights"]

    constraint = load_constraint(config["constraint_path"], idx=run_idx)

    run_folder_name = f"Run_{run_idx + 1:03d}"
    run_folder_path = data_dir / zycos_name / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"=== {zycos_name} | Run {run_idx + 1}/{config['n_runs']} | constraint idx={run_idx} ===")
    print(f"{'=' * 60}")

    logger = SummaryLogger(
        run_folder_path=run_folder_path,
        n_batches=N_batch,
        sim_params={},
        weights=weights,
        constraint_idx=run_idx,
    )

    history: List[Dict] = []
    selected_parents: List[Dict] = []

    for batch_idx in range(1, N_batch + 1):
        batch_start = time.time()
        current_batch_id = f"{zycos_name}/{run_folder_name}/batch_{batch_idx}"
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

        logger.log_batch_training(
            batch_idx=batch_idx,
            batch_duration=time.time() - batch_start,
            history=history,
            batch_id=current_batch_id,
        )

        run_id = f"{zycos_name}/{run_folder_name}"
        plot_softmax_probabilities(
            run_id=run_id,
            target_batch=batch_idx + 1,
            temperature=SOFTMAX_TEMPERATURE,
            top_k=TOP_K,
            epsilon=EPSILON,
        )
        plot_cumulative_probabilities(
            run_id=run_id,
            target_batch=batch_idx + 1,
            temperature=SOFTMAX_TEMPERATURE,
            top_k=TOP_K,
            epsilon=EPSILON,
        )

        print(f"History size: {len(history)} evaluated netlists")

        if batch_idx < N_batch:
            selected_parents = epsilon_greedy_topk_sample_parents(
                history=history,
                k=PARENTS_PER_BATCH,
                top_k=TOP_K,
                epsilon=EPSILON,
                temperature=SOFTMAX_TEMPERATURE,
                seed=RANDOM_SEED + batch_idx,
            )

            print("Selected parents for next batch:")
            for p in selected_parents:
                score = p["fitness"]
                mode = p.get("selection_mode", "unknown")
                print(
                    f"  {p['netlist_id']} | "
                    f"fitness={p['fitness']:.4f}, "
                    f"depth={p['depth']}, "
                    f"score={score:.4f}, "
                    f"mode={mode}, "
                    f"batch={p['batch_id']}"
                )

        print(f"--- Finished {current_batch_id} ---")

    print(f"=== Finished {zycos_name}/{run_folder_name} ===")

    print(f"[training_loop] Generating result plots for {run_folder_name}...")
    plot_run_results(str(run_folder_path))


def main():
    config_path = Path("configs/training_config.json")
    assert config_path.exists(), f"training_config.json not found at {config_path.resolve()}"

    with config_path.open("r") as f:
        config = json.load(f)

    N_RUNS = config["n_runs"]

    data_dir = Path("pipeline/data")
    zycos_name = get_next_zycos_folder(data_dir)
    zycos_path = data_dir / zycos_name
    zycos_path.mkdir(parents=True, exist_ok=True)

    sft_lora_path = config.get("sft_lora_path", None)
    llm = TopologyLLM(
        max_new_tokens=config["max_tokens"],
        lora_path=sft_lora_path,
    )

    if sft_lora_path:
        print(f"Loaded SFT LoRA adapter from: {sft_lora_path}")

    val = validator()
    simulator = LTSpiceSimulator()
    reward_fn = RewardFunctionNorm()

    rl = config["rl_config"]
    grpo = GRPOTrainer(
        llm=llm,
        validator=val,
        simulator=simulator,
        reward_fn=reward_fn,
        constraint=None,
        output_dir=f"checkpoints/{zycos_name}/grpo-lora/final",
        rl_config=RLConfig(
            max_length=rl["max_length"],
            max_prompt_length=rl["max_prompt_length"],
            max_completion_length=rl["max_completion_length"],
            learning_rate=rl["learning_rate"],
            save_every=rl["save_every"],
            lora_r=rl["lora_r"],
            lora_alpha=rl["lora_alpha"],
        ),
    )

    print(f"=== Starting {zycos_name} | {N_RUNS} runs (constraint idx 0-{N_RUNS - 1}) ===")

    for run_idx in range(N_RUNS):
        run_single(
            run_idx=run_idx,
            zycos_name=zycos_name,
            llm=llm,
            val=val,
            simulator=simulator,
            reward_fn=reward_fn,
            grpo=grpo,
            data_dir=data_dir,
            config=config,
        )

    print(f"\n=== {zycos_name} complete — {N_RUNS} runs finished ===")


if __name__ == "__main__":
    main()