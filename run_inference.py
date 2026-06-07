import os
import re
import json
import argparse
from pathlib import Path
import time
import random
import numpy as np

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.snellius.simulation_server import SimulationServer as LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint

from pipeline.utility.netlist_database import NetlistDatabase
from pipeline.utility.summary_logger import SummaryLogger

try:
    from pipeline.graphs_and_visualizations.Visualize_demo_results import plot_run_results
    from pipeline.graphs_and_visualizations.plot_probabilities import plot_softmax_probabilities
    from pipeline.graphs_and_visualizations.plot_cumulative_probabilities import plot_cumulative_probabilities
except ImportError as e:
    print(f"Warning: Plotting modules not found. Plotting disabled. Error: {e}")
    plot_run_results = None
    plot_softmax_probabilities = None
    plot_cumulative_probabilities = None


def get_next_run_folder(data_dir: Path, prefix="Run") -> str:
    """Generate the next sequential run folder name with an optional prefix."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{prefix}_(\d+)$")
    run_folders = []

    for d in data_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                run_folders.append(int(match.group(1)))

    if not run_folders:
        return f"{prefix}_001"

    return f"{prefix}_{max(run_folders) + 1:03d}"


def run_inference(config: dict) -> str:
    """Run the Epsilon-Greedy Evolutionary Search loop.

    Args:
        config (dict): Hyperparameter dictionary with keys ``run_settings``,
            ``mcts_settings``, ``constraint_settings``, and ``weights``.

    Returns:
        str: Absolute path to the results folder for this run.
    """
    run_settings        = config.get("run_settings", {})
    mcts_settings       = config.get("mcts_settings", {})
    constraint_settings = config.get("constraint_settings", {})
    weights             = config.get("weights", {})

    SEED = run_settings.get("seed", 42)
    random.seed(SEED)
    np.random.seed(SEED)

    N_BATCHES                = run_settings.get("n_batches", 20)
    SAMPLED_STATES_PER_BATCH = run_settings.get("sampled_states_per_batch", 2)
    CANDIDATES_PER_PROMPT    = run_settings.get("candidates_per_prompt", 4)
    MAX_TOKENS               = run_settings.get("max_tokens", 2048)
    UPDATE_PLOTS             = run_settings.get("update_plots_per_batch", True)
    MODEL_ID                 = run_settings.get("model_id", "Qwen/Qwen3-14B")
    RUN_PREFIX               = run_settings.get("run_prefix", "Run")
    LORA_PATH                = run_settings.get("lora_path", None)

    TEMP    = mcts_settings.get("temperature", 0.05)
    TOP_K   = mcts_settings.get("top_k", 15)
    EPSILON = mcts_settings.get("epsilon", 0.15)

    CONSTRAINT_PATH  = constraint_settings.get("dataset_path", "pipeline/data/datasets/constraints.json")
    CONSTRAINT_INDEX = constraint_settings.get("index", 4)
    CONSTRAINT_PHASE = constraint_settings.get("phase", "Phase1")
    custom_label     = f"{CONSTRAINT_PHASE}_cons{CONSTRAINT_INDEX}"

    print(f"Loading Model: {MODEL_ID} ...")
    if LORA_PATH:
        print(f"Loading LoRA adapter from: {LORA_PATH}")

    llm = TopologyLLM(
        model_id=MODEL_ID,
        max_new_tokens=MAX_TOKENS,
        lora_path=LORA_PATH,
    )
    val        = validator()
    simulator  = LTSpiceSimulator()
    reward_fn  = RewardFunctionNorm()
    constraint = load_constraint(CONSTRAINT_PATH, idx=CONSTRAINT_INDEX)
    db         = NetlistDatabase(temperature=TEMP)

    data_dir        = Path("pipeline/data")
    run_folder_name = get_next_run_folder(data_dir, prefix=RUN_PREFIX)
    run_folder_path = data_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    database_dir = run_folder_path / "database"
    database_dir.mkdir(parents=True, exist_ok=True)

    sim_params = {
        "sampled_states":  SAMPLED_STATES_PER_BATCH,
        "cands_per_prompt": CANDIDATES_PER_PROMPT,
        "mcts_temp":       TEMP,
        "top_k":           TOP_K,
        "epsilon":         EPSILON,
        "max_tokens":      MAX_TOKENS,
        "model_id":        MODEL_ID,
        "lora_path":       LORA_PATH,
    }

    logger = SummaryLogger(
        run_folder_path=run_folder_path,
        n_batches=N_BATCHES,
        sim_params=sim_params,
        weights=weights,
    )

    print(f"\n{'='*50}")
    print(f"🚀 Starting Iterative Inference Loop: {run_folder_name}")
    print(f"🧠 Model: {MODEL_ID}")
    print(f"🎯 Constraint: {custom_label}")
    if LORA_PATH:
        print(f"🔧 LoRA: {LORA_PATH}")
    print(f"{'='*50}\n")

    run_start_time = time.time()

    profiling = {
        "llm_generation":     0.0,
        "validation":         0.0,
        "simulation":         0.0,
        "reward_calculation": 0.0,
        "database_io":        0.0,
        "logging_and_plotting": 0.0,
    }

    for i in range(1, N_BATCHES + 1):
        batch_start_time = time.time()
        current_batch_id = f"{run_folder_name}/batch_{i}"
        print(f"\n--- Processing {current_batch_id} ---")

        t_start        = time.time()
        sampled_states = db.sample_states(n=SAMPLED_STATES_PER_BATCH, top_k=TOP_K, epsilon=EPSILON)

        if not sampled_states:
            print("Database empty. Generating seed netlists from scratch.")
            written = llm.generate_for_batch(
                constraint,
                batchID=current_batch_id,
                n=SAMPLED_STATES_PER_BATCH * CANDIDATES_PER_PROMPT,
                DEMO=False,
            )
        else:
            print(f"Sampled {len(sampled_states)} states from database. Branching {CANDIDATES_PER_PROMPT} candidates each.")
            parent_entries = [
                {"netlist_id": s["id"], "batch_id": s["batch_id"]}
                for s in sampled_states
            ]
            written = llm.generate_grouped_for_parent_entries(
                constraint=constraint,
                batchID=current_batch_id,
                parent_entries=parent_entries,
                outputs_per_parent=CANDIDATES_PER_PROMPT,
                DEMO=True,
            )
        t_gen = time.time()
        profiling["llm_generation"] += (t_gen - t_start)

        print("Validating netlists...")
        val.validate(current_batch_id)
        t_val = time.time()
        profiling["validation"] += (t_val - t_gen)

        print("Running simulations...")
        simulator.simulate(current_batch_id)
        t_sim = time.time()
        profiling["simulation"] += (t_sim - t_val)

        print("Evaluating fitness...")
        reward_fn.process_batch(current_batch_id, constraint, weights=weights)
        t_rew = time.time()
        profiling["reward_calculation"] += (t_rew - t_sim)

        print("Updating State Database...")
        db.ingest_batch_data(current_batch_id, database_dir=database_dir)
        t_db = time.time()
        profiling["database_io"] += (t_db - t_rew)

        best_cand_id = max(db.records, key=lambda k: db.records[k]["fitness"]) if db.records else None

        if best_cand_id:
            logger.log_batch(
                batch_idx=i,
                batch_duration=(time.time() - batch_start_time),
                db=db,
                best_cand_id=best_cand_id,
            )
            print(f"--- Batch {i} Complete in {(time.time() - batch_start_time):.2f} sec ---")
            print(f"--- Current Global Best: {db.records[best_cand_id]['fitness']:.4f} ({best_cand_id}) ---")

        if UPDATE_PLOTS and plot_run_results:
            plot_run_results(str(run_folder_path))
            if plot_cumulative_probabilities:
                plot_cumulative_probabilities(run_id=run_folder_name, target_batch=i+1, temperature=TEMP, top_k=TOP_K, epsilon=EPSILON)
            if plot_softmax_probabilities:
                plot_softmax_probabilities(run_id=run_folder_name, target_batch=i+1, temperature=TEMP, top_k=TOP_K, epsilon=EPSILON)
        t_log = time.time()
        profiling["logging_and_plotting"] += (t_log - t_db)

    if not UPDATE_PLOTS and plot_run_results:
        t_final_plot = time.time()
        print("\n📊 Generating final run plots...")
        plot_run_results(str(run_folder_path))
        profiling["logging_and_plotting"] += (time.time() - t_final_plot)

    run_duration = time.time() - run_start_time

    print(f"\n{'='*50}")
    print(f"⏱️  PIPELINE PROFILING REPORT")
    print(f"{'='*50}")
    for step, duration in profiling.items():
        percentage = (duration / run_duration) * 100
        print(f"{step:<22}: {duration:>7.2f} sec  ({percentage:>5.1f}%)")
    print(f"{'-'*50}")
    print(f"{'TOTAL RUN TIME':<22}: {run_duration:>7.2f} sec")
    print(f"{'='*50}")

    prof_file = run_folder_path / "timing_profile.json"
    with open(prof_file, "w", encoding="utf-8") as f:
        json.dump(profiling, f, indent=4)

    if db.records:
        best_cand_id  = max(db.records, key=lambda k: db.records[k]["fitness"])
        champ_metrics = db.records[best_cand_id].get("metrics", {}).get("raw_metrics", {})

        export_data = {
            "id":             best_cand_id,
            "target_voltage": constraint.get("vout_target", 0),
            "target_power":   constraint.get("power_in", 10),
            "raw_voltage":    champ_metrics.get("simulation_output_voltage", 0.0),
            "raw_efficiency": champ_metrics.get("efficiency", 0.0),
            "raw_volume":     champ_metrics.get("total_volume_cm3", 0.0),
            "raw_components": champ_metrics.get("total_components", 0),
            "lora_path":      LORA_PATH,
        }

        champ_file = run_folder_path / "champion_metrics.json"
        with open(champ_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)

    return str(run_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test_config.json", help="Path to test config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    assert config_path.exists(), f"Config file not found: {config_path.resolve()}"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    run_inference(config)