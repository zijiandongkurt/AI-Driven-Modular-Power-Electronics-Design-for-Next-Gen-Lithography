import os
import re
import json
import argparse
from pathlib import Path
import time

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint

# Import the newly separated classes
from netlist_database import NetlistDatabase
from summary_logger import SummaryLogger

try:
    from pipeline.graphs_and_visualizations.visualize_demo_results import plot_run_results
    from pipeline.graphs_and_visualizations.plot_probabilities import plot_softmax_probabilities
    from pipeline.graphs_and_visualizations.plot_cumulative_probabilities import plot_cumulative_probabilities
except ImportError as e:
    print(f"Warning: Plotting modules not found. Plotting disabled. Error: {e}")
    plot_run_results = None
    plot_softmax_probabilities = None
    plot_cumulative_probabilities = None


def get_next_run_folder(data_dir: Path, prefix="Run") -> str:
    """Generates the next sequential run folder name with an optional prefix."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        
    # Look for folders matching the prefix followed by an underscore and numbers
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
    """
    Main orchestrator method. Takes a dictionary of hyperparameters and 
    executes the Epsilon-Greedy Evolutionary Search loop.
    Returns the path to the results folder.
    """
    # Unpack config sections
    run_settings = config.get("run_settings", {})
    mcts_settings = config.get("mcts_settings", {})
    constraint_settings = config.get("constraint_settings", {})
    weights = config.get("weights", {})

    # Run Variables
    N_BATCHES = run_settings.get("n_batches", 20)
    SAMPLED_STATES_PER_BATCH = run_settings.get("sampled_states_per_batch", 2)
    CANDIDATES_PER_PROMPT = run_settings.get("candidates_per_prompt", 4)
    MAX_TOKENS = run_settings.get("max_tokens", 2048)
    UPDATE_PLOTS = run_settings.get("update_plots_per_batch", True)
    MODEL_ID = run_settings.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
    RUN_PREFIX = run_settings.get("run_prefix", "Run")

    # MCTS / Epsilon-Greedy Variables
    TEMP = mcts_settings.get("temperature", 0.05)
    TOP_K = mcts_settings.get("top_k", 15)
    EPSILON = mcts_settings.get("epsilon", 0.15)

    # Constraint Variables
    CONSTRAINT_PATH = constraint_settings.get("dataset_path", "pipeline/data/datasets/constraints.json")
    CONSTRAINT_INDEX = constraint_settings.get("index", 4)
    CONSTRAINT_PHASE = constraint_settings.get("phase", "Phase1")
    custom_label = f"{CONSTRAINT_PHASE}_cons{CONSTRAINT_INDEX}"

    # --- Setup Pipeline Components ---
    print(f"Loading Model: {MODEL_ID} ...")
    llm = TopologyLLM(
        model_id=MODEL_ID, 
        max_new_tokens=MAX_TOKENS
    )
    val = validator()
    simulator = LTSpiceSimulator()
    reward_fn = RewardFunctionNorm()

    # Load constraint
    constraint = load_constraint(CONSTRAINT_PATH, idx=CONSTRAINT_INDEX)
    
    # Initialize the state database using config hyper-parameters
    db = NetlistDatabase(temperature=TEMP)

    # Setup directories
    data_dir = Path("pipeline/data")
    run_folder_name = get_next_run_folder(data_dir, prefix=RUN_PREFIX)
    run_folder_path = data_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    database_dir = run_folder_path / "database"
    database_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Summary Logger
    sim_params = {
        "sampled_states": SAMPLED_STATES_PER_BATCH,
        "cands_per_prompt": CANDIDATES_PER_PROMPT,
        "mcts_temp": TEMP,
        "top_k": TOP_K,
        "epsilon": EPSILON,
        "max_tokens": MAX_TOKENS,
        "model_id": MODEL_ID
    }
    
    logger = SummaryLogger(
        run_folder_path=run_folder_path,
        n_batches=N_BATCHES,
        sim_params=sim_params,
        weights=weights
    )

    print(f"\n{'='*50}")
    print(f"🚀 Starting Iterative Inference Loop: {run_folder_name}")
    print(f"🧠 Model: {MODEL_ID}")
    print(f"🎯 Constraint: {custom_label}")
    print(f"{'='*50}\n")
    
    run_start_time = time.time()

    # --- The Execution Loop ---
    for i in range(1, N_BATCHES + 1):
        batch_start_time = time.time()

        current_batch_id = f"{run_folder_name}/batch_{i}"
        print(f"\n--- Processing {current_batch_id} ---")

        # 1. Sample Starting States
        sampled_states = db.sample_states(n=SAMPLED_STATES_PER_BATCH, top_k=TOP_K, epsilon=EPSILON)
        
        if not sampled_states:
            print("Database empty. Generating seed netlists from scratch.")
            written = llm.generate_for_batch(
                constraint, 
                batchID=current_batch_id, 
                n=SAMPLED_STATES_PER_BATCH * CANDIDATES_PER_PROMPT,
                DEMO=True,
                label=custom_label
            )
        else:
            print(f"Sampled {len(sampled_states)} states from database. Branching {CANDIDATES_PER_PROMPT} candidates each.")
            written = llm.generate_from_states(
                constraint,
                batchID=current_batch_id,
                seed_states=sampled_states,       
                candidates_per_state=CANDIDATES_PER_PROMPT,
                label=custom_label
            )

        # 2. Validate
        print("Validating netlists...")
        val.validate(current_batch_id)

        # 3. Simulate
        print("Running simulations...")
        simulation_results = simulator.simulate(current_batch_id)

        # 4. Evaluate rewards
        print("Evaluating fitness...")
        reward_fn.process_batch(current_batch_id, constraint, weights=weights)
        
        # 5. Push to Database
        print("Updating State Database...")
        db.ingest_batch_data(current_batch_id, database_dir=database_dir)

        # 6. Log Metrics
        batch_duration = time.time() - batch_start_time
        best_cand_id = max(db.records, key=lambda k: db.records[k]["fitness"]) if db.records else None
        
        if best_cand_id:
            logger.log_batch(
                batch_idx=i, 
                batch_duration=batch_duration, 
                db=db, 
                best_cand_id=best_cand_id
            )
            print(f"--- Batch {i} Complete in {batch_duration:.2f} sec ---")
            print(f"--- Current Global Best: {db.records[best_cand_id]['fitness']:.4f} ({best_cand_id}) ---")

        # 7. Update Plots
        if UPDATE_PLOTS and plot_run_results:
            plot_run_results(str(run_folder_path))
            if plot_cumulative_probabilities:
                plot_cumulative_probabilities(run_id=run_folder_name, target_batch=i+1, temperature=TEMP, top_k=TOP_K, epsilon=EPSILON)
            if plot_softmax_probabilities:
                plot_softmax_probabilities(run_id=run_folder_name, target_batch=i+1, temperature=TEMP)
                        
    # Final plot generation if disabled during loop
    if not UPDATE_PLOTS and plot_run_results:
        print("\n📊 Generating final run plots...")
        plot_run_results(str(run_folder_path))

    run_duration = time.time() - run_start_time
    print(f"\n{'='*50}")
    print(f"✅ Inference Loop Complete in {run_duration:.2f} seconds")
    print(f"{'='*50}")
    
    return str(run_folder_path)


if __name__ == "__main__":
    # If called from CLI, load JSON and pass it as a dict.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file.")
    args = parser.parse_args()
    # 1. Define the universal config dict
    base_config = {
        "run_settings": {
            "n_batches": 15,
            "sampled_states_per_batch": 2,
            "candidates_per_prompt": 4,
            "max_tokens": 2048,
            "update_plots_per_batch": False, # Faster to only plot at the end for benchmarks
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "run_prefix": "Bench_Base" 
        },
        "mcts_settings": {
            "temperature": 0.05,
            "top_k": 15,
            "epsilon": 0.15
        },
        "constraint_settings": {
            "dataset_path": "pipeline/data/datasets/constraints.json",
            "index": 4,
            "phase": "Phase1"
        },
        "weights": {
            "v_out": 10.0, 
            "efficiency": 20.0,
            "volume": 2.0, 
            "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
        }
    }
    
    run_inference(base_config)