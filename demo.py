import os
import re
from pathlib import Path

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import LTSpiceSimulator
#from pipeline.simulation.ngspice_runner import NGSpiceSimulator
from pipeline.reward_evaluation.reward_function import RewardFunction
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint

# Import our new plotting function
try:
    from pipeline.graphs_and_visualizations.visualize_demo_results import plot_run_results
except ImportError:
    print("Warning: visualize_demo_results.py not found. Plotting disabled.")
    plot_run_results = None

def get_next_run_folder(data_dir: Path) -> str:
    """Scans the data directory and returns the next Run_XXX folder name."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Find all folders matching the pattern Run_XXX
    run_folders = [d.name for d in data_dir.iterdir() if d.is_dir() and re.match(r"Run_\d+", d.name)]
    
    if not run_folders:
        return "Run_001"
    
    # Extract the numbers, find the max, and add 1
    run_numbers = [int(f.split("_")[1]) for f in run_folders]
    next_run_number = max(run_numbers) + 1
    
    return f"Run_{next_run_number:03d}"

def main():
    # --- Configuration ---
    N_batch = 3  # Number of batches to run sequentially
    n_generations_per_batch = 2 # Number of circuits per batch
    
    UPDATE_PLOTS_PER_BATCH = True # <-- TOGGLE: Set to False to only plot at the very end
    
    # NEW: Context window hyperparameter
    MAX_TOKENS = 10 * 2048  # Increase this if your netlists are getting cut off

    # Define weight distribution for the reward function
    weights = {
        "v_out": 10.0, "efficiency": 20.0,
        "volume": 2.0, "component_cost": 1.0,
        "components": {"mosfet": 1.0, "diode": 1.0,
                       "inductor": 1.0, "capacitor": 1.0}
    }

    # --- Setup Pipeline Components ---
    llm = TopologyLLM(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=MAX_TOKENS 
    )
    val        = validator()
    simulator  = LTSpiceSimulator()
    reward_fn  = RewardFunctionNorm()
    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=2)

    # --- Setup Directories ---
    data_dir = Path("pipeline/data")
    run_folder_name = get_next_run_folder(data_dir)
    run_folder_path = data_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Starting Sequential Run: {run_folder_name} for {N_batch} batches ===")

    # Keep track of the previous batch so the LLM can use its data
    previous_batch_id = None

    # --- Sequential Batch Loop ---
    for i in range(1, N_batch + 1):
        # The underlying classes will resolve this relative to pipeline/data/
        current_batch_id = f"{run_folder_name}/batch_{i}"
        
        print(f"\n--- Processing {current_batch_id} ---")

        # 1. Generate
        written = llm.generate_for_batch(
            constraint, 
            batchID=current_batch_id, 
            n=n_generations_per_batch,
            DEMO=True,
            previous_batch_id=previous_batch_id 
        )
        print(f"Generated {len(written) if written else 0} netlists.")

        # 2. Validate
        print("Validating netlists...")
        val.validate(current_batch_id)

        # 3. Simulate
        print("Running simulations...")
        simulation_results = simulator.simulate(current_batch_id)

        # 4. Evaluate rewards
        print("Evaluating fitness and formatting JSON...")
        reward_fn.process_batch(current_batch_id, constraint, weights=weights)
        
        # 5. Plotting (Optional mid-run overwrite)
        if UPDATE_PLOTS_PER_BATCH and plot_run_results:
            plot_run_results(str(run_folder_path))
            
        print(f"--- Finished {current_batch_id} ---")

        # Update the previous_batch_id so the next iteration can feed it into the LLM
        previous_batch_id = current_batch_id

    # Final Plot Guarantee (runs if UPDATE_PLOTS_PER_BATCH was False)
    if not UPDATE_PLOTS_PER_BATCH and plot_run_results:
        print("\n📊 Generating final run plots...")
        plot_run_results(str(run_folder_path))

    print(f"\n=== Sequential Run {run_folder_name} Complete ===")

if __name__ == "__main__":
    main()