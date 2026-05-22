import os
import re
import json
import numpy as np
from pathlib import Path
import time

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint

try:
    from pipeline.graphs_and_visualizations.visualize_demo_results import plot_run_results
except ImportError:
    print("Warning: visualize_demo_results.py not found. Plotting disabled.")
    plot_run_results = None


class NetlistDatabase:
    """
    In-memory database to track all generated netlists, their fitness, and their depth.
    Implements Temperature-Scaled Softmax sampling to balance exploration and exploitation.
    """
    def __init__(self, temperature=0.2, depth_alpha=0.05):
        self.records = {}
        self.temperature = temperature
        self.depth_alpha = depth_alpha
        
        # NEW: Track validity statistics
        self.total_valid = 0
        self.total_invalid = 0

    def add_record(self, candidate_id, netlist_text, fitness, depth, feedback_text, is_valid=False):
        self.records[candidate_id] = {
            "netlist_text": netlist_text,
            "fitness": fitness,
            "depth": depth,
            "feedback": feedback_text,
            "is_valid": is_valid
        }
        
        if is_valid:
            self.total_valid += 1
        else:
            self.total_invalid += 1

    def sample_states(self, n=2):
        """Samples 'n' states probabilistically based on fitness and depth."""
        if not self.records:
            return []
        
        keys = list(self.records.keys())
        scores = []
        
        for k in keys:
            rec = self.records[k]
            # Score = Fitness + (Alpha * Depth)
            score = rec["fitness"] + (self.depth_alpha * rec["depth"])
            scores.append(score)
            
        scores = np.array(scores)
        
        # Temperature-scaled Softmax
        # Subtract max for numerical stability before exp
        exp_scores = np.exp((scores - np.max(scores)) / self.temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        # Sample without replacement (if we have enough states)
        sample_size = min(n, len(keys))
        chosen_keys = np.random.choice(keys, size=sample_size, p=probs, replace=False)
        
        return [{"id": k, **self.records[k]} for k in chosen_keys]


def get_next_run_folder(data_dir: Path) -> str:
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    run_folders = [d.name for d in data_dir.iterdir() if d.is_dir() and re.match(r"Run_\d+", d.name)]
    if not run_folders:
        return "Run_001"
    run_numbers = [int(f.split("_")[1]) for f in run_folders]
    return f"Run_{max(run_numbers) + 1:03d}"


def extract_batch_data_to_db(batch_id: str, db: NetlistDatabase, parent_depths: dict, database_dir: Path):
    """
    Parses the results of a batch and pushes all candidates into the database.
    Also saves a physical copy of the netlist to the global database folder.
    """
    data_dir = Path("pipeline/data") / batch_id
    reward_path = data_dir / "reward_results.json"
    val_path = data_dir / "validation_results.json"
    llm_out_dir = data_dir / "LLM_output"

    if not reward_path.exists():
        return

    with open(reward_path, "r") as f:
        rewards = json.load(f).get("circuits", {})
        
    with open(val_path, "r") as f:
        validations = json.load(f) if val_path.exists() else {}

    for cand_id, metrics in rewards.items():
        # 1. Get Netlist Text
        net_file = llm_out_dir / f"{cand_id}.net"
        if not net_file.exists():
            continue
        netlist_text = net_file.read_text(encoding="utf-8")

        # ---> NEW: Save copy to the central database folder <---
        db_file = database_dir / f"{cand_id}.net"
        db_file.write_text(netlist_text, encoding="utf-8")

        # 2. Get Fitness
        fitness = metrics.get("fitness_score", -1.0)

        # 3. Calculate new Depth
        parent_depth = parent_depths.get(cand_id, 0)
        new_depth = parent_depth + 1 if parent_depth > 0 else 1
        is_valid = validations.get(cand_id, {}).get("passed", False)

        # 4. Construct Feedback String
        feedback_text = f"Fitness Score: {fitness:.4f}\n"
        if metrics.get("source") == "validation_penalty":
            failed_checks = [k for k, v in validations.get(cand_id, {}).get("checks", {}).items() if not v]
            feedback_text += f"FAILED TESTS: {', '.join(failed_checks)}\n"
        else:
            raw = metrics.get("raw_metrics", {})
            feedback_text += f"V_out: {raw.get('simulation_output_voltage', 0):.2f}V, Efficiency: {raw.get('efficiency', 0):.2f}\n"

        # 5. Push to Database
        db.add_record(cand_id, netlist_text, fitness, new_depth, feedback_text, is_valid=is_valid)

def main():
    # --- Configuration ---
    N_BATCHES = 4
    SAMPLED_STATES_PER_BATCH = 1
    CANDIDATES_PER_PROMPT = 2  # Total generated per batch = 2 * 4 = 8
    
    MAX_TOKENS = 2048
    UPDATE_PLOTS_PER_BATCH = True

    weights = {
        "v_out": 10.0, "efficiency": 20.0,
        "volume": 2.0, "component_cost": 1.0,
        "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
    }

    # --- Setup Pipeline ---
    llm = TopologyLLM(
        model_id="Qwen/Qwen2.5-3B-Instruct", 
        max_new_tokens=MAX_TOKENS)
    val = validator()
    simulator = LTSpiceSimulator()
    reward_fn = RewardFunctionNorm()

    # --- NEW: Define your constraint and custom label here ---
    CONSTRAINT_INDEX = 4
    CONSTRAINT_PHASE = "Phase1"
    
    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=CONSTRAINT_INDEX)
    custom_label = f"{CONSTRAINT_PHASE}_cons{CONSTRAINT_INDEX}"
    constraint = load_constraint("pipeline/data/datasets/constraints_easy.json", idx=0)
    
    # Initialize the new state database
    db = NetlistDatabase(temperature=0.3, depth_alpha=0.1)

    data_dir = Path("pipeline/data")
    run_folder_name = get_next_run_folder(data_dir)
    run_folder_path = data_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    results_dir = run_folder_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    database_dir = run_folder_path / "database"
    database_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Iterative Inference Loop: {run_folder_name} ===")
    run_start_time = time.time()
    batch_times = []
    best_fitness_history = []

    for i in range(1, N_BATCHES + 1):
        batch_start_time = time.time()

        current_batch_id = f"{run_folder_name}/batch_{i}"
        print(f"\n--- Processing {current_batch_id} ---")

        # 1. Sample Starting States
        sampled_states = db.sample_states(n=SAMPLED_STATES_PER_BATCH)
        parent_depth_map = {} # Maps the upcoming new candidates to their parent's depth
        
        if not sampled_states:
            print("Database empty. Generating seed netlists from scratch.")
            # Needs to generate standard zero-shot prompts
            written = llm.generate_for_batch(
                constraint, 
                batchID=current_batch_id, 
                n=SAMPLED_STATES_PER_BATCH * CANDIDATES_PER_PROMPT,
                DEMO=True,
                label=custom_label
            )
        else:
            print(f"Sampled {len(sampled_states)} states from database. Branching {CANDIDATES_PER_PROMPT} candidates each.")
            
            # NOTE: You will need to update TopologyLLM to accept `seed_states` 
            # instead of `previous_batch_id` so it can format the prompts correctly.
            written = llm.generate_from_states(
                constraint,
                batchID=current_batch_id,
                seed_states=sampled_states,       
                candidates_per_state=CANDIDATES_PER_PROMPT,
                label=custom_label
            )
            
            # Map the newly written candidate filenames to their parent's depth 
            # (Assuming llm returns a dict mapping new_cand_id -> parent_id)
            for new_cand_id, parent_id in written.items():
                parent_depth_map[new_cand_id] = db.records[parent_id]["depth"]

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
        extract_batch_data_to_db(current_batch_id, db, parent_depths=parent_depth_map, database_dir=database_dir)
        print(f"Database now contains {len(db.records)} evaluated states.")

        # -------------------------------------------------------------
        # METRICS & SUMMARY GENERATION (Runs at the end of every batch)
        # -------------------------------------------------------------
        batch_duration = time.time() - batch_start_time
        batch_times.append(batch_duration)
        avg_time = sum(batch_times) / len(batch_times)
        
        # Find the absolute best candidate currently in the database
        best_cand_id = max(db.records, key=lambda k: db.records[k]["fitness"])
        best_record = db.records[best_cand_id]
        best_fitness = best_record["fitness"]
        best_fitness_history.append(best_fitness)
        
        # 1. Overwrite the Best Netlist File
        best_netlist_path = results_dir / "best_topology.net"
        with open(best_netlist_path, "w", encoding="utf-8") as f:
            f.write(f"* OVERALL BEST CANDIDATE: {best_cand_id}\n")
            f.write(f"* FITNESS SCORE: {best_fitness:.4f}\n*\n")
            f.write(best_record["netlist_text"])
            
        # 2. Calculate Validity Percentages
        total_generated = db.total_valid + db.total_invalid
        valid_pct = (db.total_valid / total_generated * 100) if total_generated > 0 else 0
        
        # 3. Overwrite the Summary Log
        summary_path = results_dir / "run_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== MCTS INFERENCE SUMMARY: {run_folder_name} ===\n")
            f.write(f"Batches Completed: {i} / {N_BATCHES}\n")
            f.write(f"Total Time Elapsed: {sum(batch_times):.2f} sec ({sum(batch_times)/60:.2f} min)\n")
            f.write(f"Average Time per Batch: {avg_time:.2f} sec\n\n")
            
            f.write("--- TOPOLOGY YIELD ---\n")
            f.write(f"Total Candidates Generated: {total_generated}\n")
            f.write(f"Valid Topologies: {db.total_valid}\n")
            f.write(f"Invalid Topologies: {db.total_invalid}\n")
            f.write(f"Validity Rate: {valid_pct:.1f}%\n\n")
            
            f.write("--- FITNESS PROGRESSION ---\n")
            f.write(f"Overall Best Fitness: {best_fitness:.4f} ({best_cand_id})\n")
            for b_idx, b_fit in enumerate(best_fitness_history, 1):
                f.write(f"  Batch {b_idx} Best DB Score: {b_fit:.4f}\n")
                
            f.write("\n=========================================\n")
            f.write("OVERALL BEST SPICE NETLIST:\n")
            f.write("=========================================\n")
            f.write(best_record["netlist_text"])
            f.write("\n")

        print(f"--- Batch {i} Complete in {batch_duration:.2f} sec ---")
        print(f"--- Current Global Best: {best_fitness:.4f} ({best_cand_id}) ---")

        # 6. Plotting
        if UPDATE_PLOTS_PER_BATCH and plot_run_results:
            plot_run_results(str(run_folder_path))

    if not UPDATE_PLOTS_PER_BATCH and plot_run_results:
        print("\n📊 Generating final run plots...")
        plot_run_results(str(run_folder_path))

    run_duration = time.time() - run_start_time
    print(f"\n=== Inference Loop Complete in {run_duration:.2f} seconds ===")

if __name__ == "__main__":
    main()