import os
import re
import json
import numpy as np
from pathlib import Path

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
        self.depth_alpha = depth_alpha  # How much we reward pushing a design deeper

    def add_record(self, candidate_id, netlist_text, fitness, depth, feedback_text):
        self.records[candidate_id] = {
            "netlist_text": netlist_text,
            "fitness": fitness,
            "depth": depth,
            "feedback": feedback_text
        }

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


def extract_batch_data_to_db(batch_id: str, db: NetlistDatabase, parent_depths: dict):
    """
    Parses the results of a batch and pushes all candidates into the database.
    parent_depths is a dict mapping {candidate_id: parent_depth} to calculate new depth.
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
        netlist_text = net_file.read_text()

        # 2. Get Fitness
        fitness = metrics.get("fitness_score", -1.0)

        # 3. Calculate new Depth
        # Default to 0 if parent unknown (e.g., seed batch), otherwise parent + 1
        parent_depth = parent_depths.get(cand_id, 0)
        new_depth = parent_depth + 1 if parent_depth > 0 else 1

        # 4. Construct Feedback String for the LLM's next prompt
        feedback_text = f"Fitness Score: {fitness:.4f}\n"
        if metrics.get("source") == "validation_penalty":
            failed_checks = [k for k, v in validations.get(cand_id, {}).get("checks", {}).items() if not v]
            feedback_text += f"FAILED TESTS: {', '.join(failed_checks)}\n"
        else:
            raw = metrics.get("raw_metrics", {})
            feedback_text += f"V_out: {raw.get('simulation_output_voltage', 0):.2f}V, Efficiency: {raw.get('efficiency', 0):.2f}\n"

        # 5. Push to Database
        db.add_record(cand_id, netlist_text, fitness, new_depth, feedback_text)


def main():
    # --- Configuration ---
    N_BATCHES = 5
    SAMPLED_STATES_PER_BATCH = 2
    CANDIDATES_PER_PROMPT = 4  # Total generated per batch = 2 * 4 = 8
    
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
    constraint = load_constraint("pipeline/data/datasets/constraints_easy.json", idx=0)
    
    # Initialize the new state database
    db = NetlistDatabase(temperature=0.3, depth_alpha=0.1)

    data_dir = Path("pipeline/data")
    run_folder_name = get_next_run_folder(data_dir)
    run_folder_path = data_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Starting Iterative Inference Loop: {run_folder_name} ===")

    for i in range(1, N_BATCHES + 1):
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
                DEMO=True
            )
        else:
            print(f"Sampled {len(sampled_states)} states from database. Branching {CANDIDATES_PER_PROMPT} candidates each.")
            
            # NOTE: You will need to update TopologyLLM to accept `seed_states` 
            # instead of `previous_batch_id` so it can format the prompts correctly.
            written = llm.generate_from_states(
                constraint,
                batchID=current_batch_id,
                seed_states=sampled_states,       # Pass the text, feedback, and scores
                candidates_per_state=CANDIDATES_PER_PROMPT
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
        extract_batch_data_to_db(current_batch_id, db, parent_depths=parent_depth_map)
        print(f"Database now contains {len(db.records)} evaluated states.")

        # 6. Plotting
        if UPDATE_PLOTS_PER_BATCH and plot_run_results:
            plot_run_results(str(run_folder_path))

    if not UPDATE_PLOTS_PER_BATCH and plot_run_results:
        print("\n📊 Generating final run plots...")
        plot_run_results(str(run_folder_path))

    print(f"\n=== Inference Loop Complete ===")

if __name__ == "__main__":
    main()