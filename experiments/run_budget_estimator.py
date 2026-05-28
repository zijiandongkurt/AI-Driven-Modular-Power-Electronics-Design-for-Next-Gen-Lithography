import os
import json
from pathlib import Path
import sys

# Add the parent directory (project root) to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from run_inference import run_inference

def main():
    print("\n" + "="*60)
    print("⏱️  RUNNING PIPELINE PROFILING TEST")
    print("="*60)

    # 1. Minimal config for a speed test (2 batches = 16 netlists)
    test_config = {
        "run_settings": {
            "n_batches": 2, 
            "sampled_states_per_batch": 2,
            "candidates_per_prompt": 4,
            "max_tokens": 2048,
            "update_plots_per_batch": False, 
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "run_prefix": "Timing_Profile_Run" 
        },
        "mcts_settings": {
            "temperature": 0.05,
            "top_k": 15,
            "epsilon": 0.15
        },
        "constraint_settings": {
            "dataset_path": "pipeline/data/datasets/constraints_easy.json",
            "index": 0,
            "phase": "Phase1"
        },
        "weights": {
            "v_out": 10.0, "efficiency": 20.0,
            "volume": 2.0, "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
        }
    }

    # 2. Run the inference loop (this uses your newly profiled demo_inference)
    run_folder = run_inference(test_config)
    
    # 3. Load the generated timing profile
    profile_path = Path(run_folder) / "timing_profile.json"
    
    if not profile_path.exists():
        print("❌ Could not find timing_profile.json. Make sure demo_inference.py is updated with the profiling code!")
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        profiling = json.load(f)

    # Calculate time spent specifically on processing netlists (exclude logging/plotting overhead)
    core_time = (
        profiling.get("llm_generation", 0.0) + 
        profiling.get("validation", 0.0) + 
        profiling.get("simulation", 0.0) + 
        profiling.get("reward_calculation", 0.0) + 
        profiling.get("database_io", 0.0)
    )
    
    total_netlists = test_config["run_settings"]["n_batches"] * \
                     test_config["run_settings"]["sampled_states_per_batch"] * \
                     test_config["run_settings"]["candidates_per_prompt"]

    time_per_netlist = core_time / total_netlists if total_netlists > 0 else 0

    print("\n" + "="*60)
    print("📊 PROFILING RESULTS & COST PROJECTIONS")
    print("="*60)
    print(f"Test Run Folder:       {run_folder}")
    print(f"Netlists Evaluated:    {total_netlists}")
    print(f"Total Core Time:       {core_time:.2f} seconds")
    print(f"Average Time/Netlist:  {time_per_netlist:.2f} seconds")
    print("-" * 60)

    # 4. Project Full Benchmark Costs
    static_total_netlists = 9720
    dynamic_total_netlists = 17280

    static_hours = (static_total_netlists * time_per_netlist) / 3600
    dynamic_hours = (dynamic_total_netlists * time_per_netlist) / 3600

    print("🚀 Projection 1: STATIC Search Horizon (15 Batches for all phases)")
    print(f"   Total Netlists: {static_total_netlists:,}")
    print(f"   Estimated Time: {static_hours:.2f} hours")
    print()
    print("🚀 Projection 2: DYNAMIC Search Horizon (15/25/40 Batches)")
    print(f"   Total Netlists: {dynamic_total_netlists:,}")
    print(f"   Estimated Time: {dynamic_hours:.2f} hours")
    print("-" * 60)
    
    # 5. SBU Cost Estimator
    print("💰 SBU Compute Budget Estimator")
    sbu_rate_str = input("Enter your cluster's SBU rate per hour (e.g., 2.5 for an A100 node): ")
    
    try:
        sbu_rate = float(sbu_rate_str)
        print(f"   Static Setup Cost:  {static_hours * sbu_rate:.2f} SBUs")
        print(f"   Dynamic Setup Cost: {dynamic_hours * sbu_rate:.2f} SBUs")
    except ValueError:
        print("   Invalid SBU rate entered. Skipping cost calculation.")

    print("="*60)
    print("Note: Phase 3 (Hard) simulations may take slightly longer per netlist due to complex")
    print("transient states. Add a ~10% buffer to these estimates to be safe!")

if __name__ == "__main__":
    main()