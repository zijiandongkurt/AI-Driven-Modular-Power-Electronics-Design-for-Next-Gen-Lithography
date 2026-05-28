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

    # 2. Run the inference loop
    run_folder = run_inference(test_config)
    
    # 3. Load the generated timing profile
    profile_path = Path(run_folder) / "timing_profile.json"
    
    if not profile_path.exists():
        print("❌ Could not find timing_profile.json. Make sure demo_inference.py is updated with the profiling code!")
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        profiling = json.load(f)

    # Calculate time spent specifically on processing netlists
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

    # 4. Project Full Benchmark Costs
    static_total_netlists = 9720
    dynamic_total_netlists = 17280

    static_hours = (static_total_netlists * time_per_netlist) / 3600
    dynamic_hours = (dynamic_total_netlists * time_per_netlist) / 3600

    # Build the report string
    report_text = f"""
============================================================
📊 PROFILING RESULTS & COST PROJECTIONS
============================================================
Test Run Folder:       {run_folder}
Netlists Evaluated:    {total_netlists}
Total Core Time:       {core_time:.2f} seconds
Average Time/Netlist:  {time_per_netlist:.2f} seconds
------------------------------------------------------------
🚀 Projection 1: STATIC Search Horizon (15 Batches for all phases)
   Total Netlists: {static_total_netlists:,}
   Estimated Time: {static_hours:.2f} hours

🚀 Projection 2: DYNAMIC Search Horizon (15/25/40 Batches)
   Total Netlists: {dynamic_total_netlists:,}
   Estimated Time: {dynamic_hours:.2f} hours
------------------------------------------------------------
"""
    print(report_text)
    
    # 5. SBU Cost Estimator
    print("💰 SBU Compute Budget Estimator")
    sbu_rate_str =196
    
    cost_text = ""
    try:
        sbu_rate = float(sbu_rate_str)
        cost_text = f"""💰 SBU Compute Budget Estimator
   SBU Rate Applied:   {sbu_rate}
   Static Setup Cost:  {static_hours * sbu_rate:.2f} SBUs
   Dynamic Setup Cost: {dynamic_hours * sbu_rate:.2f} SBUs
"""
        print(cost_text)
    except ValueError:
        cost_text = "   Invalid SBU rate entered. Cost calculation skipped.\n"
        print(cost_text)

    footer = """============================================================
Note: Phase 3 (Hard) simulations may take slightly longer per netlist due to complex
transient states. Add a ~10% buffer to these estimates to be safe!
"""
    print(footer)

    # 6. Save everything to budget_estimation.txt
    full_report = report_text + cost_text + footer
    output_file = PROJECT_ROOT / "experiments" / "budget_estimation.txt"
    
    # Ensure experiments directory exists just in case
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_report)
        
    print(f"✅ Budget estimation successfully saved to: {output_file}")

if __name__ == "__main__":
    main()