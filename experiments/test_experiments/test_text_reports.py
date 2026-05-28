import os
import numpy as np
from pathlib import Path

# Add the parent directory to paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def test_benchmark_report_generation():
    print("\n" + "="*50)
    print("📝 TESTING BENCHMARK .TXT REPORT GENERATION")
    print("="*50)
    
    # 1. Mock Data (Simulating a finished run)
    models = {"Base": "", "SFT": "", "RL": ""}
    tasks = [
        {"label": "P1_Buck_Std"},
        {"label": "P2_Buck_HighPwr"},
        {"label": "P3_Buck_Mains"}
    ]
    
    master_results = {model: {task["label"]: [] for task in tasks} for model in models.keys()}
    
    # Fill with dummy metrics
    for model in models:
        for task in tasks:
            for _ in range(3):
                master_results[model][task["label"]].append({
                    "fitness": np.random.uniform(0.6, 0.95),
                    "auc": np.random.uniform(0.5, 0.8),
                    "validity": np.random.uniform(40, 100)
                })

    # 2. The exact formatting block from run_benchmark.py
    report_text = f"\n\n{'='*100}\n"
    report_text += f"🏆 COMPREHENSIVE BENCHMARK RESULTS (Mean ± StdDev) 🏆\n"
    report_text += f"{'='*100}\n"
    
    header = f"{'Task (Constraint)':<22} | {'Metric':<14} | " + " | ".join([f"{m:<18}" for m in models.keys()])
    report_text += header + "\n"
    report_text += "-" * 100 + "\n"

    for task in tasks:
        task_lbl = task['label']
        report_text += f"{task_lbl:<22} | \n"
        
        for metric_name, key in [("Max Fitness", "fitness"), ("AUC (Speed)", "auc"), ("Validity %", "validity")]:
            row = f"{'':<22} | {metric_name:<14} | "
            for model_name in models.keys():
                trial_data = master_results[model_name][task_lbl]
                vals = [t[key] for t in trial_data if t]
                
                if vals:
                    row += f"{np.mean(vals):.2f} ± {np.std(vals):.2f}    | "
                else:
                    row += f"{'ERR':<18} | "
            report_text += row + "\n"
        report_text += "-" * 100 + "\n"

    # 3. Save it
    out_path = PROJECT_ROOT / "experiments" / "test_outputs"
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "benchmark_report.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # 4. Verify it
    if report_file.exists() and report_file.stat().st_size > 0:
        print(f"✅ Success! Benchmark report saved to: {report_file}")
        print("Preview of generated table:")
        # Print just the first 15 lines so it doesn't flood the terminal
        with open(report_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print("".join(lines[:15]))
    else:
        print("❌ Failed to generate benchmark report.")


def test_budget_report_generation():
    print("\n" + "="*50)
    print("💰 TESTING BUDGET ESTIMATOR .TXT GENERATION")
    print("="*50)

    # 1. Mock variables
    run_folder = "pipeline/data/Timing_Profile_Run"
    total_netlists = 16
    core_time = 45.32
    time_per_netlist = core_time / total_netlists
    
    static_total_netlists = 9720
    dynamic_total_netlists = 17280
    static_hours = (static_total_netlists * time_per_netlist) / 3600
    dynamic_hours = (dynamic_total_netlists * time_per_netlist) / 3600
    
    sbu_rate = 196 # Mocking the user input

    # 2. The exact formatting block from run_budget_estimator.py
    report_text = f"""============================================================
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
    
    cost_text = f"""💰 SBU Compute Budget Estimator
   SBU Rate Applied:   {sbu_rate}
   Static Setup Cost:  {static_hours * sbu_rate:.2f} SBUs
   Dynamic Setup Cost: {dynamic_hours * sbu_rate:.2f} SBUs
"""

    footer = """============================================================
Note: Phase 3 (Hard) simulations may take slightly longer per netlist due to complex
transient states. Add a ~10% buffer to these estimates to be safe!
"""
    
    # 3. Save it
    full_report = report_text + cost_text + footer
    out_path = PROJECT_ROOT / "experiments" / "test_outputs"
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "budget_estimation.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report)
        
    # 4. Verify it
    if report_file.exists() and report_file.stat().st_size > 0:
        print(f"✅ Success! Budget report saved to: {report_file}")
    else:
        print("❌ Failed to generate budget report.")

if __name__ == "__main__":
    test_benchmark_report_generation()
    test_budget_report_generation()