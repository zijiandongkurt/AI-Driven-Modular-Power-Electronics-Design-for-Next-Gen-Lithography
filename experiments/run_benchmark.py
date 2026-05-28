import copy
import re
import numpy as np
from pathlib import Path
import json
import sys
import os
from pathlib import Path

# Add the parent directory (project root) to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from run_inference import run_inference

# Import all plotting modules from the new visualizations directory
try:
    from pipeline.graphs_and_visualizations.plot_validity_bar_chart import plot_validity_bar_chart
    from pipeline.graphs_and_visualizations.plot_radar_chart import plot_radar_chart
    from pipeline.graphs_and_visualizations.plot_pareto_scatter import plot_pareto_scatter
    from pipeline.graphs_and_visualizations.plot_learning_curves import plot_learning_curves
    from pipeline.graphs_and_visualizations.plot_combined_fitness import plot_combined_fitness
except ImportError as e:
    print(f"Warning: Plotting modules not found. Plotting disabled. Error: {e}")

def extract_metrics(run_folder: str) -> dict:
    """Parses the run summary and champion JSON to extract fitness and physical metrics."""
    summary_path = Path(run_folder) / "run_summary.txt"
    champ_path = Path(run_folder) / "champion_metrics.json"
    
    metrics = {
        "fitness": 0.0, "auc": 0.0, "validity": 0.0, "learning_curve": [], 
        "v_error_pct": None, "efficiency": None, 
        "volume": None, "components": None, "target_power": None
    }
    
    # Extract Learning & Fitness Data
    if summary_path.exists():
        content = summary_path.read_text(encoding="utf-8")
        fit_match = re.search(r"Overall Best Fitness:\s*([0-9.]+)", content)
        metrics["fitness"] = float(fit_match.group(1)) if fit_match else 0.0
        
        val_match = re.search(r"Validity Rate:\s*([0-9.]+)%", content)
        metrics["validity"] = float(val_match.group(1)) if val_match else 0.0
        
        raw_scores = [float(x) for x in re.findall(r"Batch \d+ Best DB Score:\s*([-0-9.]+)", content)]
        metrics["learning_curve"] = raw_scores
        
        norm_scores = [0.0 if s < 0.5 else (s - 0.5) * 2.0 for s in raw_scores]
        metrics["auc"] = np.trapz(norm_scores) if len(norm_scores) > 1 else 0.0

    # Extract Physical Trade-off Data
    if champ_path.exists():
        with open(champ_path, "r", encoding="utf-8") as f:
            champ = json.load(f)
            target_v = champ.get("target_voltage", 1.0) 
            raw_v = champ.get("raw_voltage", 0.0)
            
            error_pct = abs(raw_v - target_v) / target_v * 100.0
            
            metrics["v_error_pct"] = error_pct
            metrics["efficiency"] = champ.get("raw_efficiency", 0.0) * 100.0 
            
            metrics["volume"] = champ.get("raw_volume", 0.0)
            metrics["components"] = champ.get("raw_components", 0)
            metrics["target_power"] = champ.get("target_power", 10.0)
            
    return metrics

def main():
    TRIALS_PER_TASK = 3 
    
    base_config = {
        "run_settings": {
            "n_batches": 15,
            "sampled_states_per_batch": 2,
            "candidates_per_prompt": 4,
            "max_tokens": 2048,
            "update_plots_per_batch": False, 
            "model_id": "",
            "run_prefix": "" 
        },
        "mcts_settings": {
            "temperature": 0.05,
            "top_k": 15,
            "epsilon": 0.15
        },
        "constraint_settings": {
            "dataset_path": "",
            "index": 0,
            "phase": ""
        },
        "weights": {
            "v_out": 10.0, "efficiency": 20.0,
            "volume": 2.0, "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
        }
    }

    models = {
        "Base": "Qwen/Qwen2.5-3B-Instruct"#,
        #"SFT": "path/to/your/sft/model",     
        #"RL": "path/to/your/rl/model"        
    }

    tasks = [
        {"phase": "Phase1", "label": "P1_Buck_Std",      "file": "pipeline/data/datasets/constraints_easy.json",   "idx": 0},
        {"phase": "Phase1", "label": "P1_Boost_Std",     "file": "pipeline/data/datasets/constraints_easy.json",   "idx": 6},
        {"phase": "Phase1", "label": "P1_Buck_RatioLim", "file": "pipeline/data/datasets/constraints_easy.json",   "idx": 16},
        
        {"phase": "Phase2", "label": "P2_Buck_Extreme",  "file": "pipeline/data/datasets/constraints_medium.json", "idx": 0},
        {"phase": "Phase2", "label": "P2_Buck_HighPwr",  "file": "pipeline/data/datasets/constraints_medium.json", "idx": 10},
        {"phase": "Phase2", "label": "P2_Boost_Extreme", "file": "pipeline/data/datasets/constraints_medium.json", "idx": 19},
        
        {"phase": "Phase3", "label": "P3_Buck_Mains",    "file": "pipeline/data/datasets/constraints_hard.json",   "idx": 0},
        {"phase": "Phase3", "label": "P3_Boost_Extreme", "file": "pipeline/data/datasets/constraints_hard.json",   "idx": 1},
        {"phase": "Phase3", "label": "P3_Buck_MaxPwr",   "file": "pipeline/data/datasets/constraints_hard.json",   "idx": 10},
    ]

    master_results = {model: {task["label"]: [] for task in tasks} for model in models.keys()}

    for model_name, model_path in models.items():
        for task in tasks:
            for trial in range(1, TRIALS_PER_TASK + 1):
                run_config = copy.deepcopy(base_config)
                run_config["run_settings"]["model_id"] = model_path
                run_config["run_settings"]["run_prefix"] = f"Bench_{model_name}_{task['label']}_T{trial}"
                run_config["run_settings"]["seed"] = 42000 + trial
                run_config["constraint_settings"]["dataset_path"] = task["file"]
                run_config["constraint_settings"]["index"] = task["idx"]
                run_config["constraint_settings"]["phase"] = task["phase"]

                try:
                    output_folder = run_inference(run_config)
                    metrics = extract_metrics(output_folder)
                    master_results[model_name][task['label']].append(metrics)
                except Exception as e:
                    master_results[model_name][task['label']].append({"fitness": 0.0, "auc": 0.0, "validity": 0.0})

    # --- FINAL MULTI-METRIC REPORT ---
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

    # 1. Print to console
    print(report_text)

    # 2. Save to .txt file
    out_path = "experiments/benchmark_results"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    report_file = Path(out_path) / "benchmark_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"📝 Benchmark table successfully saved to: {report_file}\n")

    # Triggering Extracted Plotting Modules
    try:
        plot_combined_fitness(master_results, output_dir=out_path)
        plot_learning_curves(master_results, output_dir=out_path, n_batches=base_config["run_settings"]["n_batches"])
        plot_pareto_scatter(master_results, output_dir=out_path)
        plot_radar_chart(master_results, output_dir=out_path)
        plot_validity_bar_chart(master_results, output_dir=out_path)
    except NameError:
        print("Plots skipped because the visualization modules could not be loaded.")

if __name__ == "__main__":
    main()