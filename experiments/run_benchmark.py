import argparse
import copy
import re
import numpy as np
from pathlib import Path
import json
import sys
import os
import tempfile

# Add the parent directory (project root) to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from run_inference import run_inference
from pipeline.utility.topology_hasher import get_topological_hash

# Import all plotting modules from the new visualizations directory
try:
    from pipeline.graphs_and_visualizations.plot_validity_bar_chart import plot_validity_bar_chart
    from pipeline.graphs_and_visualizations.plot_radar_chart import plot_radar_chart
    from pipeline.graphs_and_visualizations.plot_pareto_scatter import plot_pareto_scatter
    from pipeline.graphs_and_visualizations.plot_learning_curves import plot_learning_curves
    from pipeline.graphs_and_visualizations.plot_combined_fitness import plot_combined_fitness
except ImportError as e:
    print(f"Warning: Plotting modules not found. Plotting disabled. Error: {e}")

def get_empty_metrics() -> dict:
    """Return a perfectly structured empty metrics dictionary to prevent KeyErrors."""
    return {
        "fitness": 0.0, 
        "auc": 0.0, 
        "validity": 0.0, 
        "duplicate_rate": None, 
        "learning_curve": [], 
        "v_error_pct": None, 
        "efficiency": None, 
        "volume": None, 
        "components": None, 
        "target_power": None,
        "total_time_sec": None,
        "time_per_netlist_sec": None
    }

def extract_metrics(run_folder: str) -> dict:
    """Parse the run summary, champion JSON, and netlist files to extract all metrics."""
    folder_path = Path(run_folder)
    summary_path = folder_path / "run_summary.txt"
    champ_path = folder_path / "champion_metrics.json"
    
    metrics = get_empty_metrics()
    
    # Extract Learning, Fitness, and Hardware Tracking Data
    if summary_path.exists():
        content = summary_path.read_text(encoding="utf-8")
        
        fit_match = re.search(r"Overall Best Fitness:\s*([0-9.]+)", content)
        metrics["fitness"] = float(fit_match.group(1)) if fit_match else 0.0
        
        val_match = re.search(r"Validity Rate:\s*([0-9.]+)%", content)
        metrics["validity"] = float(val_match.group(1)) if val_match else 0.0
        
        # Hardware Time Metrics
        time_match = re.search(r"Total Time Elapsed:\s*([0-9.]+)", content)
        metrics["total_time_sec"] = float(time_match.group(1)) if time_match else None
        
        net_time_match = re.search(r"Average Time per Netlist:\s*([0-9.]+)", content)
        metrics["time_per_netlist_sec"] = float(net_time_match.group(1)) if net_time_match else None
        
        raw_scores = [float(x) for x in re.findall(r"Batch \d+ Best DB Score:\s*([-0-9.]+)", content)]
        metrics["learning_curve"] = raw_scores
        
        norm_scores = [0.0 if s < 0.5 else (s - 0.5) * 2.0 for s in raw_scores]
        
        # --- FIX: NumPy 2.0 Compatibility (trapz -> trapezoid) ---
        if len(norm_scores) > 1:
            try:
                metrics["auc"] = np.trapezoid(norm_scores)
            except AttributeError:
                metrics["auc"] = np.trapz(norm_scores) # Fallback for older numpy versions
        else:
            metrics["auc"] = 0.0

    # Calculate Duplicate Rate using the WL Graph Hasher
    netlist_files = [f for f in folder_path.rglob("*.net") if f.name != "best_topology.net"]
    unique_hashes = set()
    total_netlists = 0
    
    for net_file in netlist_files:
        try:
            net_text = net_file.read_text(encoding="utf-8")
            topo_hash = get_topological_hash(net_text)
            unique_hashes.add(topo_hash)
            total_netlists += 1
        except Exception:
            continue
            
    # FIX: Safe duplicate rate to prevent "Fake 0%" when files are missing
    if total_netlists > 0:
        metrics["duplicate_rate"] = (1.0 - (len(unique_hashes) / total_netlists)) * 100.0

    # Extract Physical Trade-off Data
    if champ_path.exists():
        try:
            with open(champ_path, "r", encoding="utf-8") as f:
                champ = json.load(f)
                target_v = champ.get("target_voltage", 1.0) 
                raw_v = champ.get("raw_voltage", 0.0)
                
                safe_target_v = target_v if target_v != 0 else 1e-6
                metrics["v_error_pct"] = abs(raw_v - target_v) / safe_target_v * 100.0
                
                metrics["efficiency"] = champ.get("raw_efficiency", 0.0) * 100.0 
                metrics["volume"] = champ.get("raw_volume", 0.0)
                metrics["components"] = champ.get("raw_components", 0)
                metrics["target_power"] = champ.get("target_power", 10.0)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Corrupted champion_metrics.json found in {run_folder}. Using empty metrics.")
            
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark_config.json", help="Path to benchmark config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    assert config_path.exists(), f"Config file not found: {config_path.resolve()}"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    TRIALS_PER_TASK = config["trials_per_task"]
    base_config     = config["base_config"]
    tasks           = config["tasks"]

    # Pull models from config, fallback to hardcoded if not present
    models = config.get("models", {
        "Baseline": None,
        "sft": "checkpoints/sft_model", 
        "trained_easy": "checkpoints/zycos_008/grpo-lora/final",
        "trained_medium": "checkpoints/zycos_009/grpo-lora/final",
        "trained_hard": "checkpoints/zycos_010/grpo-lora/final"
    })

    out_path = Path("experiments/benchmark_results")
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_path / "benchmark_checkpoint.json"

    # --- NEW: Checkpointing System ---
    if checkpoint_file.exists():
        print(f"🔄 Resuming from existing checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            master_results = json.load(f)
    else:
        master_results = {model: {task["label"]: [] for task in tasks} for model in models.keys()}

    for model_name, model_path in models.items():
        # Ensure the dict structure exists if the config added new models
        if model_name not in master_results:
            master_results[model_name] = {task["label"]: [] for task in tasks}
            
        for task in tasks:
            if task["label"] not in master_results[model_name]:
                 master_results[model_name][task["label"]] = []
                 
            for trial in range(1, TRIALS_PER_TASK + 1):
                # Check if this specific trial is already in the checkpoint
                if len(master_results[model_name][task["label"]]) >= trial:
                    print(f"⏩ Skipping {model_name} on '{task['label']}' (Trial {trial}/{TRIALS_PER_TASK}) - Already completed.")
                    continue
                    
                print(f"\n🚀 Running {model_name} on '{task['label']}' (Trial {trial}/{TRIALS_PER_TASK})...")
                
                run_config = copy.deepcopy(base_config)
                if isinstance(model_path, dict):
                    run_config["run_settings"]["model_id"] = model_path["model_id"]
                    run_config["run_settings"]["lora_path"] = model_path.get("lora_path")
                else:
                    run_config["run_settings"]["model_id"] = model_path
                    run_config["run_settings"]["lora_path"] = None
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
                    print(f"❌ Error evaluating {model_name} on {task['label']}: {e}")
                    # Appends the safe fallback dictionary to prevent plotting crashes
                    master_results[model_name][task['label']].append(get_empty_metrics())
                    
                # --- BULLETPROOF ATOMIC CHECKPOINT SAVING ---
                # 1. Write to a temporary file
                fd, temp_path = tempfile.mkstemp(dir=out_path, suffix='.tmp')
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(master_results, f, indent=4)
                
                # 2. Atomically rename/replace the actual checkpoint file. 
                # If power fails exactly here, the original file is completely unharmed!
                os.replace(temp_path, checkpoint_file)

    # --- FINAL MULTI-METRIC REPORT ---
    report_text = f"\n\n{'='*120}\n"
    report_text += f"🏆 COMPREHENSIVE BENCHMARK RESULTS (Mean ± StdDev) 🏆\n"
    report_text += f"{'='*120}\n"
    
    header = f"{'Task (Constraint)':<22} | {'Metric':<16} | " + " | ".join([f"{m:<18}" for m in models.keys()])
    report_text += header + "\n"
    report_text += "-" * 120 + "\n"

    # Define which metrics to print in the table
    display_metrics = [
        ("Max Fitness", "fitness"), 
        ("AUC (Speed)", "auc"), 
        ("Validity %", "validity"), 
        ("Duplicate %", "duplicate_rate"),
        ("Time/Netlist (s)", "time_per_netlist_sec")
    ]

    for task in tasks:
        task_lbl = task['label']
        report_text += f"{task_lbl:<22} | \n"
        
        for metric_name, key in display_metrics:
            row = f"{'':<22} | {metric_name:<16} | "
            for model_name in models.keys():
                trial_data = master_results.get(model_name, {}).get(task_lbl, [])
                
                # FIX: Safely filter out None types before using numpy
                vals = [t.get(key) for t in trial_data if t and t.get(key) is not None]
                
                if vals:
                    row += f"{np.mean(vals):.2f} ± {np.std(vals):.2f}    | "
                else:
                    row += f"{'ERR':<18} | "
            report_text += row + "\n"
        report_text += "-" * 120 + "\n"

    # 1. Print to console
    print(report_text)

    # 2. Save to .txt file
    report_file = out_path / "benchmark_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"📝 Benchmark table successfully saved to: {report_file}\n")

    # Triggering Extracted Plotting Modules
    try:
        plot_combined_fitness(master_results, output_dir=str(out_path))
        plot_learning_curves(master_results, output_dir=str(out_path), n_batches=base_config["run_settings"]["n_batches"])
        plot_pareto_scatter(master_results, output_dir=str(out_path))
        plot_radar_chart(master_results, output_dir=str(out_path))
        plot_validity_bar_chart(master_results, output_dir=str(out_path))
    except NameError:
        print("Plots skipped because the visualization modules could not be loaded.")

if __name__ == "__main__":
    main()