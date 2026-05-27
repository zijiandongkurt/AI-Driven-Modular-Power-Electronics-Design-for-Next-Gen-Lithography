import copy
import os
import re
import numpy as np
from pathlib import Path
from demo_inference import run_inference
import matplotlib
matplotlib.use('Agg') # Thread-safe backend for headless execution
import matplotlib.pyplot as plt


def extract_metrics(run_folder: str) -> dict:
    """Parses the run_summary.txt to extract Fitness, normalized AUC (Convergence), Validity Rate, and Learning Curve."""
    summary_path = Path(run_folder) / "run_summary.txt"
    if not summary_path.exists():
        return {"fitness": 0.0, "auc": 0.0, "validity": 0.0, "learning_curve": []}
    
    content = summary_path.read_text(encoding="utf-8")
    
    fit_match = re.search(r"Overall Best Fitness:\s*([0-9.]+)", content)
    fitness = float(fit_match.group(1)) if fit_match else 0.0
    
    val_match = re.search(r"Validity Rate:\s*([0-9.]+)%", content)
    validity = float(val_match.group(1)) if val_match else 0.0
    
    # Grab all raw scores at every batch step
    raw_scores = [float(x) for x in re.findall(r"Batch \d+ Best DB Score:\s*([-0-9.]+)", content)]
    
    normalized_scores = []
    for score in raw_scores:
        if score < 0.5:
            normalized_scores.append(0.0)
        else:
            normalized_scores.append((score - 0.5) * 2.0)
            
    auc = np.trapz(normalized_scores) if len(normalized_scores) > 1 else 0.0
    
    # --- ADD "learning_curve" TO THE RETURN DICTIONARY ---
    return {"fitness": fitness, "auc": auc, "validity": validity, "learning_curve": raw_scores}


def plot_learning_curves(master_results: dict, output_dir: str = "experiments", n_batches: int = 15):
    """
    Plots the global learning curves (Max Fitness over Time) for each model,
    including shaded regions for the standard deviation.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    colors = ['#5dade2', '#f5b041', '#58d68d'] # Blue, Orange, Green
    
    for idx, model in enumerate(master_results.keys()):
        all_curves = []
        
        # 1. Gather every learning curve for this model
        for task_label, trials in master_results[model].items():
            for trial in trials:
                if trial and "learning_curve" in trial and len(trial["learning_curve"]) > 0:
                    curve = trial["learning_curve"]
                    
                    # Pad the curve if it crashed early, so numpy math still works
                    if len(curve) < n_batches:
                        curve = curve + [curve[-1]] * (n_batches - len(curve))
                        
                    # Truncate just in case
                    all_curves.append(curve[:n_batches])
                else:
                    all_curves.append([0.0] * n_batches)
        
        # 2. Calculate Mean and StdDev across all runs
        if all_curves:
            curve_array = np.array(all_curves)
            mean_curve = np.mean(curve_array, axis=0)
            std_curve = np.std(curve_array, axis=0)
            
            x_batches = np.arange(1, n_batches + 1)
            
            # 3. Plot the Line and Shaded Variance Area
            plt.plot(x_batches, mean_curve, label=model, color=colors[idx % len(colors)], linewidth=2.5)
            
            # Clip the shading so it doesn't visually bleed past 1.1 or -1.0
            lower_bound = np.clip(mean_curve - std_curve, -1.0, 1.1)
            upper_bound = np.clip(mean_curve + std_curve, -1.0, 1.1)
            plt.fill_between(x_batches, lower_bound, upper_bound, alpha=0.15, color=colors[idx % len(colors)])

    # 4. Formatting
    plt.title('Global Learning Curves per Model\n(Mean Max Fitness over Batches ± StdDev)', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12, fontweight='bold')
    plt.ylabel('Global Max Fitness Score', fontsize=12, fontweight='bold')
    
    plt.xlim(1, n_batches)
    plt.ylim(0, 1.1)
    plt.xticks(np.arange(1, n_batches + 1))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    
    # 5. Save to disk
    filepath = Path(output_dir) / "combined_learning_curves.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Global learning curves plot successfully saved to: {filepath}")

def plot_combined_fitness(master_results: dict, output_dir: str = "experiments"):
    """
    Aggregates all fitness scores across all tasks and trials for each model,
    and plots a standard boxplot showing the distribution of max fitness.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = list(master_results.keys())
    data_to_plot = []
    
    # 1. Aggregate the raw data per model
    for model in models:
        all_fitness_scores = []
        for task_label, trials in master_results[model].items():
            for trial in trials:
                # Append the fitness, defaulting to 0.0 if the trial errored out
                score = trial.get("fitness", 0.0) if trial else 0.0
                all_fitness_scores.append(score)
        data_to_plot.append(all_fitness_scores)
        
    # 2. Generate the Boxplot
    plt.figure(figsize=(10, 7))
    
    # Define distinct colors for Base, SFT, and RL
    colors = ['#5dade2', '#f5b041', '#58d68d'] 
    
    # Create the boxplot
    bplot = plt.boxplot(
        data_to_plot,
        patch_artist=True,  # allows us to fill the boxes with color
        labels=models,      # x-axis labels
        medianprops=dict(color='black', linewidth=2.5),
        boxprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.6) # Highlight outliers
    )
    
    # Apply colors to the boxes
    for patch, color in zip(bplot['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    # 3. Formatting
    plt.ylabel('Global Max Fitness Score', fontsize=12, fontweight='bold')
    plt.xlabel('LLM Model', fontsize=12, fontweight='bold')
    plt.title('Combined Model Capability Benchmark\n(Fitness Distribution across all Constraints)', fontsize=14, fontweight='bold')
    
    # Bound the Y-axis from 0 to 1.05 (Fitness maxes at 1.0, give it a tiny bit of breathing room)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 4. Save to disk
    filepath = Path(output_dir) / "combined_max_fitness_benchmark.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Global boxplot successfully saved to: {filepath}")

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

    # Initialize nested dictionary for 3 metrics per task
    master_results = {model: {task["label"]: [] for task in tasks} for model in models.keys()}

    for model_name, model_path in models.items():
        for task in tasks:
            for trial in range(1, TRIALS_PER_TASK + 1):
                run_config = copy.deepcopy(base_config)
                run_config["run_settings"]["model_id"] = model_path
                run_config["run_settings"]["run_prefix"] = f"Bench_{model_name}_{task['label']}_T{trial}"
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
    print(f"\n\n{'='*100}")
    print(f"🏆 COMPREHENSIVE BENCHMARK RESULTS (Mean ± StdDev) 🏆")
    print(f"{'='*100}")
    
    header = f"{'Task (Constraint)':<22} | {'Metric':<14} | " + " | ".join([f"{m:<18}" for m in models.keys()])
    print(header)
    print("-" * 100)

    for task in tasks:
        task_lbl = task['label']
        print(f"{task_lbl:<22} | ")
        
        for metric_name, key in [("Max Fitness", "fitness"), ("AUC (Speed)", "auc"), ("Validity %", "validity")]:
            row = f"{'':<22} | {metric_name:<14} | "
            for model_name in models.keys():
                trial_data = master_results[model_name][task_lbl]
                vals = [t[key] for t in trial_data if t]
                
                if vals:
                    row += f"{np.mean(vals):.2f} ± {np.std(vals):.2f}    | "
                else:
                    row += f"{'ERR':<18} | "
            print(row)
        print("-" * 100)

    plot_combined_fitness(master_results, output_dir="experiments")
    plot_learning_curves(master_results, output_dir="experiments", n_batches=base_config["run_settings"]["n_batches"])

if __name__ == "__main__":
    main()