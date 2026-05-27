import copy
import os
import re
import numpy as np
from pathlib import Path
from demo_inference import run_inference
import matplotlib
matplotlib.use('Agg') # Thread-safe backend for headless execution
import matplotlib.pyplot as plt
import json


def extract_metrics(run_folder: str) -> dict:
    """Parses the run summary and champion JSON to extract fitness and physical metrics."""
    summary_path = Path(run_folder) / "run_summary.txt"
    champ_path = Path(run_folder) / "champion_metrics.json"
    
    # 1. Added "volume" and "components" to the default dictionary
    metrics = {
        "fitness": 0.0, "auc": 0.0, "validity": 0.0, "learning_curve": [], 
        "v_error_pct": None, "efficiency": None, 
        "volume": None, "components": None, "target_power": None
    }
    
    # 2. Extract Learning & Fitness Data
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

    # 3. Extract Physical Trade-off Data
    if champ_path.exists():
        with open(champ_path, "r", encoding="utf-8") as f:
            champ = json.load(f)
            target_v = champ.get("target_voltage", 1.0) 
            raw_v = champ.get("raw_voltage", 0.0)
            
            error_pct = abs(raw_v - target_v) / target_v * 100.0
            
            metrics["v_error_pct"] = error_pct
            metrics["efficiency"] = champ.get("raw_efficiency", 0.0) * 100.0 
            
            # ---> GRAB THE NEW HARDWARE METRICS <---
            metrics["volume"] = champ.get("raw_volume", 0.0)
            metrics["components"] = champ.get("raw_components", 0)
            metrics["target_power"] = champ.get("target_power", 10.0)
            
    return metrics

import math # Make sure this is imported at the top!

def plot_radar_chart(master_results: dict, output_dir: str = "experiments"):
    """
    Plots a 4-axis Radar Chart (Spider Plot) to visualize the physical
    trade-offs (The "Shape" of the Engineer) for each model.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    labels = ['Voltage Accuracy', 'Power Efficiency', 'Size Compactness', 'Simplicity (Low Parts)']
    num_vars = len(labels)
    
    # Compute angles for each axis in the polar plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {'Base': '#5dade2', 'SFT': '#f5b041', 'RL': '#58d68d'}
    
    for model_name, tasks in master_results.items():
        model_scores = {'v_acc': [], 'eff': [], 'size': [], 'simp': []}
        
        # 1. Aggregate and Normalize Data per Model
        for task_label, trials in tasks.items():

            # ---> NEW: Dynamic Simplicity Scaling by Phase <---
            if task_label.startswith("P1"):
                max_comp = 10.0  # Easy circuits should be very lean
            elif task_label.startswith("P2"):
                max_comp = 15.0  # Medium circuits get some breathing room
            elif task_label.startswith("P3"):
                max_comp = 25.0  # Industrial grid circuits need many parts
            else:
                max_comp = 15.0  # Fallback

            for trial in trials:
                if trial and trial.get("v_error_pct") is not None:
                    
                    # Voltage Accuracy: 0% error = 100 score. 50%+ error = 0 score.
                    v_acc = max(0.0, 100.0 - (trial["v_error_pct"] * 2)) 
                    
                    # Power Efficiency: Already 0-100
                    eff = trial.get("efficiency", 0.0)
                    
                    # Size Compactness: Scored on Power Density (W/cm^3)
                    vol = trial.get("volume", 1.0) # default 1.0 to avoid zero division
                    power = trial.get("target_power", 10.0)
                    
                    if vol <= 0.01: # Impossible physics / Perfect score
                        size = 100.0
                    else:
                        power_density = power / vol
                        
                        # Assume 2.0 W/cm^3 is a perfect 100 score. 
                        # Anything above that caps at 100. Anything below scales down to 0.
                        size = min(100.0, max(0.0, (power_density / 2.0) * 100.0))
                    
                    # Simplicity: Assume 10 components is bad (0), 0 is perfect (100)
                    comp = trial.get("components", 10)
                    simp = max(0.0, 100.0 - (comp * 10.0))
                    
                    model_scores['v_acc'].append(v_acc)
                    model_scores['eff'].append(eff)
                    model_scores['size'].append(size)
                    model_scores['simp'].append(simp)
                    
        # 2. Average the scores and map them to the polygon
        if model_scores['v_acc']:
            avg_values = [
                np.mean(model_scores['v_acc']),
                np.mean(model_scores['eff']),
                np.mean(model_scores['size']),
                np.mean(model_scores['simp'])
            ]
            avg_values += avg_values[:1] # Close the loop
            
            # 3. Plot the boundary line and fill the area
            ax.plot(angles, avg_values, color=colors.get(model_name, 'gray'), linewidth=2.5, label=model_name)
            ax.fill(angles, avg_values, color=colors.get(model_name, 'gray'), alpha=0.15)

    # 4. Formatting the Spider Web
    ax.set_theta_offset(np.pi / 2) # Put the first axis at the very top
    ax.set_theta_direction(-1)     # Draw clockwise
    
    # Set the labels at the edges
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
    
    # Format the concentric circles (0 to 100 scale)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color="grey", size=10)
    
    plt.title('Global Physical Trade-Off Profile\n(100 = Perfect Score)', size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    # 5. Save to disk
    filepath = Path(output_dir) / "combined_radar_profile.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🕸️ Radar (Spider) plot successfully saved to: {filepath}")


def plot_pareto_scatter(master_results: dict, output_dir: str = "experiments"):
    """
    Plots a Trade-Off Scatter plot (Voltage Error vs Efficiency) for all champion circuits.
    Reveals the physical Pareto frontier of each model's generation logic, 
    and exposes catastrophic failures in the bottom-left "graveyard".
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    colors = {'Base': '#5dade2', 'SFT': '#f5b041', 'RL': '#58d68d'}
    markers = {'Base': 'o', 'SFT': 's', 'RL': '^'}
    
    # To avoid duplicate legend labels
    plotted_labels = set()
    failed_labels = set()
    
    for model_name, tasks in master_results.items():
        for task_label, trials in tasks.items():
            for trial in trials:
                # 1. SUCCESSFUL RUNS
                if trial and trial.get("v_error_pct") is not None and trial.get("efficiency") is not None:
                    
                    v_err = trial["v_error_pct"]
                    eff = trial["efficiency"]
                    
                    # Cap extreme voltage errors at 50% just to keep the plot readable
                    v_err_clipped = min(v_err, 50.0)
                    
                    label = model_name if model_name not in plotted_labels else ""
                    plotted_labels.add(model_name)
                    
                    plt.scatter(
                        v_err_clipped, 
                        eff, 
                        color=colors.get(model_name, 'gray'), 
                        marker=markers.get(model_name, 'o'),
                        s=100, 
                        alpha=0.7, 
                        edgecolor='black',
                        label=label
                    )
                # 2. CATASTROPHIC FAILURES (The Graveyard)
                else:
                    label = f"{model_name} (Failed)" if model_name not in failed_labels else ""
                    failed_labels.add(model_name)
                    
                    plt.scatter(
                        50.0, # Worst voltage error boundary
                        0.0,  # Zero efficiency
                        color=colors.get(model_name, 'gray'), 
                        marker='X', # Big cross to indicate failure
                        s=180,      # Make it slightly larger so it stands out
                        alpha=0.8, 
                        edgecolor='black',
                        label=label
                    )

    # Formatting: Invert X-axis so "better" (0% error) is on the right side!
    plt.gca().invert_xaxis()
    
    plt.title('Physical Trade-Off Profile (Pareto Frontier)\nAll Champion Circuits Across All Constraints', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Voltage Error (%) ⟵ WORSE | BETTER ⟶', fontsize=12, fontweight='bold')
    plt.ylabel('Power Efficiency (%)', fontsize=12, fontweight='bold')
    
    # Add a crosshair at the "Perfect Circuit" zone (0% error, 100% efficiency)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    plt.text(1, 98, 'The "Perfect" Zone', color='red', fontsize=10, fontweight='bold')
    
    # Add a label for the Graveyard
    plt.text(49, 3, 'Failure Graveyard', color='black', fontsize=10, fontweight='bold', style='italic')
    
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Organize legend nicely
    plt.legend(title="LLM Model", fontsize=11, title_fontsize=12, loc='lower left')
    
    filepath = Path(output_dir) / "combined_pareto_tradeoffs.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎯 Pareto trade-off scatter plot successfully saved to: {filepath}")

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
    plot_pareto_scatter(master_results, output_dir="experiments")
    plot_radar_chart(master_results, output_dir="experiments")

if __name__ == "__main__":
    main()