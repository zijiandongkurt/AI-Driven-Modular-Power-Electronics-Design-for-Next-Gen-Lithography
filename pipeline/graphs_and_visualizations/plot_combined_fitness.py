import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_combined_fitness(master_results: dict, output_dir: str = "experiments"):
    """Aggregate fitness scores and plot a combined model capability boxplot.

    Aggregates all fitness scores across all tasks and trials for each model
    and plots a standard boxplot showing the distribution of max fitness.

    Args:
        master_results (dict): Nested dict of shape
            ``{model: {task_label: [trial_dicts]}}``.
        output_dir (str): Directory where the PNG file is saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models = list(master_results.keys())
    data_to_plot = []
    
    for model in models:
        all_fitness_scores = []
        for task_label, trials in master_results[model].items():
            for trial in trials:
                score = trial.get("fitness", 0.0) if trial else 0.0
                all_fitness_scores.append(score)
        data_to_plot.append(all_fitness_scores)
        
    plt.figure(figsize=(10, 7))
    colors = ['#5dade2', '#f5b041', '#58d68d'] 
    
    bplot = plt.boxplot(
        data_to_plot, patch_artist=True, labels=models,
        medianprops=dict(color='black', linewidth=2.5),
        boxprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.6) 
    )
    
    for patch, color in zip(bplot['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    plt.ylabel('Global Max Fitness Score', fontsize=12, fontweight='bold')
    plt.xlabel('LLM Model', fontsize=12, fontweight='bold')
    plt.title('Combined Model Capability Benchmark\n(Fitness Distribution across all Constraints)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filepath = Path(output_dir) / "combined_max_fitness_benchmark.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Global boxplot successfully saved to: {filepath}")