import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves(master_results: dict, output_dir: str = "experiments", n_batches: int = 15):
    """Plot global learning curves (max fitness over time) for each model.

    Includes shaded regions for the standard deviation across trials.

    Args:
        master_results (dict): Nested dict of shape
            ``{model: {task_label: [trial_dicts]}}``.  Each trial dict should
            have a ``"learning_curve"`` key with a list of per-batch max
            fitness values.
        output_dir (str): Directory where the PNG file is saved.
        n_batches (int): Number of batches to plot on the x-axis.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    colors = ['#5dade2', '#f5b041', '#58d68d'] 
    
    for idx, model in enumerate(master_results.keys()):
        all_curves = []
        for task_label, trials in master_results[model].items():
            for trial in trials:
                if trial and "learning_curve" in trial and len(trial["learning_curve"]) > 0:
                    curve = trial["learning_curve"]
                    if len(curve) < n_batches:
                        curve = curve + [curve[-1]] * (n_batches - len(curve))
                    all_curves.append(curve[:n_batches])
                else:
                    all_curves.append([0.0] * n_batches)
        
        if all_curves:
            curve_array = np.array(all_curves)
            mean_curve = np.mean(curve_array, axis=0)
            std_curve = np.std(curve_array, axis=0)
            x_batches = np.arange(1, n_batches + 1)
            
            plt.plot(x_batches, mean_curve, label=model, color=colors[idx % len(colors)], linewidth=2.5)
            lower_bound = np.clip(mean_curve - std_curve, -1.0, 1.1)
            upper_bound = np.clip(mean_curve + std_curve, -1.0, 1.1)
            plt.fill_between(x_batches, lower_bound, upper_bound, alpha=0.15, color=colors[idx % len(colors)])

    plt.title('Global Learning Curves per Model\n(Mean Max Fitness over Batches ± StdDev)', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12, fontweight='bold')
    plt.ylabel('Global Max Fitness Score', fontsize=12, fontweight='bold')
    plt.xlim(1, n_batches)
    plt.ylim(0, 1.1)
    plt.xticks(np.arange(1, n_batches + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    
    filepath = Path(output_dir) / "combined_learning_curves.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Global learning curves plot successfully saved to: {filepath}")