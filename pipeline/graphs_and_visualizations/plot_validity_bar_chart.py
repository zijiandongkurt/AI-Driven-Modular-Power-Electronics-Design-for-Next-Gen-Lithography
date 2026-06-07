import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_validity_bar_chart(master_results: dict, output_dir: str = "experiments"):
    """Plot a grouped bar chart of mean validity rates across all tasks per model.

    Includes standard deviation error bars for each bar.

    Args:
        master_results (dict): Nested dict of shape
            ``{model: {task_label: [trial_dicts]}}``.  Each trial dict should
            have a ``"validity"`` key (float, percentage).
        output_dir (str): Directory where the PNG file is saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = list(master_results.keys())
    tasks = list(master_results[models[0]].keys())
    
    x = np.arange(len(tasks))  
    width = 0.25               
    multiplier = 0
    
    colors = ['#5dade2', '#f5b041', '#58d68d']
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for idx, model in enumerate(models):
        means = []
        stds = []
        
        for task in tasks:
            trials = master_results[model][task]
            validities = [t.get("validity", 0.0) for t in trials if t and "validity" in t]
            
            if validities:
                means.append(np.mean(validities))
                stds.append(np.std(validities))
            else:
                means.append(0.0)
                stds.append(0.0)
                
        offset = width * multiplier
        
        rects = ax.bar(
            x + offset, means, width, 
            label=model, 
            color=colors[idx % len(colors)], 
            yerr=stds, 
            capsize=4, 
            edgecolor='black', 
            alpha=0.85,
            error_kw=dict(lw=1.5, capthick=1.5, alpha=0.7)
        )
        multiplier += 1
        
    ax.set_ylabel('Mean Validity Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Benchmark Constraint (Task)', fontsize=12, fontweight='bold')
    ax.set_title('Global Syntax & Topology Validity Rates\n(Percentage of generated netlists that compiled successfully)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(title='LLM Model', fontsize=11, loc='upper right')
    
    plt.tight_layout()
    filepath = Path(output_dir) / "combined_validity_rates.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f" Global validity chart successfully saved to: {filepath}")