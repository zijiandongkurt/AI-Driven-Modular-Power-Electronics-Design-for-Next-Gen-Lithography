import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pareto_scatter(master_results: dict, output_dir: str = "experiments"):
    """
    Plots a Trade-Off Scatter plot (Voltage Error vs Efficiency) for all champion circuits.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    colors = {'Base': '#5dade2', 'SFT': '#f5b041', 'RL': '#58d68d'}
    markers = {'Base': 'o', 'SFT': 's', 'RL': '^'}
    plotted_labels = set()
    failed_labels = set()
    
    for model_name, tasks in master_results.items():
        for task_label, trials in tasks.items():
            for trial in trials:
                if trial and trial.get("v_error_pct") is not None and trial.get("efficiency") is not None:
                    v_err = trial["v_error_pct"]
                    eff = trial["efficiency"]
                    v_err_clipped = min(v_err, 50.0)
                    label = model_name if model_name not in plotted_labels else ""
                    plotted_labels.add(model_name)
                    
                    plt.scatter(
                        v_err_clipped, eff, color=colors.get(model_name, 'gray'), 
                        marker=markers.get(model_name, 'o'), s=100, alpha=0.7, 
                        edgecolor='black', label=label
                    )
                else:
                    label = f"{model_name} (Failed)" if model_name not in failed_labels else ""
                    failed_labels.add(model_name)
                    plt.scatter(
                        50.0, 0.0, color=colors.get(model_name, 'gray'), 
                        marker='X', s=180, alpha=0.8, edgecolor='black', label=label
                    )

    plt.gca().invert_xaxis()
    plt.title('Physical Trade-Off Profile (Pareto Frontier)\nAll Champion Circuits Across All Constraints', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Voltage Error (%) ⟵ WORSE | BETTER ⟶', fontsize=12, fontweight='bold')
    plt.ylabel('Power Efficiency (%)', fontsize=12, fontweight='bold')
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    plt.text(1, 98, 'The "Perfect" Zone', color='red', fontsize=10, fontweight='bold')
    plt.text(49, 3, 'Failure Graveyard', color='black', fontsize=10, fontweight='bold', style='italic')
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title="LLM Model", fontsize=11, title_fontsize=12, loc='lower left')
    
    filepath = Path(output_dir) / "combined_pareto_tradeoffs.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎯 Pareto trade-off scatter plot successfully saved to: {filepath}")