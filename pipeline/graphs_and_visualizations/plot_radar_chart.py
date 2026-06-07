import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_radar_chart(master_results: dict, output_dir: str = "experiments"):
    """Plot a 4-axis radar chart visualizing physical trade-offs per model.

    The four axes are voltage accuracy, power efficiency, size compactness,
    and simplicity (low component count).

    Args:
        master_results (dict): Nested dict of shape
            ``{model: {task_label: [trial_dicts]}}``.  Each trial dict should
            have ``"v_error_pct"``, ``"efficiency"``, ``"volume"``,
            ``"target_power"``, and ``"components"`` keys.
        output_dir (str): Directory where the PNG file is saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    labels = ['Voltage Accuracy', 'Power Efficiency', 'Size Compactness', 'Simplicity (Low Parts)']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {'Base': '#5dade2', 'SFT': '#f5b041', 'RL': '#58d68d'}
    
    for model_name, tasks in master_results.items():
        model_scores = {'v_acc': [], 'eff': [], 'size': [], 'simp': []}
        
        for task_label, trials in tasks.items():
            if task_label.startswith("P1"): max_comp = 10.0
            elif task_label.startswith("P2"): max_comp = 15.0
            elif task_label.startswith("P3"): max_comp = 25.0
            else: max_comp = 15.0 

            for trial in trials:
                if trial and trial.get("v_error_pct") is not None:
                    v_acc = max(0.0, 100.0 - (trial["v_error_pct"] * 2)) 
                    eff = trial.get("efficiency", 0.0)
                    vol = trial.get("volume", 1.0) 
                    power = trial.get("target_power", 10.0)
                    
                    if vol <= 0.01:
                        size = 100.0
                    else:
                        power_density = power / vol
                        size = min(100.0, max(0.0, (power_density / 2.0) * 100.0))
                    
                    comp = trial.get("components", 10)
                    simp = max(0.0, 100.0 - (comp * (100.0 / max_comp))) # Adjusted for dynamic scaling
                    
                    model_scores['v_acc'].append(v_acc)
                    model_scores['eff'].append(eff)
                    model_scores['size'].append(size)
                    model_scores['simp'].append(simp)
                    
        if model_scores['v_acc']:
            avg_values = [
                np.mean(model_scores['v_acc']),
                np.mean(model_scores['eff']),
                np.mean(model_scores['size']),
                np.mean(model_scores['simp'])
            ]
            avg_values += avg_values[:1] 
            ax.plot(angles, avg_values, color=colors.get(model_name, 'gray'), linewidth=2.5, label=model_name)
            ax.fill(angles, avg_values, color=colors.get(model_name, 'gray'), alpha=0.15)

    ax.set_theta_offset(np.pi / 2) 
    ax.set_theta_direction(-1)     
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color="grey", size=10)
    plt.title('Global Physical Trade-Off Profile\n(100 = Perfect Score)', size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    filepath = Path(output_dir) / "combined_radar_profile.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Radar plot successfully saved to: {filepath}")