import os
import glob
import json
import re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tailored font settings for readability
plt.rcParams.update({
    'axes.titlesize': 16,    
    'axes.labelsize': 13,    
    'xtick.labelsize': 11,   
    'ytick.labelsize': 11,   
    'legend.fontsize': 11,
    'figure.titlesize': 20   
})

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

def extract_benchmark_champion_data():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'benchmark_config.json')
    config = load_json(config_path)
    if not config: return None, None, None

    models = ["Base", "SFT", "Zycos10"]
    tasks = [t['label'] for t in config.get('tasks', [])]
    
    # Store frequencies per model
    champ_counts = {m: Counter() for m in models}
    
    # Store specific choices per model per task for the scatter plot
    data_scatter = {m: {t: [] for t in tasks} for m in models}
    
    print("🔍 Extracting `.tran` commands from Benchmark Champions...")

    for model in models:
        run_folders = glob.glob(os.path.join(DATA_DIR, f"Bench_{model}_*_001"))
        for run_folder in run_folders:
            folder_name = os.path.basename(run_folder)
            
            # Extract Task and Trial from folder name
            match_name = re.search(f"Bench_{model}_(.+)_T(\d+)_001", folder_name)
            if not match_name: continue
            task_label = match_name.group(1)
            
            # UPDATED: Now looks inside the "database" subfolder
            champ_json = os.path.join(run_folder, "database", "champion_metrics.json")
            cdata = load_json(champ_json)
            if not cdata: continue
                
            cid = cdata.get('id', '')
            if not cid: continue

            # Extract the specific candidate and batch from the ID to find the exact .net file
            cand_match = re.search(r'_(g\d+_cand\d+)_b(\d+)$', cid)
            net_file = None
            
            if cand_match:
                cand_name = cand_match.group(1)
                batch_num = cand_match.group(2)
                search_path = os.path.join(run_folder, f"batch_{batch_num}", "LLM_output", f"*{cand_name}*.net")
                files = glob.glob(search_path)
                if files: net_file = files[0]

            # Fallback search if strict regex misses it
            if not net_file:
                alt_match = re.search(r'(g\d+_cand\d+)', cid)
                if alt_match:
                    all_nets = glob.glob(os.path.join(run_folder, "**", f"*{alt_match.group(1)}*.net"), recursive=True)
                    if all_nets: net_file = all_nets[0]

            tran_str = "NO_TRAN_COMMAND"
            
            if not net_file:
                tran_str = "MISSING_FILE"
            else:
                try:
                    with open(net_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            clean = line.strip().lower()
                            if re.match(r'^\s*\.tran', clean):
                                parts = clean.split(maxsplit=1)
                                if len(parts) > 1:
                                    tran_str = " ".join(parts[1].split())
                                break
                except: pass

            champ_counts[model][tran_str] += 1
            if task_label in data_scatter[model]:
                data_scatter[model][task_label].append(tran_str)

    return champ_counts, data_scatter, tasks, DATA_DIR

def plot_champion_frequencies(champ_counts, data_dir):
    print("📊 Generating Champion Frequency Plots...")
    models = ["Base", "SFT", "Zycos10"]
    colors = {"Base": "gray", "SFT": "darkorange", "Zycos10": "seagreen"}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=False)
    fig.suptitle('Frequency of `.tran` Commands Used by Benchmark Champions', fontweight='bold', y=1.02)

    for idx, model in enumerate(models):
        ax = axs[idx]
        counts = champ_counts[model]
        
        if not counts:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        top_items = counts.most_common()
        labels = [item[0] for item in top_items][::-1]
        values = [item[1] for item in top_items][::-1]
        
        bars = ax.barh(labels, values, color=colors.get(model, 'steelblue'), edgecolor='black', alpha=0.8)
        
        ax.set_title(f"{model} Champions", fontweight='bold')
        ax.set_xlabel("Number of Champions")
        if idx == 0: ax.set_ylabel("`.tran` Arguments")
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=11)
                    
        ax.set_xlim(0, max(values) * 1.2 if values else 1)
        ax.set_xticks(range(0, int(max(values)) + 2 if values else 2))

    plt.tight_layout()
    out_path = os.path.join(data_dir, "benchmark_champion_tran_frequencies.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {os.path.basename(out_path)}")

def plot_task_distribution(data_scatter, tasks, data_dir):
    print("📈 Generating Champion Task Distribution Scatter...")
    models = ["Base", "SFT", "Zycos10"]
    styles = {
        "Base":    {"color": "gray", "marker": "o", "offset": -0.2},
        "SFT":     {"color": "darkorange", "marker": "s", "offset": 0.0},
        "Zycos10": {"color": "seagreen", "marker": "^", "offset": 0.2}
    }
    
    # 1. Collect unified Y-axis
    all_trans = set()
    for m in models:
        for t in tasks:
            all_trans.update(data_scatter[m][t])
            
    # Sort strings normally, but push "NO_TRAN_COMMAND" to the very bottom
    unique_trans = sorted([t for t in all_trans if t not in ("NO_TRAN_COMMAND", "MISSING_FILE")])
    if "NO_TRAN_COMMAND" in all_trans: unique_trans.insert(0, "NO_TRAN_COMMAND")
    if "MISSING_FILE" in all_trans: unique_trans.insert(0, "MISSING_FILE")
    
    tran_to_y = {tran: i for i, tran in enumerate(unique_trans)}
    x_positions = np.arange(len(tasks))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('Benchmark Champions: Chosen `.tran` Commands per Task', fontweight='bold', y=0.96)

    # Shaded Backgrounds for Tasks
    session_boundaries = [("Easy", 0, 1), ("Medium", 2, 3), ("Hard", 4, 5)]
    for idx, (tier_name, start, end) in enumerate(session_boundaries):
        bg_alpha = 0.03 if idx % 2 == 0 else 0.08
        ax.axvspan(start - 0.5, end + 0.5, color='gray', alpha=bg_alpha, zorder=0)
        if end < len(tasks) - 1:
            ax.axvline(x=end + 0.5, color='black', linestyle=':', alpha=0.4, zorder=1)
        mid_rel = ((start + end) / 2.0) / (len(tasks) - 1)
        ax.text(mid_rel, 1.02, tier_name, transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))

    # Plot Scatter Points
    for model in models:
        st = styles[model]
        for t_idx, task in enumerate(tasks):
            trans_used = data_scatter[model].get(task, [])
            for tran in trans_used:
                x_val = x_positions[t_idx] + st['offset']
                y_val = tran_to_y[tran]
                # Label only once per model for legend
                lbl = model if (t_idx == 0 and tran == trans_used[0]) else ""
                ax.scatter(x_val, y_val, color=st['color'], marker=st['marker'], s=90, alpha=0.85, zorder=3, label=lbl)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels([t.replace('_', '\n') for t in tasks], fontsize=10)
    ax.set_yticks(range(len(unique_trans)))
    ax.set_yticklabels(unique_trans)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
    ax.set_ylabel("`.tran` Argument")
    
    # Highlight error states with a red line
    for err_state in ["NO_TRAN_COMMAND", "MISSING_FILE"]:
        if err_state in unique_trans:
            ax.axhline(y=tran_to_y[err_state], color='crimson', linestyle=':', alpha=0.5, zorder=1)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_path = os.path.join(data_dir, "benchmark_champion_tran_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {os.path.basename(out_path)}")

if __name__ == "__main__":
    counts, scatter, tasks, directory = extract_benchmark_champion_data()
    if counts:
        plot_champion_frequencies(counts, directory)
        plot_task_distribution(scatter, tasks, directory)
        print("\n🎉 Benchmark Champion Analysis Complete!")