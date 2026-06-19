import os
import glob
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
    'figure.titlesize': 20   
})

def extract_run_number(folder_name):
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def extract_champion_data():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')

    sessions = ["zycos_008", "zycos_009", "zycos_010"]
    
    # Structure: champ_data[session_name][run_number] = "tran_args"
    champ_data = {s: {} for s in sessions}
    
    print("🔍 Scanning for best_topology.net files...")

    for session in sessions:
        zycos_dir = os.path.join(DATA_DIR, session)
        if not os.path.exists(zycos_dir):
            continue
            
        run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
        
        for run_folder in run_folders:
            run_idx = extract_run_number(os.path.basename(run_folder))
            champ_file = os.path.join(run_folder, 'results', 'best_topology.net')
            
            if not os.path.exists(champ_file):
                print(f"  ⚠️ Missing {session}/Run_{run_idx:03d}/results/best_topology.net")
                continue
                
            try:
                with open(champ_file, 'r', encoding='utf-8', errors='ignore') as f:
                    found_tran = False
                    for line in f:
                        clean_line = line.strip().lower()
                        if re.match(r'^\s*\.tran', clean_line):
                            parts = clean_line.split(maxsplit=1)
                            if len(parts) > 1:
                                args = " ".join(parts[1].split())
                                champ_data[session][run_idx] = args
                                found_tran = True
                            break
                    
                    if not found_tran:
                        champ_data[session][run_idx] = "MISSING_TRAN"
                        
            except Exception as e:
                print(f"  ❌ Error reading {champ_file}: {e}")
                
    return champ_data, DATA_DIR

def plot_champion_frequencies(champ_data, data_dir):
    print("\n📊 Generating Champion Frequency Plots...")
    
    sessions = list(champ_data.keys())
    colors = {"zycos_008": "royalblue", "zycos_009": "darkorange", "zycos_010": "seagreen"}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=False)
    fig.suptitle('Frequency of `.tran` Commands Used by Top Champions', fontweight='bold', y=1.02)

    for idx, session in enumerate(sessions):
        ax = axs[idx]
        
        # Count frequencies for this session
        counts = Counter(champ_data[session].values())
        
        if not counts:
            ax.text(0.5, 0.5, "No Champion data found.", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{session}", fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Sort by frequency
        top_items = counts.most_common()
        labels = [item[0] for item in top_items][::-1]
        values = [item[1] for item in top_items][::-1]
        
        bars = ax.barh(labels, values, color=colors.get(session, 'steelblue'), edgecolor='black', alpha=0.8)
        
        ax.set_title(f"{session}", fontweight='bold')
        ax.set_xlabel("Number of Champions")
        if idx == 0:
            ax.set_ylabel("`.tran` Arguments")
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')

        # Add integer labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=11)
                    
        ax.set_xlim(0, max(values) * 1.2 if values else 1)
        ax.set_xticks(range(0, max(values) + 2 if values else 2)) # Force integer X-ticks

    plt.tight_layout()
    out_path = os.path.join(data_dir, "champion_tran_frequencies.png")
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {out_path}")
    except PermissionError:
        print(f"  ❌ Close {os.path.basename(out_path)} and try again.")
    plt.close()

def plot_champion_evolution(champ_data, data_dir):
    print("📈 Generating Champion Evolution Timeline...")
    
    sessions = list(champ_data.keys())
    colors = {"zycos_008": "royalblue", "zycos_009": "darkorange", "zycos_010": "seagreen"}
    
    # 1. Collect ALL unique .tran values across all sessions to create a unified Y-axis
    all_trans = set()
    for s_data in champ_data.values():
        all_trans.update(s_data.values())
        
    # Sort them alphabetically (or could try to sort by numerical time, but string sort is safer for raw LLM output)
    unique_trans = sorted(list(all_trans))
    tran_to_y = {tran: i for i, tran in enumerate(unique_trans)}
    
    if not unique_trans:
        print("  ⚠️ No data to plot for evolution.")
        return

    # 2. Build the Plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Evolution of Champion `.tran` Configurations Across RL Runs', fontweight='bold', y=0.96)

    for idx, session in enumerate(sessions):
        ax = axs[idx]
        run_data = champ_data[session]
        
        if not run_data:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue
            
        runs = sorted(list(run_data.keys()))
        y_vals = [tran_to_y[run_data[r]] for r in runs]
        
        # Plot the timeline line
        ax.plot(runs, y_vals, marker='o', linestyle='-', linewidth=2, markersize=8, 
                color=colors.get(session, 'black'), alpha=0.85, label=f"Champion config")
        
        # Formatting
        ax.set_title(f"{session}", fontweight='bold')
        ax.set_yticks(range(len(unique_trans)))
        ax.set_yticklabels(unique_trans)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Highlight MISSING_TRAN in red if it exists
        if "MISSING_TRAN" in unique_trans:
            missing_idx = tran_to_y["MISSING_TRAN"]
            ax.axhline(y=missing_idx, color='crimson', linestyle=':', alpha=0.5)

    axs[-1].set_xlabel("Run Number")
    axs[-1].set_xticks(range(1, 11))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    out_path = os.path.join(data_dir, "champion_tran_evolution.png")
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {out_path}")
    except PermissionError:
        print(f"  ❌ Close {os.path.basename(out_path)} and try again.")
    plt.close()

if __name__ == "__main__":
    extracted_data, data_directory = extract_champion_data()
    if extracted_data:
        plot_champion_frequencies(extracted_data, data_directory)
        plot_champion_evolution(extracted_data, data_directory)
        print("\n🎉 Analysis Complete!")