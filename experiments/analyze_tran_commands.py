import os
import glob
import re
from collections import Counter
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

def get_tran_counts(base_dirs):
    """Recursively scans directories and strictly counts .tran variations."""
    counts = Counter()
    files_processed = 0
    files_with_tran = 0
    
    for run_folder in base_dirs:
        for root, _, files in os.walk(run_folder):
            for file in files:
                # Strictly look for netlist extensions
                if file.endswith('.net') or file.endswith('.sp') or file.endswith('.txt'):
                    if file.endswith('.txt') and 'raw_output' not in file.lower() and 'prompt' not in file.lower():
                        continue

                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            files_processed += 1
                            for line in f:
                                clean_line = line.strip().lower()
                                # Use regex to catch .tran even if there's weird spacing
                                if re.match(r'^\s*\.tran', clean_line):
                                    parts = clean_line.split(maxsplit=1)
                                    if len(parts) > 1:
                                        args = " ".join(parts[1].split())
                                        counts[args] += 1
                                        files_with_tran += 1
                                    break
                    except Exception as e:
                        print(f"    ⚠️ Error reading {file}: {e}")
                        
    return counts, files_processed, files_with_tran

def plot_frequencies(counts_dict, title, out_path, colors):
    """Generates the horizontal bar plots safely."""
    try:
        keys = list(counts_dict.keys())
        n_plots = len(keys)
        
        if n_plots == 0:
            print(f"  ⚠️ No keys provided for {title}")
            return

        fig, axs = plt.subplots(1, n_plots, figsize=(6 * n_plots + 2, 8), sharex=False)
        if n_plots == 1:
            axs = [axs]
            
        fig.suptitle(title, fontweight='bold', y=0.98)

        for idx, key in enumerate(keys):
            ax = axs[idx]
            counts = counts_dict[key]
            
            if not counts:
                # Safe fallback if the model/session generated NO .tran lines
                ax.text(0.5, 0.5, "No `.tran` data found\nin these files.", ha='center', va='center', fontsize=14, color='crimson', fontweight='bold')
                ax.set_title(f"{key} (0 Matches)", fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            top_items = counts.most_common(15)
            labels = [item[0] for item in top_items][::-1]
            values = [item[1] for item in top_items][::-1]
            
            bars = ax.barh(labels, values, color=colors.get(key, 'steelblue'), edgecolor='black', alpha=0.8)
            
            ax.set_title(f"{key}", fontweight='bold')
            ax.set_xlabel("Frequency (Count)")
            if idx == 0:
                ax.set_ylabel("`.tran` Arguments")
            ax.grid(True, linestyle='--', alpha=0.5, axis='x')

            for bar in bars:
                width = bar.get_width()
                # Prevent text from squishing if values are small
                offset = max(1, max(values) * 0.02)
                ax.text(width + offset, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', ha='left', va='center', fontsize=11)
                        
            ax.set_xlim(0, max(values) * 1.15)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  🎉 Plot successfully saved to: {os.path.basename(out_path)}")
        plt.close()
        
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR generating plot {out_path}: {e}")

def analyze_all_tran_commands():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')

    # ---------------------------------------------------------
    # 1. Benchmark Analysis
    # ---------------------------------------------------------
    print("\n🔍 Scanning Benchmark directories...")
    models = ["Base", "SFT", "Zycos10"]
    bench_counts = {}
    bench_colors = {"Base": "gray", "SFT": "darkorange", "Zycos10": "seagreen"}
    
    for model in models:
        run_folders = glob.glob(os.path.join(DATA_DIR, f"Bench_{model}_*_001"))
        counts, f_proc, f_tran = get_tran_counts(run_folders)
        bench_counts[model] = counts
        print(f"  -> {model}: Processed {f_proc} files. Found {f_tran} files with a `.tran` line.")
        
    plot_frequencies(
        counts_dict=bench_counts,
        title="Frequency of `.tran` Command Arguments (Benchmark Evaluations)",
        out_path=os.path.join(DATA_DIR, "benchmark_tran_frequencies.png"),
        colors=bench_colors
    )

    # ---------------------------------------------------------
    # 2. Training Analysis
    # ---------------------------------------------------------
    print("\n🔍 Scanning Training directories...")
    training_sessions = ["zycos_008", "zycos_009", "zycos_010"]
    train_counts = {}
    train_colors = {"zycos_008": "royalblue", "zycos_009": "darkorange", "zycos_010": "seagreen"}
    
    for session in training_sessions:
        run_folders = glob.glob(os.path.join(DATA_DIR, session, "Run_*"))
        counts, f_proc, f_tran = get_tran_counts(run_folders)
        train_counts[session] = counts
        print(f"  -> {session}: Processed {f_proc} files. Found {f_tran} files with a `.tran` line.")

    plot_frequencies(
        counts_dict=train_counts,
        title="Frequency of `.tran` Command Arguments (During RL Training)",
        out_path=os.path.join(DATA_DIR, "training_tran_frequencies.png"),
        colors=train_colors
    )
    
    print("\n✅ Analysis Complete!")

if __name__ == "__main__":
    analyze_all_tran_commands()