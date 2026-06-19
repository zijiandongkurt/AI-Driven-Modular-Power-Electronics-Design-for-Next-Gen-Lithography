import os
import json
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tailored font settings for full-width LaTeX page integration
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 22
})

TOP_K_SPREAD = 9

def extract_run_number(folder_name):
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def plot_sequential_components():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')

    target_sessions = ["zycos_008", "zycos_009", "zycos_010"]
    
    # Data tracking
    all_valid_seq = []
    top_k_seq = []
    x_labels = []
    session_boundaries = []
    current_index = 0

    print("🔍 Extracting component frequency data across sessions...")

    for session_name in target_sessions:
        zycos_dir = os.path.join(DATA_DIR, session_name)
        if not os.path.exists(zycos_dir):
            print(f"  ⚠️ Skipping {session_name} (Directory not found)")
            continue

        run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
        if not run_folders:
            continue

        start_idx = current_index

        for run_folder in run_folders:
            run_idx = extract_run_number(os.path.basename(run_folder))
            history_file = os.path.join(run_folder, 'history_db.json')
            if not os.path.exists(history_file): continue
                
            with open(history_file, 'r', encoding='utf-8') as f:
                try: history = json.load(f)
                except json.JSONDecodeError: continue

            # Pre-load Validation and Reward data
            val_db = {}
            reward_db = {}
            for bf in glob.glob(os.path.join(run_folder, 'batch_*')):
                v_file = os.path.join(bf, 'validation_results.json')
                if os.path.exists(v_file):
                    with open(v_file, 'r') as vf: val_db.update(json.load(vf))
                        
                r_file = os.path.join(bf, 'reward_results.json')
                if os.path.exists(r_file):
                    with open(r_file, 'r') as rf:
                        r_data = json.load(rf)
                        for cid, cdata in r_data.get('circuits', {}).items():
                            reward_db[cid] = cdata

            # Filter for unique valid candidates exactly like plot_history_spread.py
            unique_cands = {}
            for cand in history:
                nid = cand.get('netlist_id')
                if not nid: continue
                if not val_db.get(nid, {}).get('passed', False): continue
                rdata = reward_db.get(nid)
                if not rdata: continue
                    
                fit = float(cand.get('fitness', -9999.0))
                thash = cand.get('topo_hash', nid)
                comps = rdata.get('raw_metrics', {}).get('total_components', 0)
                
                if thash not in unique_cands or fit > unique_cands[thash]['fitness']:
                    unique_cands[thash] = {'fitness': fit, 'comps': comps}

            # Extract component arrays
            valid_list = list(unique_cands.values())
            valid_list_sorted = sorted(valid_list, key=lambda x: x['fitness'], reverse=True)
            top_k_list = valid_list_sorted[:TOP_K_SPREAD]

            all_valid_seq.append([c['comps'] for c in valid_list_sorted])
            top_k_seq.append([c['comps'] for c in top_k_list])
            x_labels.append(f"R{run_idx}")
            current_index += 1

        end_idx = current_index - 1
        session_boundaries.append((session_name, start_idx, end_idx))
        print(f"  ✅ Processed {session_name} ({end_idx - start_idx + 1} runs)")

    if not all_valid_seq:
        print("❌ No valid data extracted.")
        return

    # --- Plotting Builder ---
    def build_stacked_plot(comp_lists, title, filename):
        plt.figure(figsize=(18, 7))
        x_positions = np.arange(len(comp_lists))
        
        # Determine all unique component counts across the entire sequence
        all_comps = sorted(list(set(c for lst in comp_lists for c in lst if not np.isnan(c))))
        
        if all_comps:
            bottoms = np.zeros(len(x_positions))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_comps)))
            
            for c, color in zip(all_comps, colors):
                counts = [lst.count(c) for lst in comp_lists]
                plt.bar(x_positions, counts, bottom=bottoms, width=0.7, 
                        label=f'{int(c)} Comps', color=color, edgecolor='black', alpha=0.85, zorder=3)
                bottoms += np.array(counts)

        # Draw backgrounds and labels
        label_mapping = {
            "zycos_008": "zycos_008 (Easy Constraints)",
            "zycos_009": "zycos_009 (Medium Constraints)",
            "zycos_010": "zycos_010 (Hard Constraints)"
        }

        max_y = max([len(lst) for lst in comp_lists]) if comp_lists else 10
        plt.ylim(0, max_y * 1.15) # Add 15% headroom for titles

        for idx, (session_name, start, end) in enumerate(session_boundaries):
            bg_alpha = 0.04 if idx % 2 == 0 else 0.09
            plt.axvspan(start - 0.5, end + 0.5, color='gray', alpha=bg_alpha, zorder=1)
            
            if end < len(comp_lists) - 1:
                plt.axvline(x=end + 0.5, color='black', linestyle=':', alpha=0.4, zorder=2)
                
            mid_x = (start + end) / 2
            plt.text(mid_x, max_y * 1.05, label_mapping.get(session_name, session_name), 
                     ha='center', va='bottom', fontsize=13, fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

        plt.title(title, fontweight='bold', pad=25)
        plt.xlabel("Sequential Iterations (Runs 1-10 per Difficulty Group)")
        plt.ylabel("Number of Valid Circuits")
        
        plt.xticks(x_positions, x_labels)
        plt.grid(True, linestyle='--', alpha=0.4, axis='y', zorder=0)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fancybox=True, shadow=True)

        results_root = os.path.join(DATA_DIR, 'combined_results')
        os.makedirs(results_root, exist_ok=True)
        out_path = os.path.join(results_root, filename)
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Generated: {out_path}")

    # --- Generate Both Plots ---
    print("\n📊 Generating Master Sequential Plots...")
    build_stacked_plot(all_valid_seq, 
                       "Sequential Blueprint: All Valid Unique Circuit Component Frequencies", 
                       "master_components_all_valid.png")
                       
    build_stacked_plot(top_k_seq, 
                       f"Sequential Blueprint: Top {TOP_K_SPREAD} Valid Circuit Component Frequencies", 
                       "master_components_top_k.png")
    
    print("🎉 Done!")

if __name__ == "__main__":
    plot_sequential_components()