import os
import json
import glob
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Adjusted font sizes for a full LaTeX page (smaller than previous, but readable)
plt.rcParams.update({
    'axes.titlesize': 18,    
    'axes.labelsize': 14,    
    'xtick.labelsize': 12,   
    'ytick.labelsize': 12,   
    'legend.fontsize': 12,   
    'figure.titlesize': 22   
})

def extract_run_number(folder_name):
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def extract_zycos_data(zycos_name, data_dir):
    """Extracts the best candidate metrics across all runs for a single zycos session."""
    zycos_dir = os.path.join(data_dir, zycos_name)
    if not os.path.exists(zycos_dir):
        print(f"⚠️ Directory not found: {zycos_dir}. Skipping...")
        return None
        
    run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
    
    if not run_folders:
        return None

    run_numbers, v_err_pcts, eff_errors, volumes, components = [], [], [], [], []

    for run_folder in run_folders:
        run_idx = extract_run_number(os.path.basename(run_folder))
        history_file = os.path.join(run_folder, 'history_db.json')
        
        if not os.path.exists(history_file): continue
            
        with open(history_file, 'r', encoding='utf-8') as f:
            try: history = json.load(f)
            except json.JSONDecodeError: continue
                
        if not history: continue
            
        best_cand = max(history, key=lambda x: float(x.get('fitness', -9999.0)))
        cand_id = best_cand.get('netlist_id')
        batch_id = best_cand.get('batch_id') 
        
        if not batch_id or not cand_id: continue
            
        reward_file = os.path.join(data_dir, batch_id, 'reward_results.json')
        if not os.path.exists(reward_file): continue
            
        with open(reward_file, 'r', encoding='utf-8') as f:
            try: reward_data = json.load(f)
            except json.JSONDecodeError: continue
                
        ac = reward_data.get('active_constraints', {})
        target_v = ac.get('vout_target', 5.0)
        target_eff = ac.get('efficiency_target', 0.8)
        
        cdata = reward_data.get('circuits', {}).get(cand_id, {})
        raw_metrics = cdata.get('raw_metrics', {})
        
        v_out = raw_metrics.get('simulation_output_voltage', 0)
        eff = raw_metrics.get('efficiency', 0)
        comps = raw_metrics.get('total_components', 0)
        
        try: raw_vol = float(raw_metrics.get('total_volume_cm3', 1.0))
        except (TypeError, ValueError): raw_vol = 1.0
            
        vol = raw_vol
        if vol <= 0: vol = 1.0
        elif vol > 10000.0: vol = 10000.0
            
        v_err = target_v - v_out
        safe_target_v = target_v if target_v != 0 else 1e-6
        v_err_pct = (v_err / safe_target_v) * 100.0
        eff_err = target_eff - eff 
        
        run_numbers.append(run_idx)
        v_err_pcts.append(v_err_pct)
        eff_errors.append(eff_err)
        volumes.append(vol)
        components.append(comps)
        
    return {
        'runs': run_numbers,
        'v_err_pct': v_err_pcts,
        'eff_err': eff_errors,
        'vol': volumes,
        'comps': components
    }

def plot_combined_sessions():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    sessions = ["zycos_008", "zycos_009", "zycos_010"]
    styles = {
        "zycos_008": {"color": "royalblue", "marker": "o", "label": "zycos_008 (Easy)"},
        "zycos_009": {"color": "darkorange", "marker": "s", "label": "zycos_009 (Medium)"},
        "zycos_010": {"color": "seagreen", "marker": "^", "label": "zycos_010 (Hard)"}
    }
    
    all_data = {}
    for s in sessions:
        data = extract_zycos_data(s, DATA_DIR)
        if data and data['runs']:
            all_data[s] = data
            
    if not all_data:
        print("❌ No data found to plot.")
        return

    # Create 2x2 Plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Constraint Satisfaction Comparison Across Difficulty Tiers', fontweight='bold', y=0.98)
    
    global_min_eff = 0
    
    # Plot data on subplots
    for session_name, data in all_data.items():
        st = styles[session_name]
        runs = data['runs']
        
        # Track global min efficiency for the green shaded area
        if data['eff_err']:
            global_min_eff = min(global_min_eff, min(data['eff_err']))
        
        axs[0, 0].plot(runs, data['v_err_pct'], marker=st['marker'], color=st['color'], linewidth=2, label=st['label'])
        axs[0, 1].plot(runs, data['eff_err'], marker=st['marker'], color=st['color'], linewidth=2, label=st['label'])
        axs[1, 0].plot(runs, data['vol'], marker=st['marker'], color=st['color'], linewidth=2, label=st['label'])
        axs[1, 1].plot(runs, data['comps'], marker=st['marker'], color=st['color'], linewidth=2, label=st['label'])

    # --- Formatting Subplot 0,0: Voltage Error ---
    axs[0, 0].axhline(y=10, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Tol. (+/- 10%)')
    axs[0, 0].axhline(y=-10, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    axs[0, 0].axhspan(ymin=-10, ymax=10, color='mediumseagreen', alpha=0.1)
    axs[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axs[0, 0].set_title('Voltage Error (%)', fontweight='bold')
    axs[0, 0].set_ylabel('Error (%)')
    axs[0, 0].legend(loc='best')
    
    # --- Formatting Subplot 0,1: Efficiency Error ---
    axs[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Target Met (0)')
    y_bottom_eff = global_min_eff - abs(global_min_eff * 0.2) if global_min_eff < 0 else -0.1
    axs[0, 1].axhspan(ymin=y_bottom_eff, ymax=0, color='mediumseagreen', alpha=0.1)
    axs[0, 1].set_title('Efficiency Error (Target - Actual)', fontweight='bold')
    axs[0, 1].set_ylabel('Efficiency Error')
    axs[0, 1].legend(loc='best')
    
    # --- Formatting Subplot 1,0: Volume ---
    axs[1, 0].axhline(y=9000, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Burn Threshold')
    axs[1, 0].axhspan(ymin=9000, ymax=10000, color='crimson', alpha=0.15)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylim(bottom=1.0, top=10000.0)
    axs[1, 0].set_title('Total Volume', fontweight='bold')
    axs[1, 0].set_ylabel('Volume (cm³)')
    axs[1, 0].legend(loc='best')
    
    # --- Formatting Subplot 1,1: Components ---
    axs[1, 1].set_title('Total Components', fontweight='bold')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend(loc='best')
    
    # Apply global formatting to all subplots
    for ax in axs.flat:
        ax.set_xlabel('Run Number')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Ensure x-ticks line up with integers (1-10)
        ax.set_xticks(range(1, 11))
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    out_path = os.path.join(DATA_DIR, "combined_cs_summary_8_to_10.png")
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Combined constraint satisfaction plot saved to {out_path}")
    except PermissionError:
        print(f"❌ PERMISSION ERROR: Cannot overwrite {out_path}! Close it in your viewer and try again.")
    plt.close()

if __name__ == "__main__":
    plot_combined_sessions()