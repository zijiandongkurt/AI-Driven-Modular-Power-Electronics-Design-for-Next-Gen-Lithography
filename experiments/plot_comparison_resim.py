import os
import json
import glob
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Cranked up global font sizes for LaTeX readability ---
plt.rcParams.update({
    'axes.titlesize': 22,    
    'axes.labelsize': 18,    
    'xtick.labelsize': 14,   
    'ytick.labelsize': 14,   
    'legend.fontsize': 14,   
    'figure.titlesize': 26   
})

def extract_run_number(folder_name):
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def extract_new_data(zycos_name, data_dir):
    """Extracts data from the pristine pipeline/data folder."""
    zycos_dir = os.path.join(data_dir, zycos_name)
    if not os.path.exists(zycos_dir): return None
        
    run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
    if not run_folders: return None

    run_numbers, v_err_pcts, eff_errors, raw_efficiencies, volumes, components = [], [], [], [], [], []

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
            
        vol = max(1.0, min(raw_vol, 10000.0))
        v_err_pct = ((target_v - v_out) / (target_v if target_v != 0 else 1e-6)) * 100.0
        
        run_numbers.append(run_idx)
        v_err_pcts.append(v_err_pct)
        eff_errors.append(target_eff - eff)
        raw_efficiencies.append(eff * 100.0) 
        volumes.append(vol)
        components.append(comps)
        
    return {'runs': run_numbers, 'v_err_pct': v_err_pcts, 'eff_err': eff_errors, 'raw_eff': raw_efficiencies, 'vol': volumes, 'comps': components}

def extract_legacy_data(zycos_name, data_dir_legacy):
    """Extracts data from the broken pipeline/data_legacy_broken folder."""
    zycos_dir = os.path.join(data_dir_legacy, zycos_name)
    if not os.path.exists(zycos_dir): return None
        
    run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
    if not run_folders: return None

    run_numbers, v_err_pcts, eff_errors, raw_efficiencies, volumes, components = [], [], [], [], [], []

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
            
        # Legacy mapping route
        batch_folder = batch_id.split('/')[-1]
        reward_file = os.path.join(run_folder, batch_folder, 'reward_results.json')
        
        if not os.path.exists(reward_file): continue # Skip Ghost circuits
            
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
            
        vol = max(1.0, min(raw_vol, 10000.0))
        v_err_pct = ((target_v - v_out) / (target_v if target_v != 0 else 1e-6)) * 100.0
        
        run_numbers.append(run_idx)
        v_err_pcts.append(v_err_pct)
        eff_errors.append(target_eff - eff)
        raw_efficiencies.append(eff * 100.0) 
        volumes.append(vol)
        components.append(comps)
        
    return {'runs': run_numbers, 'v_err_pct': v_err_pcts, 'eff_err': eff_errors, 'raw_eff': raw_efficiencies, 'vol': volumes, 'comps': components}

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    
    DATA_DIR_NEW = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    DATA_DIR_LEGACY = os.path.join(PROJECT_ROOT, 'pipeline', 'data_legacy_broken')
    OUT_DIR = os.path.join(DATA_DIR_NEW, 'comparisons')
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    styles = {
        "zycos_008": {"color": "royalblue", "label": "zycos_008 (Easy)"},
        "zycos_009": {"color": "darkorange", "label": "zycos_009 (Medium)"},
        "zycos_010": {"color": "seagreen", "label": "zycos_010 (Hard)"}
    }
    
    new_data = {}
    legacy_data = {}
    
    print(f"\n{'='*60}\n Extracting Data for Comparisons\n{'='*60}")
    for s in styles.keys():
        n_data = extract_new_data(s, DATA_DIR_NEW)
        l_data = extract_legacy_data(s, DATA_DIR_LEGACY)
        if n_data: new_data[s] = n_data
        if l_data: legacy_data[s] = l_data
        print(f"✅ Loaded {s}")

    def plot_legacy_vs_new_boxplot(models_to_compare, metric_key, title, ylabel, filename, config):
        plt.figure(figsize=(10, 7))
        box_data, labels, face_colors, alphas = [], [], [], []
        
        for m in models_to_compare:
            st = styles[m]
            
            # 1. Pull Legacy Data (Faded)
            leg_d = legacy_data.get(m)
            if leg_d and leg_d[metric_key]:
                box_data.append(leg_d[metric_key])
                labels.append(f"{st['label']}\n[Legacy]")
                face_colors.append(st['color'])
                alphas.append(0.3) # Highly faded alpha
                
            # 2. Pull New Data (Solid)
            new_d = new_data.get(m)
            if new_d and new_d[metric_key]:
                box_data.append(new_d[metric_key])
                labels.append(f"{st['label']}\n[Fixed]")
                face_colors.append(st['color'])
                alphas.append(0.9) # Solid alpha
                
        if not box_data: 
            print(f"❌ No data found for {title}")
            return
            
        bplot = plt.boxplot(box_data, labels=labels, patch_artist=True, widths=0.5, 
                            medianprops=dict(color='black', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=8, alpha=0.6))
                            
        # Apply the alternating alphas
        for patch, color, alpha in zip(bplot['boxes'], face_colors, alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        # Apply specific formatting
        if config == 'voltage':
            plt.axhline(y=10, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Tol. (+/- 10%)')
            plt.axhline(y=-10, color='black', linestyle='--', linewidth=2, alpha=0.7)
            plt.axhspan(ymin=-10, ymax=10, color='mediumseagreen', alpha=0.1, label='Good Zone')
            plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        plt.title(title, fontweight='bold', pad=15)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        if plt.gca().get_legend_handles_labels()[0]: 
            plt.legend(loc='best')
            
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated Comparison: {filename}")

    # =========================================================
    # TARGETED EXECUTION
    # =========================================================
    plot_legacy_vs_new_boxplot(
        models_to_compare=["zycos_008", "zycos_009", "zycos_010"], 
        metric_key="v_err_pct", 
        title="Impact of Code Fix: Voltage Error Distribution", 
        ylabel="Voltage Error (%)", 
        filename="comparison_voltage_error_combined.png", 
        config="voltage"
    )
    
    print(f"\n{'='*60}\n COMPARISON PLOTTING COMPLETE\n{'='*60}\n")

if __name__ == "__main__":
    main()