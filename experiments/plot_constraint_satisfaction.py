import os
import json
import glob
import re

# Use thread-safe, headless backend to prevent Tkinter crashes during ML loops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_run_number(folder_name):
    """Extract the integer run number from a folder name like 'Run_005'."""
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def process_zycos_run(zycos_name):
    """Scan all runs in a zycos folder, extract best metrics, and plot constraint satisfaction."""
    
    # Setup Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    zycos_dir = os.path.join(DATA_DIR, zycos_name)
    
    if not os.path.exists(zycos_dir):
        print(f"⚠️ Directory not found: {zycos_dir}. Skipping...")
        return
        
    results_dir = os.path.join(zycos_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
    
    if not run_folders:
        print(f"⚠️ No Run folders found in {zycos_name}. Skipping...")
        return

    # --- NEW: Added v_err_pcts to track percentage error ---
    run_numbers, v_errors, abs_v_errors, v_err_pcts, eff_errors, volumes, components = [], [], [], [], [], [], []

    print(f"\n🔍 Scanning {zycos_name} for Best Candidates:")
    for run_folder in run_folders:
        run_idx = extract_run_number(os.path.basename(run_folder))
        history_file = os.path.join(run_folder, 'history_db.json')
        
        if not os.path.exists(history_file):
            continue
            
        with open(history_file, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                continue
                
        if not history:
            continue
            
        # 1. Find the global best candidate exactly like SummaryLogger
        best_cand = max(history, key=lambda x: float(x.get('fitness', -9999.0)))
        cand_id = best_cand.get('netlist_id')
        batch_id = best_cand.get('batch_id') 
        
        if not batch_id or not cand_id:
            continue
            
        # 2. Locate the specific batch to grab its reward metrics
        reward_file = os.path.join(DATA_DIR, batch_id, 'reward_results.json')
        if not os.path.exists(reward_file):
            continue
            
        with open(reward_file, 'r', encoding='utf-8') as f:
            try:
                reward_data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        # 3. Extract the targets
        ac = reward_data.get('active_constraints', {})
        target_v = ac.get('vout_target', 5.0)
        target_eff = ac.get('efficiency_target', 0.8)
        
        # 4. Extract physical metrics
        cdata = reward_data.get('circuits', {}).get(cand_id, {})
        raw_metrics = cdata.get('raw_metrics', {})
        
        v_out = raw_metrics.get('simulation_output_voltage', 0)
        eff = raw_metrics.get('efficiency', 0)
        comps = raw_metrics.get('total_components', 0)
        
        # Safely extract and format volume
        try:
            raw_vol = float(raw_metrics.get('total_volume_cm3', 1.0))
        except (TypeError, ValueError):
            raw_vol = 1.0
            
        # Debug print commented out for cleaner terminal output
        # print(f"  -> Run {run_idx:02d} | Best: {cand_id[:20]}... | Raw Vol: {raw_vol:.4f} cm³")
            
        # Clamp bounds so log scales don't crash
        vol = raw_vol
        if vol <= 0:
            vol = 1.0
        elif vol > 10000.0:
            vol = 10000.0
            
        # 5. Calculate Errors (Target - Actual)
        v_err = target_v - v_out
        abs_v_err = abs(v_err)
        
        # --- NEW: Calculate Percentage Voltage Error ---
        # Safeguard against division by zero just in case
        safe_target_v = target_v if target_v != 0 else 1e-6
        v_err_pct = (v_err / safe_target_v) * 100.0
        
        eff_err = target_eff - eff 
        
        run_numbers.append(run_idx)
        v_errors.append(v_err)
        abs_v_errors.append(abs_v_err)
        v_err_pcts.append(v_err_pct)
        eff_errors.append(eff_err)
        volumes.append(vol)
        components.append(comps)
        
    if not run_numbers:
        return
        
    print(f"📊 Generating plots for {zycos_name}...")
    
    # --- PLOTTING LOGIC ---
    def plot_individual(x, y, title, ylabel, filename, color='blue', is_log=False, hline_y=None, hline_name='Target Met', limit_type='lower', y_max=None, y_min=None):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2, markersize=8)
        
        if hline_y is not None:
            plt.axhline(y=hline_y, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'{hline_name}')
            
            if limit_type == 'lower':
                y_bottom = y_min if y_min is not None else (1.0 if is_log else (min(y)-abs(min(y)*0.2) if y else hline_y - 1))
                plt.axhspan(ymin=y_bottom, ymax=hline_y, color='mediumseagreen', alpha=0.1, label=f'Good Zone')
            
            elif limit_type == 'upper':
                y_top = y_max if y_max is not None else 10000.0
                plt.axhspan(ymin=hline_y, ymax=y_top, color='crimson', alpha=0.15, label='Burn Zone')
                
            elif limit_type == 'band':
                band_bottom = y_min if y_min is not None else -hline_y
                plt.axhline(y=band_bottom, color='black', linestyle='--', linewidth=2, alpha=0.7)
                plt.axhspan(ymin=band_bottom, ymax=hline_y, color='mediumseagreen', alpha=0.1, label=f'Good Zone')
                plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5) # Centerline
                
            plt.legend(loc='best')
            
        plt.title(f"{zycos_name} - {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Run Number", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if is_log:
            plt.yscale('log')
            plt.ylim(bottom=1.0, top=10000.0)
        else:
            if y_min is not None and limit_type != 'band':
                plt.ylim(bottom=y_min)
            if y_max is not None:
                plt.ylim(top=y_max)
            
        plt.xticks(x)
        
        out_path = os.path.join(results_dir, filename)
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        except PermissionError:
            print(f"❌ PERMISSION ERROR: Cannot overwrite {filename}. Close the image in VS Code and try again!")
            
        plt.close()
        
    # 1. Generate Individual Plots
    plot_individual(run_numbers, v_errors, "Voltage Error (Target - Actual)", "Voltage Error (V)", "cs_err_voltage.png", color='royalblue', hline_y=1, hline_name='Tolerance (+/- 1V)', limit_type='band', y_min=-1)
    plot_individual(run_numbers, abs_v_errors, "Absolute Voltage Error", "|Voltage Error| (V)", "cs_err_abs_voltage.png", color='crimson', hline_y=1, hline_name='Tolerance (1V)', limit_type='lower', y_min=0)
    
    # --- NEW: Standalone Percentage Voltage Error Plot (+/- 5% Tolerance Band) ---
    plot_individual(run_numbers, v_err_pcts, "Percentage Voltage Error", "Voltage Error (%)", "cs_err_voltage_pct.png", color='deepskyblue', hline_y=5, hline_name='Tolerance (+/- 5%)', limit_type='band', y_min=-5)
    
    plot_individual(run_numbers, eff_errors, "Efficiency Error (Target - Actual)", "Efficiency Error", "cs_err_efficiency.png", color='darkorange', hline_y=0, hline_name='Target Met (0)', limit_type='lower')
    plot_individual(run_numbers, volumes, "Total Volume", "Volume (cm³)", "cs_metric_volume.png", color='purple', is_log=True, hline_y=9000, hline_name='Burn Threshold', limit_type='upper', y_max=10000)
    plot_individual(run_numbers, components, "Total Component Count", "Components", "cs_metric_components.png", color='seagreen')
    
    # 2. Generate 2x2 Subplot Summary
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Constraint Satisfaction & Metrics Summary ({zycos_name})', fontsize=16, fontweight='bold')
    
    # --- NEW: Subplot 1 is now Percentage Voltage Error with a +/- 5% Tolerance Band ---
    axs[0, 0].plot(run_numbers, v_err_pcts, marker='o', color='deepskyblue', linewidth=2)
    
    axs[0, 0].axhline(y=5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Tolerance (+/- 5%)')
    axs[0, 0].axhline(y=-5, color='black', linestyle='--', linewidth=2, alpha=0.7)
    axs[0, 0].axhspan(ymin=-5, ymax=5, color='mediumseagreen', alpha=0.1, label='Good Zone (+/- 5%)')
    axs[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5) # Center line
    
    axs[0, 0].legend(loc='best')
    axs[0, 0].set_title('Voltage Error (%)', fontweight='bold')
    axs[0, 0].set_ylabel('Error (%)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].set_xticks(run_numbers)
    
    # Subplot 2: Efficiency
    axs[0, 1].plot(run_numbers, eff_errors, marker='s', color='darkorange', linewidth=2)
    axs[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Target Met (0)')
    if eff_errors:
        axs[0, 1].axhspan(ymin=min(eff_errors)-abs(min(eff_errors)*0.2), ymax=0, color='mediumseagreen', alpha=0.1)
    axs[0, 1].legend(loc='best')
    axs[0, 1].set_title('Efficiency Error (Target - Actual)', fontweight='bold')
    axs[0, 1].set_ylabel('Efficiency Error')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].set_xticks(run_numbers)
    
    # Subplot 3: Volume
    axs[1, 0].plot(run_numbers, volumes, marker='^', color='purple', linewidth=2)
    axs[1, 0].axhline(y=9000, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Burn Threshold')
    axs[1, 0].axhspan(ymin=9000, ymax=10000, color='crimson', alpha=0.15, label='Burn Zone')
    
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylim(bottom=1.0, top=10000.0) # Strictly clamp 1 to 10^4
    
    axs[1, 0].legend(loc='best')
    axs[1, 0].set_title('Total Volume', fontweight='bold')
    axs[1, 0].set_ylabel('Volume (cm³)')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].set_xticks(run_numbers)
    
    # Subplot 4: Components
    axs[1, 1].plot(run_numbers, components, marker='D', color='seagreen', linewidth=2)
    axs[1, 1].set_title('Total Components', fontweight='bold')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].set_xticks(run_numbers)
    
    for ax in axs.flat:
        ax.set_xlabel('Run Number')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path_2x2 = os.path.join(results_dir, "cs_summary_2x2.png")
    try:
        plt.savefig(out_path_2x2, dpi=300, bbox_inches='tight')
        print(f"✅ Constraint Satisfaction plots saved to {results_dir}")
    except PermissionError:
        print(f"❌ PERMISSION ERROR: Cannot overwrite cs_summary_2x2.png! Close it in VS Code and try again.")
        
    plt.close()

if __name__ == "__main__":
    for i in range(5, 10):
        run_name = f"zycos_{i:03d}"
        process_zycos_run(run_name)