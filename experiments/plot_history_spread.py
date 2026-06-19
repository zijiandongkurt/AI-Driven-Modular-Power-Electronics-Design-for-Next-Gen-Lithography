import os
import json
import glob
import re
import numpy as np

# Use thread-safe, headless backend to prevent Tkinter crashes during ML loops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOP_K_SPREAD = 10  # How many top unique circuits to include in the Top K plots

def extract_run_number(folder_name):
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def process_zycos_run(zycos_name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    zycos_dir = os.path.join(DATA_DIR, zycos_name)
    
    if not os.path.exists(zycos_dir):
        print(f" Directory not found: {zycos_dir}. Skipping...")
        return
        
    results_dir = os.path.join(zycos_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')), key=lambda x: extract_run_number(os.path.basename(x)))
    if not run_folders:
        return

    all_valid_data = {}
    top_k_data = {}

    print(f"\n Scanning {zycos_name} for Spread Analysis:")
    
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
                
        # Pre-load all Validation and Reward data into fast dictionaries
        val_db = {}
        reward_db = {}
        batch_folders = glob.glob(os.path.join(run_folder, 'batch_*'))
        
        for bf in batch_folders:
            v_file = os.path.join(bf, 'validation_results.json')
            if os.path.exists(v_file):
                with open(v_file, 'r') as vf:
                    val_db.update(json.load(vf))
                    
            r_file = os.path.join(bf, 'reward_results.json')
            if os.path.exists(r_file):
                with open(r_file, 'r') as rf:
                    r_data = json.load(rf)
                    # Sync active constraints from the reward file
                    ac = r_data.get('active_constraints', {})
                    t_v = ac.get('vout_target', 5.0)
                    t_e = ac.get('efficiency_target', 0.8)
                    for cid, cdata in r_data.get('circuits', {}).items():
                        cdata['target_v'] = t_v
                        cdata['target_eff'] = t_e
                        reward_db[cid] = cdata
        
        # Filter for unique valid candidates
        unique_cands = {}
        
        for cand in history:
            nid = cand.get('netlist_id')
            if not nid: continue
            
            # Check validation status
            is_valid = val_db.get(nid, {}).get('passed', False)
            if not is_valid: continue
                
            rdata = reward_db.get(nid)
            if not rdata: continue
                
            raw_metrics = rdata.get('raw_metrics', {})
            fit = float(cand.get('fitness', -9999.0))
            thash = cand.get('topo_hash', nid)
            
            target_v = rdata.get('target_v', 5.0)
            target_eff = rdata.get('target_eff', 0.8)
            
            v_out = raw_metrics.get('simulation_output_voltage', 0)
            eff = raw_metrics.get('efficiency', 0)
            comps = raw_metrics.get('total_components', 0)
            
            try:
                vol = float(raw_metrics.get('total_volume_cm3', 1.0))
            except:
                vol = 1.0
            vol = max(1.0, min(vol, 10000.0))
            
            v_err = target_v - v_out
            abs_v_err = abs(v_err)
            v_err_pct = (v_err / target_v) * 100.0 if target_v != 0 else 0
            eff_err = target_eff - eff
            
            cand_payload = {
                'fitness': fit, 'v_err': v_err, 'abs_v_err': abs_v_err, 
                'v_err_pct': v_err_pct, 'eff_err': eff_err, 
                'vol': vol, 'comps': comps
            }
            
            # Keep only the highest fitness variant of a topological duplicate
            if thash not in unique_cands or fit > unique_cands[thash]['fitness']:
                unique_cands[thash] = cand_payload
                
        valid_list = list(unique_cands.values())
        if not valid_list:
            continue
            
        # Sort by fitness to extract the Top K
        valid_list_sorted = sorted(valid_list, key=lambda x: x['fitness'], reverse=True)
        top_k_list = valid_list_sorted[:TOP_K_SPREAD]
        
        all_valid_data[run_idx] = valid_list_sorted
        top_k_data[run_idx] = top_k_list
        
        print(f"  -> Run {run_idx:02d} | Valid Unique: {len(valid_list_sorted)} | Top K Analyzed: {len(top_k_list)}")
        
    if not all_valid_data:
        print(f" No valid data extracted for {zycos_name}")
        return
        
    def generate_spread_plots(dataset, file_prefix, master_title):
        run_numbers = sorted(list(dataset.keys()))
        
        v_errors = [[d['v_err'] for d in dataset[r]] for r in run_numbers]
        abs_v_errors = [[d['abs_v_err'] for d in dataset[r]] for r in run_numbers]
        v_err_pcts = [[d['v_err_pct'] for d in dataset[r]] for r in run_numbers]
        eff_errors = [[d['eff_err'] for d in dataset[r]] for r in run_numbers]
        volumes = [[d['vol'] for d in dataset[r]] for r in run_numbers]
        components = [[d['comps'] for d in dataset[r]] for r in run_numbers]
        
        # 1. Boxplot Builder for Continuous Metrics
        def plot_box(ax, x, y_lists, title, ylabel, color, is_log=False, hline_y=None, hline_name=None, limit_type='lower', y_max=None, y_min=None, is_standalone=False):
            # Sanitize empty lists to avoid matplotlib crashes
            plot_data = [lst if len(lst) > 0 else [np.nan] for lst in y_lists]
            
            box = ax.boxplot(plot_data, positions=x, patch_artist=True, widths=0.5,
                             boxprops=dict(facecolor=color, color='black', alpha=0.7),
                             capprops=dict(color='black'),
                             whiskerprops=dict(color='black'),
                             flierprops=dict(marker='o', markerfacecolor=color, alpha=0.5, markersize=4),
                             medianprops=dict(color='black', linewidth=2))
                             
            if hline_y is not None:
                label_text = hline_name if hline_name else f'Threshold ({hline_y})'
                ax.axhline(y=hline_y, color='black', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
                
                # Dynamically find the data bounds so we don't squish the graph
                flat_data = [val for lst in plot_data for val in lst if not np.isnan(val)]
                
                if limit_type == 'lower':
                    if y_min is not None:
                        bottom = y_min
                    elif is_log:
                        bottom = 1.0
                    else:
                        data_min = min(flat_data) if flat_data else (hline_y - 1)
                        bottom = data_min - abs(data_min * 0.2)
                        
                    ax.axhspan(ymin=bottom, ymax=hline_y, color='mediumseagreen', alpha=0.1, label='Good Zone')
                    
                elif limit_type == 'upper':
                    if y_max is not None:
                        top = y_max
                    else:
                        data_max = max(flat_data) if flat_data else (hline_y + 1)
                        top = data_max + abs(data_max * 0.2)
                        
                    ax.axhspan(ymin=hline_y, ymax=top, color='crimson', alpha=0.15, label='Burn Zone')
                    
                elif limit_type == 'band':
                    band_bottom = y_min if y_min is not None else -hline_y
                    ax.axhline(y=band_bottom, color='black', linestyle='--', linewidth=2, alpha=0.7)
                    ax.axhspan(ymin=band_bottom, ymax=hline_y, color='mediumseagreen', alpha=0.1, label='Tolerance Zone')
                    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                
                ax.legend(loc='best', fontsize=9)
                
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xticks(x)
            
            if is_log:
                ax.set_yscale('log')
                ax.set_ylim(bottom=1.0, top=10000.0)
            else:
                if y_min is not None and limit_type != 'band':
                    ax.set_ylim(bottom=y_min)
                if y_max is not None:
                    ax.set_ylim(top=y_max)
                    
            if is_standalone:
                ax.set_xlabel('Run Number', fontsize=10)

        # 2. Stacked Bar Chart Builder for Discrete Components
        def plot_comp_bars(ax, x, comp_lists, title, ylabel, is_standalone=False):
            all_comps = sorted(list(set(c for lst in comp_lists for c in lst if not np.isnan(c))))
            if not all_comps: return
            
            bottoms = np.zeros(len(x))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_comps)))
            
            for c, color in zip(all_comps, colors):
                counts = [lst.count(c) for lst in comp_lists]
                ax.bar(x, counts, bottom=bottoms, width=0.6, label=f'{int(c)} Comps', color=color, edgecolor='black', alpha=0.85)
                bottoms += np.array(counts)
                
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax.set_xticks(x)
            ax.legend(loc='best', fontsize=8)
            
            if is_standalone:
                ax.set_xlabel('Run Number', fontsize=10)

        def save_standalone(func, y_data, title, ylabel, filename, **kwargs):
            fig, ax = plt.subplots(figsize=(10, 6))
            func(ax, run_numbers, y_data, f"{master_title} - {title}", ylabel, is_standalone=True, **kwargs)
            try:
                plt.savefig(os.path.join(results_dir, f"{file_prefix}_{filename}"), dpi=300, bbox_inches='tight')
            except PermissionError:
                print(f" PERMISSION ERROR: Close {filename} to allow overwrite.")
            plt.close()

        save_standalone(plot_box, v_err_pcts, "Voltage Error (%)", "Error (%)", "err_voltage_pct.png", color='deepskyblue', hline_y=10, hline_name='Tolerance (+/- 10%)', limit_type='band', y_min=-10)
        save_standalone(plot_box, abs_v_errors, "Absolute Voltage Error", "|Error| (V)", "err_abs_voltage.png", color='crimson', hline_y=1, hline_name='Tolerance (1V)', limit_type='lower', y_min=0)
        save_standalone(plot_box, eff_errors, "Efficiency Error", "Error", "err_efficiency.png", color='darkorange', hline_y=0, hline_name='Target Met (0)', limit_type='lower')
        save_standalone(plot_box, volumes, "Volume Spread", "Volume (cm³)", "metric_volume.png", color='purple', is_log=True, hline_y=9000, hline_name='Burn Threshold', limit_type='upper', y_max=10000)
        save_standalone(plot_comp_bars, components, "Component Frequency", "Number of Valid Circuits", "metric_components.png")

        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{master_title} ({zycos_name})', fontsize=16, fontweight='bold')
        
        plot_box(axs[0, 0], run_numbers, v_err_pcts, 'Voltage Error Spread (%)', 'Error (%)', color='deepskyblue', hline_y=10, hline_name='Tolerance (+/- 10%)', limit_type='band', y_min=-10)
        plot_box(axs[0, 1], run_numbers, eff_errors, 'Efficiency Error Spread', 'Error', color='darkorange', hline_y=0, hline_name='Target Met (0)', limit_type='lower')
        plot_box(axs[1, 0], run_numbers, volumes, 'Total Volume Spread', 'Volume (cm³)', color='purple', is_log=True, hline_y=9000, hline_name='Burn Threshold', limit_type='upper', y_max=10000)
        plot_comp_bars(axs[1, 1], run_numbers, components, 'Component Count Distribution', 'Valid Circuit Yield')
        
        for ax in axs.flat:
            ax.set_xlabel('Run Number')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = os.path.join(results_dir, f"{file_prefix}_summary_2x2.png")
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f" Spread plots saved: {file_prefix}")
        except PermissionError:
            print(f" PERMISSION ERROR: Close {file_prefix}_summary_2x2.png to allow overwrite.")
        plt.close()

    # Generate both data subsets!
    generate_spread_plots(all_valid_data, "spread_all", "All Valid Unique Spread")
    generate_spread_plots(top_k_data, "spread_topk", f"Top {TOP_K_SPREAD} Valid Spread")

if __name__ == "__main__":
    for i in range(10, 11):
        run_name = f"zycos_{i:03d}"
        process_zycos_run(run_name)