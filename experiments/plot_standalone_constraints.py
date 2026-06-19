import os
import json
import glob
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'axes.titlesize': 24,    
    'axes.labelsize': 20,    
    'xtick.labelsize': 16,   
    'ytick.labelsize': 16,   
    'legend.fontsize': 16,   
    'figure.titlesize': 28   
})

def extract_run_number(folder_name):
    match = re.search(r'Run_(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def extract_zycos_data(zycos_name, data_dir):
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
            
        vol = raw_vol
        if vol <= 0: vol = 1.0
        elif vol > 10000.0: vol = 10000.0
            
        safe_target_v = target_v if target_v != 0 else 1e-6
        v_err_pct = ((target_v - v_out) / safe_target_v) * 100.0
        eff_err = target_eff - eff 
        
        run_numbers.append(run_idx)
        v_err_pcts.append(v_err_pct)
        eff_errors.append(eff_err)
        raw_efficiencies.append(eff * 100.0) 
        volumes.append(vol)
        components.append(comps)
        
    return {
        'runs': run_numbers, 'v_err_pct': v_err_pcts, 'eff_err': eff_errors,
        'raw_eff': raw_efficiencies, 'vol': volumes, 'comps': components
    }

def generate_standalone_plots():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    OUT_DIR = os.path.join(DATA_DIR, 'constraint_satisfaction')
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    sessions = ["zycos_008", "zycos_009", "zycos_010"]
    styles = {
        "zycos_008": {"color": "royalblue", "marker": "o", "label": "zycos_008 (Easy)"},
        "zycos_009": {"color": "darkorange", "marker": "s", "label": "zycos_009 (Medium)"},
        "zycos_010": {"color": "seagreen", "marker": "^", "label": "zycos_010 (Hard)"}
    }
    
    all_data = {}
    print(f"\n{'='*60}\n Extracting Data for Standalone Plots\n{'='*60}")
    for s in sessions:
        data = extract_zycos_data(s, DATA_DIR)
        if data and data['runs']:
            all_data[s] = data
            print(f"✅ Loaded {s} ({len(data['runs'])} runs)")
            
    if not all_data: return

    def plot_metric(models_to_plot, metric_key, title, ylabel, filename, config):
        plt.figure(figsize=(12, 7))
        global_min_y = 0
        all_runs = set()
        
        for m in models_to_plot:
            d = all_data[m]
            st = styles[m]
            plt.plot(d['runs'], d[metric_key], marker=st['marker'], color=st['color'], linewidth=2, markersize=10, label=st['label'])
            all_runs.update(d['runs'])
            if d[metric_key]: global_min_y = min(global_min_y, min(d[metric_key]))

        if config == 'voltage':
            plt.axhline(y=10, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Tol. (+/- 10%)')
            plt.axhline(y=-10, color='black', linestyle='--', linewidth=2, alpha=0.7)
            plt.axhspan(ymin=-10, ymax=10, color='mediumseagreen', alpha=0.1, label='Good Zone')
            plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        elif config == 'efficiency':
            plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Target Met (0)')
            y_bottom = global_min_y - abs(global_min_y * 0.2) if global_min_y < 0 else -0.1
            plt.axhspan(ymin=y_bottom, ymax=0, color='mediumseagreen', alpha=0.1, label='Good Zone')
        elif config == 'raw_efficiency':
            plt.ylim(bottom=-5.0, top=105.0)
        elif config == 'volume':
            plt.axhline(y=9000, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Burn Threshold')
            plt.axhspan(ymin=9000, ymax=10000, color='crimson', alpha=0.15, label='Burn Zone')
            plt.yscale('log')
            plt.ylim(bottom=1.0, top=10000.0)

        plot_title = title if len(models_to_plot) > 1 else f"{title} ({styles[models_to_plot[0]]['label']})"
        plt.title(plot_title, fontweight='bold', pad=15)
        plt.xlabel('Run Number')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        if all_runs: plt.xticks(range(1, max(all_runs) + 1))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_boxplot(models_to_plot, metric_key, title, ylabel, filename, config):
        plt.figure(figsize=(10, 7))
        box_data, labels, colors, global_min_y = [], [], [], 0
        
        for m in models_to_plot:
            d = all_data[m]
            if d[metric_key]:
                box_data.append(d[metric_key])
                labels.append(styles[m]['label'])
                colors.append(styles[m]['color'])
                global_min_y = min(global_min_y, min(d[metric_key]))
                
        if not box_data: return
            
        bplot = plt.boxplot(box_data, labels=labels, patch_artist=True, widths=0.4, 
                            medianprops=dict(color='black', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=8, alpha=0.6))
                            
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if config == 'voltage':
            plt.axhline(y=10, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Tol. (+/- 10%)')
            plt.axhline(y=-10, color='black', linestyle='--', linewidth=2, alpha=0.7)
            plt.axhspan(ymin=-10, ymax=10, color='mediumseagreen', alpha=0.1, label='Good Zone')
            plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        elif config == 'efficiency':
            plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Target Met (0)')
            y_bottom = global_min_y - abs(global_min_y * 0.2) if global_min_y < 0 else -0.1
            plt.axhspan(ymin=y_bottom, ymax=0, color='mediumseagreen', alpha=0.1, label='Good Zone')
        elif config == 'raw_efficiency':
            plt.ylim(bottom=-5.0, top=105.0)
        elif config == 'volume':
            plt.axhline(y=9000, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Burn Threshold')
            plt.axhspan(ymin=9000, ymax=10000, color='crimson', alpha=0.15, label='Burn Zone')
            plt.yscale('log')
            plt.ylim(bottom=1.0, top=10000.0)

        plt.title(f"Distribution of {title}", fontweight='bold', pad=15)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        if plt.gca().get_legend_handles_labels()[0]: plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_frequency(models_to_plot, metric_key, title, ylabel, filename, config):
        plt.figure(figsize=(12, 7))
        all_vals = []
        for m in models_to_plot:
            if all_data[m][metric_key]:
                all_vals.extend(all_data[m][metric_key])
                
        if not all_vals: return
            
        min_val, max_val = int(min(all_vals)), int(max(all_vals))
        x_ticks = list(range(min_val, max_val + 1))
        width = 0.8 / len(models_to_plot)
        
        for idx, m in enumerate(models_to_plot):
            vals = all_data[m][metric_key]
            if not vals: continue
            
            counts = [vals.count(x) for x in x_ticks]
            offset = (idx - len(models_to_plot)/2.0 + 0.5) * width
            x_pos = [x + offset for x in x_ticks]
            
            plt.bar(x_pos, counts, width=width, color=styles[m]['color'], alpha=0.8, edgecolor='black', label=styles[m]['label'])

        plot_title = f"Frequency of {title}" if len(models_to_plot) > 1 else f"Frequency of {title} ({styles[models_to_plot[0]]['label']})"
        plt.title(plot_title, fontweight='bold', pad=15)
        plt.xlabel(title)
        plt.ylabel(ylabel)
        plt.xticks(x_ticks)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

    metrics_map = [
        ('v_err_pct', 'Voltage Error (%)', 'Error (%)', 'voltage_error', 'voltage'),
        ('eff_err', 'Efficiency Error (Target - Actual)', 'Efficiency Error', 'efficiency_error', 'efficiency'),
        ('raw_eff', 'Raw Efficiency', 'Efficiency (%)', 'raw_efficiency', 'raw_efficiency'),
        ('vol', 'Total Volume', 'Volume (cm³)', 'total_volume', 'volume'),
        ('comps', 'Total Components', 'Count', 'total_components', 'components')
    ]
    
    print(f"\nGenerating Plots in: {OUT_DIR}")
    for metric_key, title, ylabel, file_prefix, config in metrics_map:
        if metric_key == 'comps':
            for m in sessions:
                if m in all_data:
                    plot_metric([m], metric_key, title, ylabel, f"{file_prefix}_{m}.png", config)
                    plot_frequency([m], metric_key, title, "Frequency", f"{file_prefix}_freq_{m}.png", config)
            plot_metric(sessions, metric_key, f"Combined {title}", ylabel, f"{file_prefix}_combined.png", config)
            plot_frequency(sessions, metric_key, f"Combined {title}", "Frequency", f"{file_prefix}_freq_combined.png", config)
            
        else:
            for m in sessions:
                if m in all_data:
                    plot_metric([m], metric_key, title, ylabel, f"{file_prefix}_{m}.png", config)
                    
            if metric_key == 'v_err_pct' and "zycos_008" in all_data:
                plot_boxplot(["zycos_008"], metric_key, f"{title} (zycos_008)", ylabel, f"{file_prefix}_boxplot_zycos_008.png", config)
                
            plot_metric(sessions, metric_key, f"Combined {title}", ylabel, f"{file_prefix}_combined.png", config)
            plot_boxplot(sessions, metric_key, title, ylabel, f"{file_prefix}_boxplot_combined.png", config)

    print(f"\n{'='*60}\n PLOTTING COMPLETE\n{'='*60}\n")

if __name__ == "__main__":
    generate_standalone_plots()