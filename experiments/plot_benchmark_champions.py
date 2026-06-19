import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tailored font settings for full-width LaTeX page integration
plt.rcParams.update({
    'axes.titlesize': 16,    
    'axes.labelsize': 13,    
    'xtick.labelsize': 11,   
    'ytick.labelsize': 12,   
    'legend.fontsize': 11,   
    'figure.titlesize': 20   
})

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def extract_benchmark_data():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'benchmark_config.json')
    config = load_json(config_path)
    if not config: return None

    models = list(config.get('models', {}).keys())
    tasks = config.get('tasks', [])
    trials = config.get('trials_per_task', 2)

    data = {m: {i: {'v_err':[], 'eff_err':[], 'vol':[], 'comps':[]} for i in range(len(tasks))} for m in models}
    task_labels = []
    
    print("🔍 Extracting Benchmark Champion Metrics...")
    
    for t_idx, task in enumerate(tasks):
        task_labels.append(task['label'])
        
        constraint_file = os.path.join(PROJECT_ROOT, task['file'].replace('/', os.sep))
        constraints = load_json(constraint_file)
        
        target_eff = 0.8
        if constraints and len(constraints) > task['idx']:
            target_eff = constraints[task['idx']].get('efficiency_target', 0.8)
            
        for model in models:
            for trial in range(1, trials + 1):
                run_folder_name = f"Bench_{model}_{task['label']}_T{trial}_001"
                
                # UPDATED: Now looks inside the "database" subfolder
                champ_path = os.path.join(DATA_DIR, run_folder_name, "champion_metrics.json")
                
                champ_data = load_json(champ_path)
                if not champ_data: continue
                
                target_v = champ_data.get('target_voltage', 5.0)
                raw_v = champ_data.get('raw_voltage', 0.0)
                raw_eff = champ_data.get('raw_efficiency', 0.0)
                raw_vol = champ_data.get('raw_volume', 1.0)
                raw_comps = champ_data.get('raw_components', 0)
                
                vol = max(1.0, min(float(raw_vol), 10000.0))
                
                safe_target_v = target_v if target_v != 0 else 1e-6
                v_err_pct = ((target_v - raw_v) / safe_target_v) * 100.0
                eff_err = target_eff - raw_eff
                
                data[model][t_idx]['v_err'].append(v_err_pct)
                data[model][t_idx]['eff_err'].append(eff_err)
                data[model][t_idx]['vol'].append(vol)
                data[model][t_idx]['comps'].append(raw_comps)

    return data, task_labels

def format_benchmark_axes(fig, axs, title, task_labels):
    fig.suptitle(title, fontweight='bold', y=0.98)
    x_positions = np.arange(len(task_labels))
    
    session_boundaries = [("Easy", 0, 1), ("Medium", 2, 3), ("Hard", 4, 5)]

    for ax in axs.flat:
        for idx, (tier_name, start, end) in enumerate(session_boundaries):
            bg_alpha = 0.03 if idx % 2 == 0 else 0.08
            ax.axvspan(start - 0.5, end + 0.5, color='gray', alpha=bg_alpha, zorder=0)
            
            if end < len(task_labels) - 1:
                ax.axvline(x=end + 0.5, color='black', linestyle=':', alpha=0.4, zorder=1)
                
            mid_rel = ((start + end) / 2.0) / (len(task_labels) - 1)
            ax.text(mid_rel, 1.02, tier_name, transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))

        ax.set_xticks(x_positions)
        ax.set_xticklabels([l.replace('_', '\n') for l in task_labels], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

    axs[0, 0].axhline(y=10, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Tol. (+/- 10%)')
    axs[0, 0].axhline(y=-10, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    axs[0, 0].axhspan(ymin=-10, ymax=10, color='mediumseagreen', alpha=0.1, zorder=0)
    axs[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axs[0, 0].set_title('Voltage Error (%)', fontweight='bold', pad=25)
    axs[0, 0].set_ylabel('Error (%)')
    axs[0, 0].legend(loc='best')
    
    axs[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Target Met (0)')
    axs[0, 1].axhspan(ymin=-0.2, ymax=0, color='mediumseagreen', alpha=0.1, zorder=0)
    axs[0, 1].set_title('Efficiency Error (Target - Actual)', fontweight='bold', pad=25)
    axs[0, 1].set_ylabel('Efficiency Error')
    axs[0, 1].legend(loc='best')
    
    axs[1, 0].axhline(y=9000, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Burn Threshold')
    axs[1, 0].axhspan(ymin=9000, ymax=10000, color='crimson', alpha=0.15, zorder=0)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylim(bottom=1.0, top=10000.0)
    axs[1, 0].set_title('Total Volume', fontweight='bold', pad=25)
    axs[1, 0].set_ylabel('Volume (cm³)')
    axs[1, 0].legend(loc='best')
    
    axs[1, 1].set_title('Total Components', fontweight='bold', pad=25)
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend(loc='best')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

def plot_all_benchmarks():
    extracted = extract_benchmark_data()
    if not extracted: return
    data, task_labels = extracted
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')

    x_positions = np.arange(len(task_labels))
    styles = {
        "Base":    {"color": "gray", "marker": "o", "offset": -0.22},
        "SFT":     {"color": "darkorange", "marker": "s", "offset": 0.0},
        "Zycos10": {"color": "seagreen", "marker": "^", "offset": 0.22}
    }
    
    def save_plot(fig, filename):
        out_path = os.path.join(DATA_DIR, filename)
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"✅ Generated: {filename}")
        except PermissionError:
            print(f"❌ PERMISSION ERROR: {filename}")
        plt.close(fig)

    # 1. Line Plot with Shaded Mean/Std
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 10))
    def plot_mean_std(ax, metric_key):
        for model, m_data in data.items():
            means, stds = [], []
            for t_idx in range(len(task_labels)):
                vals = m_data[t_idx][metric_key]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else np.nan)
            
            st = styles[model]
            ax.plot(x_positions, means, label=model, color=st['color'], marker=st['marker'], linewidth=2, markersize=8)
            ax.fill_between(x_positions, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=st['color'], alpha=0.15, edgecolor='none')

    plot_mean_std(axs1[0, 0], 'v_err')
    plot_mean_std(axs1[0, 1], 'eff_err')
    plot_mean_std(axs1[1, 0], 'vol')
    plot_mean_std(axs1[1, 1], 'comps')
    format_benchmark_axes(fig1, axs1, 'Benchmark Performance: Means & Standard Deviation', task_labels)
    save_plot(fig1, "benchmark_summary_1_mean_std.png")

    # 2. SCATTER PLOT
    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
    def plot_scatter(ax, metric_key):
        for model, m_data in data.items():
            st = styles[model]
            for t_idx in range(len(task_labels)):
                vals = m_data[t_idx][metric_key]
                if vals:
                    x_pts = [x_positions[t_idx] + st['offset']] * len(vals)
                    ax.scatter(x_pts, vals, color=st['color'], marker=st['marker'], s=60, alpha=0.8, zorder=3, label=model if t_idx==0 else "")
                    
    plot_scatter(axs2[0, 0], 'v_err')
    plot_scatter(axs2[0, 1], 'eff_err')
    plot_scatter(axs2[1, 0], 'vol')
    plot_scatter(axs2[1, 1], 'comps')
    format_benchmark_axes(fig2, axs2, 'Benchmark Performance: Scatter of Individual Trials', task_labels)
    save_plot(fig2, "benchmark_summary_2_scatter.png")

    # 3. BOX PLOT
    fig3, axs3 = plt.subplots(2, 2, figsize=(16, 10))
    def plot_box(ax, metric_key):
        for model, m_data in data.items():
            st = styles[model]
            for t_idx in range(len(task_labels)):
                vals = m_data[t_idx][metric_key]
                if vals and not all(np.isnan(v) for v in vals):
                    ax.boxplot([vals], positions=[x_positions[t_idx] + st['offset']], widths=0.15,
                               patch_artist=True, boxprops=dict(facecolor=st['color'], color='black', alpha=0.7),
                               medianprops=dict(color='black', linewidth=1.5), whiskerprops=dict(color='black'), capprops=dict(color='black'))
            ax.plot([], [], color=st['color'], marker='s', linestyle='none', label=model)
            
    plot_box(axs3[0, 0], 'v_err')
    plot_box(axs3[0, 1], 'eff_err')
    plot_box(axs3[1, 0], 'vol')
    plot_box(axs3[1, 1], 'comps')
    format_benchmark_axes(fig3, axs3, 'Benchmark Performance: Trial Distributions (Boxplot)', task_labels)
    save_plot(fig3, "benchmark_summary_3_boxplot.png")

    # 4. BAR PLOT
    fig4, axs4 = plt.subplots(2, 2, figsize=(16, 10))
    def plot_bar(ax, metric_key, is_log=False):
        for model, m_data in data.items():
            st = styles[model]
            for t_idx in range(len(task_labels)):
                vals = m_data[t_idx][metric_key]
                if vals:
                    for i, val in enumerate(vals):
                        t_off = -0.05 + (i * 0.1) if len(vals) > 1 else 0
                        x_pos = x_positions[t_idx] + st['offset'] + t_off
                        bottom_val = 1.0 if is_log else 0.0
                        bar_height = max(1e-9, val - bottom_val) if is_log else val
                        ax.bar(x_pos, bar_height, bottom=bottom_val, width=0.09, 
                               color=st['color'], edgecolor='black', alpha=0.8, zorder=3,
                               label=model if (t_idx == 0 and i == 0) else "")
                               
    plot_bar(axs4[0, 0], 'v_err')
    plot_bar(axs4[0, 1], 'eff_err')
    plot_bar(axs4[1, 0], 'vol', is_log=True)
    plot_bar(axs4[1, 1], 'comps')
    format_benchmark_axes(fig4, axs4, 'Benchmark Performance: Exact Trial Values (Barplot)', task_labels)
    save_plot(fig4, "benchmark_summary_4_barplot.png")

if __name__ == "__main__":
    plot_all_benchmarks()