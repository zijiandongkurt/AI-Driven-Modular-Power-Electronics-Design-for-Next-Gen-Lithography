import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tailored font settings for full-width LaTeX page integration
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'figure.titlesize': 22
})

def plot_sequential_topologies():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(SCRIPT_DIR, "pure_topology_counts.txt")
    
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')

    if not os.path.exists(txt_path):
        print(f"❌ Could not find {txt_path}. Please ensure analyze_pure_topologies.py has run first!")
        return

    # --- 1. Parse text report for target sessions ---
    target_sessions = ["zycos_008", "zycos_009", "zycos_010"]
    session_data = {s: [] for s in target_sessions}
    current_zycos = None
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            zycos_match = re.search(r'\[\s*(zycos_\d+)\s*\]', line)
            if zycos_match:
                current_zycos = zycos_match.group(1)
                continue
            
            if current_zycos in target_sessions:
                run_match = re.search(r'-> Run_(\d+):\s*(\d+)\s*unique', line)
                if run_match:
                    run_num = int(run_match.group(1))
                    count = int(run_match.group(2))
                    session_data[current_zycos].append((run_num, count))

    # --- 2. Construct Sequential Flat Lists ---
    combined_counts = []
    x_labels = []
    session_boundaries = []
    current_index = 0

    for s in target_sessions:
        # Sort by run number to guarantee chronological sequence
        sorted_runs = sorted(session_data[s], key=lambda x: x[0])
        if not sorted_runs:
            continue
            
        start_idx = current_index
        for run_num, count in sorted_runs:
            combined_counts.append(count)
            x_labels.append(f"R{run_num}")
            current_index += 1
        end_idx = current_index - 1
        session_boundaries.append((s, start_idx, end_idx))

    if not combined_counts:
        print("❌ No valid data extracted for zycos_008, zycos_009, or zycos_010.")
        return

    # --- 3. Build Master Figure ---
    plt.figure(figsize=(18, 7))
    x_positions = list(range(len(combined_counts)))
    
    # Plot the sequential bars
    bars = plt.bar(x_positions, combined_counts, color='cadetblue', edgecolor='black', linewidth=1.0, alpha=0.9, zorder=3)
    
    # Visual stylings for distinct sessions
    shading_colors = ['blue', 'orange', 'green']
    label_mapping = {
        "zycos_008": "zycos_008 (Easy Constraints)",
        "zycos_009": "zycos_009 (Medium Constraints)",
        "zycos_010": "zycos_010 (Hard Constraints)"
    }

    # Draw shaded background blocks and session text headers
    for idx, (session_name, start, end) in enumerate(session_boundaries):
        # alternate backgrounds to visually group 10 runs at a time
        bg_alpha = 0.04 if idx % 2 == 0 else 0.09
        plt.axvspan(start - 0.5, end + 0.5, color='gray', alpha=bg_alpha, zorder=1)
        
        # Draw a subtle vertical divider line between sessions
        if end < len(combined_counts) - 1:
            plt.axvline(x=end + 0.5, color='black', linestyle=':', alpha=0.4, zorder=2)
            
        # Place mid-point text descriptions above the highest bars
        mid_x = (start + end) / 2
        plt.text(mid_x, max(combined_counts) * 1.05, label_mapping[session_name], 
                 ha='center', va='bottom', fontsize=13, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # --- 4. Customizing Aesthetics & Scaling ---
    plt.title("Structural Exploration Blueprint: Relative Topology Shifts Across Difficulty Tiers", fontweight='bold', pad=25)
    plt.xlabel("Sequential Iterations (Runs 1-10 per Difficulty Group)")
    plt.ylabel("Unique Topologies Attempted")
    
    plt.xticks(x_positions, x_labels)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y', zorder=0)
    
    # Standardize the Y scale with generous padding for structural titles
    max_y = max(combined_counts) if combined_counts else 10
    plt.ylim(0, max_y + 3)
    
    # Force y-axis ticks to display simple integers
    plt.yticks(range(0, int(max_y) + 3))

    # --- 5. Export ---
    results_root = os.path.join(DATA_DIR, 'combined_results')
    os.makedirs(results_root, exist_ok=True)
    out_path = os.path.join(results_root, "master_topology_sequential_comparison.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Unified master topology plot generated successfully at: {out_path}")

if __name__ == "__main__":
    plot_sequential_topologies()