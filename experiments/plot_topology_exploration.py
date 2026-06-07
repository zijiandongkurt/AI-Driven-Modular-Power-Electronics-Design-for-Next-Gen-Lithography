import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_topologies():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(SCRIPT_DIR, "pure_topology_counts.txt")
    
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')

    if not os.path.exists(txt_path):
        print(f"❌ Could not find {txt_path}. Make sure analyze_pure_topologies.py has been run!")
        return

    data = {}
    current_zycos = None
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match the header like "[ zycos_005 ]"
            zycos_match = re.search(r'\[\s*(zycos_\d+)\s*\]', line)
            if zycos_match:
                current_zycos = zycos_match.group(1)
                data[current_zycos] = {'run_nums': [], 'counts': []}
                continue
            
            # Match the run lines like "-> Run_001: 6 unique structures explored"
            if current_zycos and '-> Run_' in line:
                run_match = re.search(r'Run_(\d+):\s*(\d+)', line)
                if run_match:
                    data[current_zycos]['run_nums'].append(int(run_match.group(1)))
                    data[current_zycos]['counts'].append(int(run_match.group(2)))

    print("📊 Generating Topology Exploration Bar Plots...")
    for zycos_name, runs in data.items():
        if not runs['run_nums']:
            continue
            
        results_dir = os.path.join(DATA_DIR, zycos_name, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # zorder=3 ensures the bars sit on top of the grid lines
        plt.bar(runs['run_nums'], runs['counts'], color='teal', edgecolor='black', linewidth=1.2, alpha=0.85, zorder=3)
        
        plt.title(f"Structural Exploration Space - {zycos_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Run Number", fontsize=12)
        plt.ylabel("Unique Topologies Attempted", fontsize=12)
        
        # zorder=0 pushes the grid to the background
        plt.grid(True, linestyle='--', alpha=0.6, axis='y', zorder=0)
        plt.xticks(runs['run_nums'])
        
        # Force Y-axis to use whole numbers
        y_max = max(runs['counts']) if runs['counts'] else 5
        plt.yticks(range(0, y_max + 2))
        plt.ylim(bottom=0)
        
        out_path = os.path.join(results_dir, "topology_exploration.png")
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved {out_path}")
        except PermissionError:
            print(f"  ❌ PERMISSION ERROR: Close topology_exploration.png in {zycos_name} to allow overwrite.")
        
        plt.close()

if __name__ == "__main__":
    plot_topologies()