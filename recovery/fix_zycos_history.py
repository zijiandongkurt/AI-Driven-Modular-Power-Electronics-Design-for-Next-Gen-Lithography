import os
import sys
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import your original plotting scripts safely
try:
    from pipeline.graphs_and_visualizations.Visualize_demo_results import plot_run_results
except ImportError:
    plot_run_results = None

try:
    from experiments.plot_combined_constraint_satisfaction import plot_combined_sessions
except ImportError:
    try:
        from experiments.plot_combined_constraint_satisfaction import plot_combined_sessions
    except ImportError:
        plot_combined_sessions = None

from pipeline.utility.topology_hasher import get_topological_hash

def get_active_constraints(run_dir: Path):
    c_file = run_dir / "active_constraint.json"
    if c_file.exists():
        with open(c_file, "r", encoding="utf-8") as f:
            return json.load(f).get("active_constraints", {})
    return {}

def write_best_netlist(run_dir: Path, best_entry: dict, constraints: dict, output_path: Path):
    cand_id = best_entry.get("netlist_id")
    if not cand_id: return False
    
    batch_folder = best_entry["batch_id"].split('/')[-1]
    
    source_net = run_dir / batch_folder / "LLM_output" / f"{cand_id}.net"
    if not source_net.exists():
        source_net = run_dir / batch_folder / f"{cand_id}.net"
        
    if source_net.exists():
        original_text = source_net.read_text(encoding="utf-8")
        
        header = f"* === GLOBAL BEST TOPOLOGY ===\n"
        header += f"* Candidate: {cand_id}\n"
        header += f"* Fitness: {best_entry.get('fitness', 0.0):.4f}\n"
        
        header += f"*\n* --- Active Constraints ---\n"
        for k, v in constraints.items():
            header += f"* {k}: {v}\n"
            
        header += f"*\n* --- Raw Metrics & Components ---\n"
        metrics = best_entry.get("metrics", {})
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    header += f"* {k}: {v:.6f}\n"
                else:
                    header += f"* {k}: {v}\n"
        else:
            header += f"* No raw metrics available.\n"
            
        header += f"* ============================\n\n"
        
        output_path.write_text(header + original_text, encoding="utf-8")
        return True
    return False

def generate_zycos_master_summary(zycos_dir: Path, session_runs: list):
    master_text = f"=== GLOBAL TRAINING SUMMARY: {zycos_dir.name} ===\n"
    master_text += f"Total Runs Completed: {len(session_runs)}\n\n"
    
    global_best_fit = -float('inf')
    global_best_cand = "None"
    global_best_run = "None"
    
    for r_name, r_best_fit, r_best_cand in session_runs:
        master_text += f"Run: {r_name} | Best Fitness: {r_best_fit:.4f} ({r_best_cand})\n"
        if r_best_fit > global_best_fit:
            global_best_fit = r_best_fit
            global_best_cand = r_best_cand
            global_best_run = r_name
            
    master_text += f"\n--- OVERALL SESSION CHAMPION ---\n"
    master_text += f"Run: {global_best_run}\n"
    master_text += f"Candidate: {global_best_cand}\n"
    master_text += f"Score: {global_best_fit:.4f}\n"
    
    (zycos_dir / "training_run_summary.txt").write_text(master_text, encoding="utf-8")

def main():
    global_start_time = time.time()
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    legacy_dir = PROJECT_ROOT / "pipeline" / "data_legacy_broken"
    master_results_dir = data_dir / "results"
    master_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n RETROACTIVELY FIXING ZYCOS HISTORY & TOPOLOGIES\n{'='*70}\n")

    global_training_best = {"fitness": -float('inf'), "entry": None, "constraints": {}, "run_dir": None}
    
    # 🎯 TARGET FILTER: ONLY RUN ON ZYCOS 8, 9, 10
    target_sessions = ["zycos_008", "zycos_009", "zycos_010"]
    zycos_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name in target_sessions])
    
    # --- PRE-CALCULATE TOTAL RUNS FOR ETA ---
    total_runs = 0
    for z_dir in zycos_folders:
        total_runs += len([d for d in z_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")])
        
    runs_completed = 0
    print(f"Total Runs to Process: {total_runs}\n")

    for zycos_dir in zycos_folders:
        session_start_time = time.time()
        print(f"📂 Processing Training Session: {zycos_dir.name}...")
        session_runs = []
        run_folders = sorted([d for d in zycos_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")])
        
        for run_dir in run_folders:
            print(f"  [{run_dir.name}]")
            run_start_time = time.time()
            
            legacy_run_dir = legacy_dir / zycos_dir.name / run_dir.name
            legacy_history_path = legacy_run_dir / "history_db.json"
            
            if not legacy_history_path.exists(): 
                print(f"    ⚠️ No legacy history found. Skipping.")
                runs_completed += 1
                continue
                
            with open(legacy_history_path, "r", encoding="utf-8") as f:
                old_history = json.load(f)
                
            new_history = []
            
            # --- 1. REPAIR HISTORY BY DIRECT SEQUENTIAL MAPPING ---
            t0 = time.time()
            batches_in_history = {}
            for entry in old_history:
                b_id = entry.get("batch_id")
                if not b_id: continue
                b_folder = b_id.split('/')[-1]
                if b_folder not in batches_in_history:
                    batches_in_history[b_folder] = []
                batches_in_history[b_folder].append(entry)

            for batch_folder, old_entries in batches_in_history.items():
                new_b_dir = run_dir / batch_folder
                new_rew_file = new_b_dir / "reward_results.json"
                
                if not new_rew_file.exists():
                    continue 
                    
                with open(new_rew_file, 'r', encoding="utf-8") as f:
                    new_circs = json.load(f).get("circuits", {})
                    
                new_keys = sorted(new_circs.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                
                for i, entry in enumerate(old_entries):
                    if i < len(new_keys):
                        new_id = new_keys[i]
                        entry["netlist_id"] = new_id
                        
                        c_data = new_circs[new_id]
                        entry["fitness"] = c_data.get("fitness_score", entry.get("fitness"))
                        entry["metrics"] = c_data.get("raw_metrics", {})
                        
                        net_file = new_b_dir / "LLM_output" / f"{new_id}.net"
                        if not net_file.exists():
                            net_file = new_b_dir / f"{new_id}.net"
                        
                        if net_file.exists():
                            net_text = net_file.read_text(encoding="utf-8")
                            entry["topo_hash"] = get_topological_hash(net_text)
                            
                        new_history.append(entry)
                
            if new_history:
                with open(run_dir / "history_db.json", "w", encoding="utf-8") as f:
                    json.dump(new_history, f, indent=4)
            
            t1 = time.time()
            #print(f"    ⏱️  History Sync & Graph Hashing: {t1 - t0:.2f}s")
            
            if not new_history: 
                runs_completed += 1
                continue
            
            # --- 2. GENERATE BEST TOPOLOGIES & SUMMARIES ---
            t1 = time.time()
            best_entry = max(new_history, key=lambda x: float(x.get("fitness", -9999)))
            constraints = get_active_constraints(run_dir)
            
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            written = write_best_netlist(run_dir, best_entry, constraints, results_dir / "best_topology.net")
            
            if written:
                summary_text = f"=== RUN SUMMARY: {run_dir.name} ===\nBest Fitness: {best_entry.get('fitness', 0):.4f}\nCandidate: {best_entry.get('netlist_id')}\n"
                (results_dir / "run_summary.txt").write_text(summary_text, encoding="utf-8")
                
                session_runs.append((run_dir.name, best_entry.get('fitness', 0), best_entry.get('netlist_id')))

                if zycos_dir.name in ["zycos_008", "zycos_009", "zycos_010"]:
                    if best_entry.get("fitness", -9999) > global_training_best["fitness"]:
                        global_training_best = {"fitness": best_entry["fitness"], "entry": best_entry, "constraints": constraints, "run_dir": run_dir}
            t2 = time.time()
            #print(f"    ⏱️  Summaries & File Export:      {t2 - t1:.2f}s")

            # --- 3. REPLOT THE RUN ---
            t2 = time.time()
            if plot_run_results: 
                plot_run_results(str(run_dir))
            t3 = time.time()
            #print(f"    ⏱️  Matplotlib Rendering:         {t3 - t2:.2f}s")
            
            # --- TRUE GLOBAL ETA CALCULATION ---
            runs_completed += 1
            elapsed_total = time.time() - global_start_time
            avg_time_per_run = elapsed_total / runs_completed
            runs_remaining = total_runs - runs_completed
            eta_seconds = runs_remaining * avg_time_per_run
            projected_end_time = datetime.now() + timedelta(seconds=eta_seconds)

            print(f"    ✅ Time for Run:                 {t3 - run_start_time:.2f}s")
            print(f"    📊 Progress: {runs_completed}/{total_runs} Runs | Global Avg: {avg_time_per_run:.1f}s/run")
            print(f"    ⏳ ETA: {projected_end_time.strftime('%H:%M:%S')}\n")

        if session_runs:
            generate_zycos_master_summary(zycos_dir, session_runs)
            
        print(f"🎉 Completed {zycos_dir.name} in {time.time() - session_start_time:.1f}s\n")

    # --- 4. GLOBAL TRAINING BEST ---
    print(f"\n{'='*70}\n EXPORTING GLOBAL CHAMPIONS TO {master_results_dir}\n{'='*70}")
    
    if global_training_best["entry"]:
        write_best_netlist(global_training_best["run_dir"], global_training_best["entry"], global_training_best["constraints"], master_results_dir / "best_training_netlist.net")
        print(f"🥇 Global Training Best: {global_training_best['fitness']:.4f} (From {global_training_best['run_dir'].parent.name}/{global_training_best['run_dir'].name})")

    # --- 5. COMBINED PLOT ---
    print(f"\nGenerating Combined Constraint Satisfaction Plot...")
    t_comb = time.time()
    if plot_combined_sessions:
        try:
            plot_combined_sessions()
            print(f"✅ Combined Plot Generated in {time.time() - t_comb:.2f}s")
        except Exception as e:
            print(f"❌ Error generating combined plot: {e}")
            
    print(f"\n{'='*70}\n RETROACTIVE FIX COMPLETE IN {time.time() - global_start_time:.1f}s!\n{'='*70}\n")

if __name__ == "__main__":
    main()