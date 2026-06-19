import os
import sys
import json
from pathlib import Path

# Add project root to path so we can import your pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import your original plotting scripts directly!
try:
    from pipeline.graphs_and_visualizations.Visualize_demo_results import plot_run_results
except ImportError as e:
    print(f"⚠️ Warning: Could not import Visualize_demo_results: {e}")
    plot_run_results = None

try:
    from experiments.plot_combined_constraint_satisfaction import plot_combined_sessions
except ImportError as e:
    print(f"⚠️ Warning: Could not import plot_combined_constraint_satisfaction: {e}")
    plot_combined_sessions = None


def get_active_constraints(run_dir: Path):
    """Safely loads the active constraints for a run."""
    c_file = run_dir / "active_constraint.json"
    if c_file.exists():
        with open(c_file, "r", encoding="utf-8") as f:
            return json.load(f).get("active_constraints", {})
    return {}

def build_or_update_history(run_dir: Path, is_zycos: bool, legacy_dir: Path):
    """
    Creates or updates history_db.json using the NEW reward_results.json data.
    For Zycos, it borrows the lineage (depth, parent_id, topo_hash) from the legacy folder.
    For Benchmarks, it dynamically builds it chronologically.
    """
    new_history = []
    
    if is_zycos:
        # Read from legacy data to get structural history
        zycos_name = run_dir.parent.name
        legacy_history_path = legacy_dir / zycos_name / run_dir.name / "history_db.json"
        
        if legacy_history_path.exists():
            with open(legacy_history_path, "r", encoding="utf-8") as f:
                old_history = json.load(f)
            
            for entry in old_history:
                batch_id = entry.get("batch_id")
                cand_id = entry.get("netlist_id")
                if not batch_id or not cand_id: continue
                
                # Update fitness/metrics from the newly simulated reward file
                batch_folder = batch_id.split('/')[-1]
                reward_file = run_dir / batch_folder / "reward_results.json"
                
                if reward_file.exists():
                    with open(reward_file, "r", encoding="utf-8") as f:
                        rewards = json.load(f)
                    
                    c_data = rewards.get("circuits", {}).get(cand_id, {})
                    entry["fitness"] = c_data.get("fitness_score", entry.get("fitness"))
                    entry["metrics"] = c_data.get("raw_metrics", entry.get("metrics", {}))
                
                new_history.append(entry)

    # Build dynamically if Benchmark (or if legacy history was somehow missing)
    if not new_history:
        batch_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                               key=lambda x: int(x.name.split('_')[1]))
        
        for batch_dir in batch_folders:
            reward_file = batch_dir / "reward_results.json"
            if reward_file.exists():
                with open(reward_file, "r", encoding="utf-8") as f:
                    rewards = json.load(f)
                    
                for cand_id, c_data in rewards.get("circuits", {}).items():
                    fit = c_data.get("fitness_score")
                    if fit is not None:
                        # Reconstruct basic tracking
                        new_history.append({
                            "netlist_id": cand_id,
                            "batch_id": f"{run_dir.parent.name}/{run_dir.name}/{batch_dir.name}" if is_zycos else f"{run_dir.name}/{batch_dir.name}",
                            "fitness": float(fit),
                            "metrics": c_data.get("raw_metrics", {})
                        })

    # Save exactly as history_db.json so existing plotting scripts find it naturally
    if new_history:
        out_path = run_dir / "history_db.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_history, f, indent=4)
            
    return new_history

def write_best_netlist(run_dir: Path, best_entry: dict, constraints: dict, output_path: Path):
    """Finds the raw .net file and copies it with constraint headers, metrics, and components."""
    cand_id = best_entry["netlist_id"]
    batch_folder = best_entry["batch_id"].split('/')[-1]
    
    source_net = run_dir / batch_folder / "LLM_output" / f"{cand_id}.net"
    
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
    """Aggregates all Runs in a zycos_XXX folder into one master summary."""
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
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    legacy_dir = PROJECT_ROOT / "pipeline" / "data_legacy_broken"
    master_results_dir = data_dir / "results"
    master_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n STARTING MASTER AGGREGATION & EXTRACTION\n{'='*70}\n")

    global_training_best = {"fitness": -float('inf'), "entry": None, "constraints": {}, "run_dir": None}
    global_bench_best = {
        "Base": {"fitness": -float('inf'), "entry": None, "constraints": {}, "run_dir": None},
        "SFT": {"fitness": -float('inf'), "entry": None, "constraints": {}, "run_dir": None},
        "Zycos10": {"fitness": -float('inf'), "entry": None, "constraints": {}, "run_dir": None}
    }

    # ---------------------------------------------------------
    # Process Benchmarks (Depth 1)
    # ---------------------------------------------------------
    bench_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Bench_")])
    for bench_dir in bench_folders:
        print(f"Processing Benchmark: {bench_dir.name}...")
        history = build_or_update_history(bench_dir, is_zycos=False, legacy_dir=legacy_dir)
        if not history: continue
            
        best_entry = max(history, key=lambda x: x["fitness"])
        constraints = get_active_constraints(bench_dir)
        
        # 1. Local Summaries
        results_dir = bench_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        write_best_netlist(bench_dir, best_entry, constraints, results_dir / "best_topology.net")
        
        summary_text = f"=== RUN SUMMARY: {bench_dir.name} ===\nBest Fitness: {best_entry['fitness']:.4f}\nCandidate: {best_entry['netlist_id']}\n"
        (results_dir / "run_summary.txt").write_text(summary_text, encoding="utf-8")
        
        # 2. Trigger native plots
        if plot_run_results: 
            plot_run_results(str(bench_dir))

        # 3. Global Tracking
        try:
            model_name = bench_dir.name.split('_')[1] # e.g., 'Base' from 'Bench_Base_...'
            if model_name in global_bench_best:
                if best_entry["fitness"] > global_bench_best[model_name]["fitness"]:
                    global_bench_best[model_name] = {"fitness": best_entry["fitness"], "entry": best_entry, "constraints": constraints, "run_dir": bench_dir}
        except IndexError:
            pass

    # ---------------------------------------------------------
    # Process Zycos (Depth 2)
    # ---------------------------------------------------------
    zycos_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("zycos_")])
    for zycos_dir in zycos_folders:
        print(f"Processing Training Session: {zycos_dir.name}...")
        session_runs = []
        run_folders = sorted([d for d in zycos_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")])
        
        for run_dir in run_folders:
            history = build_or_update_history(run_dir, is_zycos=True, legacy_dir=legacy_dir)
            if not history: continue
                
            best_entry = max(history, key=lambda x: x["fitness"])
            constraints = get_active_constraints(run_dir)
            
            # 1. Local Summaries
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            write_best_netlist(run_dir, best_entry, constraints, results_dir / "best_topology.net")
            
            summary_text = f"=== RUN SUMMARY: {run_dir.name} ===\nBest Fitness: {best_entry['fitness']:.4f}\nCandidate: {best_entry['netlist_id']}\n"
            (results_dir / "run_summary.txt").write_text(summary_text, encoding="utf-8")
            
            # 2. Trigger native plots
            if plot_run_results: 
                plot_run_results(str(run_dir))
                
            session_runs.append((run_dir.name, best_entry['fitness'], best_entry['netlist_id']))

            # 3. Global Tracking (Only for 8, 9, 10 as requested)
            if zycos_dir.name in ["zycos_008", "zycos_009", "zycos_010"]:
                if best_entry["fitness"] > global_training_best["fitness"]:
                    global_training_best = {"fitness": best_entry["fitness"], "entry": best_entry, "constraints": constraints, "run_dir": run_dir}

        # Session-level summary
        if session_runs:
            generate_zycos_master_summary(zycos_dir, session_runs)

    # ---------------------------------------------------------
    # Export Global Champions
    # ---------------------------------------------------------
    print(f"\n{'='*70}\n EXPORTING GLOBAL CHAMPIONS TO {master_results_dir}\n{'='*70}")
    
    if global_training_best["entry"]:
        write_best_netlist(global_training_best["run_dir"], global_training_best["entry"], global_training_best["constraints"], master_results_dir / "best_training_netlist.net")
        print(f"🥇 Global Training Best: {global_training_best['fitness']:.4f} (From {global_training_best['run_dir'].parent.name}/{global_training_best['run_dir'].name})")
        
    for model, data in global_bench_best.items():
        if data["entry"]:
            file_name = f"best_benchmark_{model}.net"
            write_best_netlist(data["run_dir"], data["entry"], data["constraints"], master_results_dir / file_name)
            print(f"🏆 Benchmark Best [{model}]: {data['fitness']:.4f} (From {data['run_dir'].name})")

    # ---------------------------------------------------------
    # Generate Combined Constraint Plot
    # ---------------------------------------------------------
    print(f"\nGenerating Combined Constraint Satisfaction Plot...")
    if plot_combined_sessions:
        try:
            plot_combined_sessions()
            print("✅ Combined Plot Generated directly via original script.")
        except Exception as e:
            print(f"❌ Error generating combined plot: {e}")
    else:
        print("⚠️ Skipped: Original combined plotting script was not found/imported.")

    print(f"\n{'='*70}\n MASTER AGGREGATION COMPLETE!\n{'='*70}\n")

if __name__ == "__main__":
    main()