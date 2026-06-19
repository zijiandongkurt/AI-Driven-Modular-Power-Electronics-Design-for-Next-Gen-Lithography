import os
import sys
import json
import re
import shutil
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.utility.topology_hasher import get_topological_hash

# Import plotting and table modules
try:
    from pipeline.graphs_and_visualizations.plot_validity_bar_chart import plot_validity_bar_chart
    from pipeline.graphs_and_visualizations.plot_radar_chart import plot_radar_chart
    from pipeline.graphs_and_visualizations.plot_pareto_scatter import plot_pareto_scatter
    from pipeline.graphs_and_visualizations.plot_learning_curves import plot_learning_curves
    from pipeline.graphs_and_visualizations.plot_combined_fitness import plot_combined_fitness
except ImportError as e:
    print(f"⚠️ Warning: Some visualization modules not found. {e}")

try:
    from experiments.plot_benchmark_champions import plot_all_benchmarks
except ImportError:
    plot_all_benchmarks = None

def get_empty_metrics():
    return {
        "fitness": 0.0, "auc": 0.0, "validity": 0.0, "duplicate_rate": 0.0, 
        "valid_uniqueness": 0.0, "learning_curve": [], "v_error_pct": 0.0, "efficiency": 0.0, 
        "volume": 10000.0, "components": 0, "target_power": 10.0,
        "total_time_sec": None, "time_per_netlist_sec": None
    }

def generate_reports(checkpoint_path, results_dir):
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        master_results = json.load(f)
        
    models = list(master_results.keys())
    if not models: return
    tasks = list(master_results[models[0]].keys())
    
    # =========================================================
    # 1. DETAILED BENCHMARK REPORT (Matches run_benchmark.py output)
    # =========================================================
    report_text = f"\n\n{'='*120}\n"
    report_text += f" COMPREHENSIVE BENCHMARK RESULTS (Mean ± StdDev) \n"
    report_text += f"{'='*120}\n"
    
    header = f"{'Task (Constraint)':<22} | {'Metric':<16} | " + " | ".join([f"{m:<18}" for m in models]) + " |"
    report_text += header + "\n"
    report_text += "-" * 120 + "\n"

    display_metrics = [
        ("Max Fitness", "fitness"), 
        ("AUC (Speed)", "auc"), 
        ("Validity %", "validity"), 
        ("Duplicate %", "duplicate_rate"),
        ("Valid Unique %", "valid_uniqueness"),
        ("Time/Netlist (s)", "time_per_netlist_sec")
    ]

    for task in tasks:
        report_text += f"{task:<22} | \n"
        for metric_name, key in display_metrics:
            row = f"{'':<22} | {metric_name:<16} | "
            for model_name in models:
                trial_data = master_results.get(model_name, {}).get(task, [])
                vals = [t.get(key) for t in trial_data if t and t.get(key) is not None]
                if vals:
                    # Format string FIRST, then pad it to guarantee alignment
                    val_str = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
                    row += f"{val_str:<18} | "
                else:
                    row += f"{'ERR':<18} | "
            report_text += row + "\n"
        report_text += "-" * 120 + "\n"

    report_file = Path(results_dir) / "benchmark_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📄 Detailed benchmark report saved to: {report_file}")
    
    # =========================================================
    # 2. AGGREGATED REPORT (Averages across ALL tasks per model)
    # =========================================================
    agg_text = f"\n| {'Model':<10} | {'Max Fitness':<20} | {'AUC':<16} | {'Validity (%)':<20} | {'Duplicate (%)':<20} | {'Valid Uniqueness (%)':<22} |\n"
    agg_text += "|" + "-"*12 + "|" + "-"*22 + "|" + "-"*18 + "|" + "-"*22 + "|" + "-"*22 + "|" + "-"*24 + "|\n"
    
    for model_name, task_dict in master_results.items():
        f_sc, a_sc, v_sc, d_sc, vu_sc = [], [], [], [], []
        for t_name, runs in task_dict.items():
            for run in runs:
                f_sc.append(run.get('fitness', 0))
                a_sc.append(run.get('auc', 0))
                v_sc.append(run.get('validity', 0))
                d_sc.append(run.get('duplicate_rate', 0) or 0)
                vu_sc.append(run.get('valid_uniqueness', 0) or 0)
                
        if f_sc:
            f_str = f"{np.mean(f_sc):.4f} ± {np.std(f_sc):.4f}"
            a_str = f"{np.mean(a_sc):.2f} ± {np.std(a_sc):.2f}"
            v_str = f"{np.mean(v_sc):.2f} ± {np.std(v_sc):.2f}"
            d_str = f"{np.mean(d_sc):.2f} ± {np.std(d_sc):.2f}"
            vu_str = f"{np.mean(vu_sc):.2f} ± {np.std(vu_sc):.2f}"
            
            agg_text += f"| {model_name:<10} | {f_str:<20} | {a_str:<16} | {v_str:<20} | {d_str:<20} | {vu_str:<22} |\n"
    
    print(agg_text)
    
    agg_file = Path(results_dir) / "benchmark_aggregated_report.txt"
    with open(agg_file, "w", encoding="utf-8") as f:
        f.write(agg_text)
    print(f"📄 Aggregated benchmark report saved to: {agg_file}\n")


def main():
    print(f"\n{'='*70}\n STARTING BENCHMARK RECONSTRUCTION\n{'='*70}\n")
    
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    experiments_dir = PROJECT_ROOT / "experiments"
    results_dir = experiments_dir / "benchmark_results"
    legacy_results_dir = experiments_dir / "benchmark_results_legacy"

    # --- 1. QUARANTINE LEGACY DATA ---
    if results_dir.exists() and not legacy_results_dir.exists():
        print("📦 Archiving legacy benchmark results...")
        shutil.move(str(results_dir), str(legacy_results_dir))
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. LOAD LEGACY TIMINGS ---
    legacy_data = {}
    legacy_checkpoint = legacy_results_dir / "benchmark_checkpoint.json"
    if legacy_checkpoint.exists():
        print("🕒 Loaded legacy checkpoint for hardware timings.")
        with open(legacy_checkpoint, "r", encoding="utf-8") as f:
            legacy_data = json.load(f)

    master_results = {}
    bench_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Bench_")])

    # --- 3. REBUILD METRICS FOR EACH RUN ---
    for run_dir in bench_folders:
        match = re.search(r"Bench_([^_]+)_(.+)_T(\d+)_\d+", run_dir.name)
        if not match: continue
        
        model_name, task_name, trial_idx_str = match.groups()
        trial_idx = int(trial_idx_str) - 1
        
        if model_name not in master_results:
            master_results[model_name] = {}
        if task_name not in master_results[model_name]:
            master_results[model_name][task_name] = []
            
        print(f"🔄 Processing {run_dir.name}...")
        metrics = get_empty_metrics()

        # A. Safely recover legacy hardware timings
        if model_name in legacy_data and task_name in legacy_data[model_name]:
            legacy_runs = legacy_data[model_name][task_name]
            if trial_idx < len(legacy_runs):
                metrics["total_time_sec"] = legacy_runs[trial_idx].get("total_time_sec")
                metrics["time_per_netlist_sec"] = legacy_runs[trial_idx].get("time_per_netlist_sec")

        # B & C. Calculate Validity, Duplicates, and Valid Uniqueness Rate
        passed_count = 0
        total_val_count = 0
        unique_valid_hashes = set()
        unique_all_hashes = set()
        total_net_files = 0
        
        for batch_dir in run_dir.iterdir():
            if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"): continue
            
            val_file = batch_dir / "validation_results.json"
            val_data = {}
            if val_file.exists():
                try:
                    with open(val_file, "r", encoding="utf-8") as f:
                        val_data = json.load(f)
                except Exception:
                    pass
            
            for cand_id, val_info in val_data.items():
                total_val_count += 1
                is_valid = val_info.get("passed", False)
                if is_valid:
                    passed_count += 1
                    
                net_path = batch_dir / "LLM_output" / f"{cand_id}.net"
                if not net_path.exists():
                    net_path = batch_dir / f"{cand_id}.net"
                    
                if net_path.exists():
                    total_net_files += 1
                    try:
                        net_text = net_path.read_text(encoding="utf-8")
                        topo_hash = get_topological_hash(net_text)
                        unique_all_hashes.add(topo_hash)
                        if is_valid:
                            unique_valid_hashes.add(topo_hash)
                    except Exception:
                        pass
        
        # Assign Metric Rates
        if total_val_count > 0:
            metrics["validity"] = (passed_count / total_val_count) * 100.0
        if total_net_files > 0:
            metrics["duplicate_rate"] = (1.0 - (len(unique_all_hashes) / total_net_files)) * 100.0
        if passed_count > 0:
            metrics["valid_uniqueness"] = (len(unique_valid_hashes) / passed_count) * 100.0
        else:
            metrics["valid_uniqueness"] = 0.0

        # D. Read History for AUC, Learning Curve, and Max Fitness
        history_file = run_dir / "history_db.json"
        best_cand = None
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            if history:
                best_cand = max(history, key=lambda x: x.get("fitness", -999.0))
                metrics["fitness"] = best_cand.get("fitness", 0.0)
                
                batch_maxes = {}
                for entry in history:
                    b_id = entry.get("batch_id")
                    if not b_id: continue
                    b_num = int(re.search(r'batch_(\d+)', b_id).group(1)) if re.search(r'batch_(\d+)', b_id) else 1
                    batch_maxes[b_num] = max(batch_maxes.get(b_num, -999.0), entry.get("fitness", 0.0))
                
                running_max = -999.0
                learning_curve = []
                for b_num in sorted(batch_maxes.keys()):
                    running_max = max(running_max, batch_maxes[b_num])
                    learning_curve.append(running_max)
                    
                metrics["learning_curve"] = learning_curve
                
                norm_scores = [0.0 if s < 0.5 else (s - 0.5) * 2.0 for s in learning_curve]
                if len(norm_scores) > 1:
                    try:
                        metrics["auc"] = float(np.trapezoid(norm_scores))
                    except AttributeError:
                        metrics["auc"] = float(np.trapz(norm_scores))

        # E. Extract Constraints and write Champion Metrics
        active_constraints = {}
        ac_file = run_dir / "active_constraint.json"
        if ac_file.exists():
            with open(ac_file, "r", encoding="utf-8") as f:
                active_constraints = json.load(f).get("active_constraints", {})

        target_v = active_constraints.get("vout_target", 5.0)
        metrics["target_power"] = active_constraints.get("target_power", 10.0)

        if best_cand:
            raw_v = best_cand.get("metrics", {}).get("simulation_output_voltage", 0.0)
            metrics["efficiency"] = best_cand.get("metrics", {}).get("efficiency", 0.0) * 100.0
            metrics["volume"] = min(10000.0, float(best_cand.get("metrics", {}).get("total_volume_cm3", 10000.0)))
            metrics["components"] = best_cand.get("metrics", {}).get("total_components", 0)
            
            safe_target_v = target_v if target_v != 0 else 1e-6
            metrics["v_error_pct"] = abs(raw_v - target_v) / safe_target_v * 100.0

            champ_data = {
                "target_voltage": target_v,
                "raw_voltage": raw_v,
                "raw_efficiency": metrics["efficiency"] / 100.0, 
                "raw_volume": metrics["volume"],
                "raw_components": metrics["components"],
                "target_power": metrics["target_power"]
            }
            with open(run_dir / "champion_metrics.json", "w", encoding="utf-8") as f:
                json.dump(champ_data, f, indent=4)

        master_results[model_name][task_name].append(metrics)

    # --- 4. EXPORT NEW CHECKPOINT ---
    checkpoint_file = results_dir / "benchmark_checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=4)
    print(f"\n✅ Created brand new checkpoint: {checkpoint_file}")

    # --- 5. TRIGGER DOWNSTREAM REPORTING ---
    print("\n📊 Generating text reports...")
    generate_reports(checkpoint_file, results_dir)
    
    print("📊 Triggering graphical visualization tools...")
    n_batches = len(next(iter(next(iter(master_results.values())).values()))[0].get("learning_curve", [0]*10)) if master_results else 10
    
    try:
        plot_combined_fitness(master_results, output_dir=str(results_dir))
        plot_learning_curves(master_results, output_dir=str(results_dir), n_batches=n_batches)
        plot_pareto_scatter(master_results, output_dir=str(results_dir))
        plot_radar_chart(master_results, output_dir=str(results_dir))
        plot_validity_bar_chart(master_results, output_dir=str(results_dir))
        print("✅ Pipeline plots successfully regenerated.")
    except NameError:
        print("⚠️ Skipped internal pipeline plots (modules not found).")
        
    if plot_all_benchmarks:
        try:
            plot_all_benchmarks()
            print("✅ 2x2 Constraint Champion plots successfully regenerated.")
        except Exception as e:
            print(f"❌ Error generating champion plots: {e}")

    print(f"\n{'='*70}\n BENCHMARK RECONSTRUCTION COMPLETE\n{'='*70}\n")

if __name__ == "__main__":
    main()