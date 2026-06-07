import sys
import numpy as np
from pathlib import Path

# 1. Setup paths so it can find the pipeline folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 2. Import the plotting modules we just separated
try:
    from pipeline.graphs_and_visualizations.plot_validity_bar_chart import plot_validity_bar_chart
    from pipeline.graphs_and_visualizations.plot_radar_chart import plot_radar_chart
    from pipeline.graphs_and_visualizations.plot_pareto_scatter import plot_pareto_scatter
    from pipeline.graphs_and_visualizations.plot_learning_curves import plot_learning_curves
    from pipeline.graphs_and_visualizations.plot_combined_fitness import plot_combined_fitness
    print("✅ All plotting modules imported successfully.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def generate_mock_trial(model, phase_power):
    """Generates realistic fake data based on the SFT briefing and project specs."""
    is_base = model == "Base"
    is_sft = model == "SFT"
    is_rl = model == "RL"
    
    if is_base:
        validity = np.random.uniform(60.0, 70.0) # ~65% from briefing
    elif is_sft:
        validity = np.random.uniform(70.0, 80.0) # ~75% from briefing
    else:
        validity = np.random.uniform(98.0, 100.0) # 100% for RL
        
    # All models start with at least 1 valid topology, so fitness >= 0.5
    curve_start = np.random.uniform(0.50, 0.55)
    
    if is_base:
        fit = np.random.uniform(0.55, 0.65)
        v_err = np.random.uniform(20.0, 45.0)
        eff = np.random.uniform(40, 65)
        auc_val = np.random.uniform(0.5, 0.6)
        components = np.random.randint(15, 25)
    elif is_sft:
        fit = np.random.uniform(0.70, 0.85)
        v_err = np.random.uniform(5.0, 20.0)
        eff = np.random.uniform(65, 85)
        auc_val = np.random.uniform(0.65, 0.8)
        components = np.random.randint(8, 15)
    else: # RL
        fit = np.random.uniform(0.90, 0.99)
        v_err = np.random.uniform(0.1, 4.0)
        eff = np.random.uniform(90, 99)
        auc_val = np.random.uniform(0.85, 0.98)
        components = np.random.randint(5, 10)
    
    # Interpolate from ~0.5 to final fitness, add slight variance
    curve = np.linspace(curve_start, fit, 15) + np.random.normal(0, 0.02, 15)
    
    # Strictly enforce that no point drops below 0.5
    curve[0] = max(0.5, curve[0]) 
    curve = np.clip(curve, 0.5, 1.0).tolist()
    
    return {
        "fitness": fit,
        "auc": auc_val,
        "validity": validity,
        "learning_curve": curve,
        "v_error_pct": v_err,
        "efficiency": eff,
        "volume": phase_power / np.random.uniform(0.5, 3.0),
        "components": components,
        "target_power": phase_power
    }

def main():
    print("\n" + "="*50)
    print("🧪 RUNNING MOCK DATA PLOTTING TEST")
    print("="*50)

    # 3. Define our fake benchmark scope
    models = ["Base", "SFT", "RL"]
    tasks = {
        "P1_Buck_Std": 15.0, 
        "P2_Buck_HighPwr": 150.0, 
        "P3_Buck_Mains": 600.0
    }
    
    # 4. Build the master_results dictionary
    master_results = {model: {task: [] for task in tasks.keys()} for model in models}
    
    for model in models:
        for task, power in tasks.items():
            for _ in range(3): # 3 Trials per task
                master_results[model][task].append(generate_mock_trial(model, power))
                
    output_directory = str(PROJECT_ROOT / "experiments" / "test_plot_outputs")
    
    # 5. Fire the plotters!
    try:
        plot_combined_fitness(master_results, output_dir=output_directory)
        plot_learning_curves(master_results, output_dir=output_directory, n_batches=15)
        plot_pareto_scatter(master_results, output_dir=output_directory)
        plot_radar_chart(master_results, output_dir=output_directory)
        plot_validity_bar_chart(master_results, output_dir=output_directory)
        print(f"\n🎉 SUCCESS! All plots generated perfectly.")
        print(f"Check the folder: {output_directory}")
    except Exception as e:
        print(f"\n❌ FAILED during plotting execution: {e}")

if __name__ == "__main__":
    main()