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
    """Generates realistic fake data based on the model type."""
    is_base = model == "Base"
    is_rl = model == "RL"
    
    # Base model fails more often, RL almost never fails
    if is_base and np.random.rand() < 0.4:
        return {"fitness": -0.6, "auc": 0.0, "validity": 0.0, "learning_curve": [-0.6]*15, 
                "v_error_pct": None, "efficiency": None, "volume": None, "components": None, "target_power": phase_power}
        
    v_err = np.random.uniform(0.5, 5.0) if is_rl else np.random.uniform(10.0, 45.0)
    eff = np.random.uniform(85, 98) if is_rl else np.random.uniform(40, 75)
    fit = np.random.uniform(0.85, 0.99) if is_rl else np.random.uniform(0.5, 0.75)
    
    curve = np.linspace(-0.6, fit, 15) + np.random.normal(0, 0.05, 15)
    curve = np.clip(curve, -0.6, 1.0).tolist()
    
    return {
        "fitness": fit,
        "auc": np.random.uniform(0.4, 0.9),
        "validity": np.random.uniform(80, 100) if is_rl else np.random.uniform(20, 60),
        "learning_curve": curve,
        "v_error_pct": v_err,
        "efficiency": eff,
        "volume": phase_power / np.random.uniform(0.5, 3.0),
        "components": np.random.randint(5, 12) if is_rl else np.random.randint(10, 25),
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