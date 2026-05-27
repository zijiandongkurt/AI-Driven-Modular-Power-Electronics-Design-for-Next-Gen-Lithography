import copy
from demo_inference import run_inference

def main():
    # 1. Define the universal hyperparameter base
    # (Reduced batches slightly so the 18 total runs don't take a week!)
    base_config = {
        "run_settings": {
            "n_batches": 15,
            "sampled_states_per_batch": 2,
            "candidates_per_prompt": 4,
            "max_tokens": 2048,
            "update_plots_per_batch": False, # Keep false to speed up the loop
            "model_id": "",
            "run_prefix": "" 
        },
        "mcts_settings": {
            "temperature": 0.05,
            "top_k": 15,
            "epsilon": 0.15
        },
        "constraint_settings": {
            "dataset_path": "",
            "index": 0,
            "phase": ""
        },
        "weights": {
            "v_out": 10.0, "efficiency": 20.0,
            "volume": 2.0, "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
        }
    }

    # 2. Define the Models to compare
    models = {
        "Base": "Qwen/Qwen2.5-3B-Instruct"
        #"FineTuned": "path/to/your/finetuned/model"  # <--- UPDATE THIS
    }

    # 3. Define the Benchmark Gauntlet
    tasks = [
        {"phase": "Phase1", "label": "P1_Buck_Standard",  "file": "pipeline/data/datasets/constraints_easy.json",   "idx": 0},
        {"phase": "Phase1", "label": "P1_Boost_Standard", "file": "pipeline/data/datasets/constraints_easy.json",   "idx": 6},
        {"phase": "Phase1", "label": "P1_Buck_RatioLim",  "file": "pipeline/data/datasets/constraints_easy.json",   "idx": 16},
        
        {"phase": "Phase2", "label": "P2_Buck_Extreme",   "file": "pipeline/data/datasets/constraints_medium.json", "idx": 0},
        {"phase": "Phase2", "label": "P2_Buck_HighPwr",   "file": "pipeline/data/datasets/constraints_medium.json", "idx": 10},
        {"phase": "Phase2", "label": "P2_Boost_Extreme",  "file": "pipeline/data/datasets/constraints_medium.json", "idx": 19},
        
        {"phase": "Phase3", "label": "P3_Buck_Mains",     "file": "pipeline/data/datasets/constraints_hard.json",   "idx": 0},
        {"phase": "Phase3", "label": "P3_Boost_Extreme",  "file": "pipeline/data/datasets/constraints_hard.json",   "idx": 1},
        {"phase": "Phase3", "label": "P3_Buck_MaxPwr",    "file": "pipeline/data/datasets/constraints_hard.json",   "idx": 10},
    ]

    results_tracker = []

    print(f"\n{'='*70}")
    print(f"🚀 INITIATING FULL CAPABILITY BENCHMARK")
    print(f"Testing {len(models)} Models across {len(tasks)} Constraints ({len(models) * len(tasks)} Total Runs)")
    print(f"{'='*70}\n")

    # 4. Execute the Nested Loop
    for model_name, model_path in models.items():
        print(f"\n\n{'*'*60}")
        print(f"🧠 LOADING MODEL: {model_name} ({model_path})")
        print(f"{'*'*60}")

        for task in tasks:
            print(f"\n--- STARTING TASK: {task['label']} ---")
            
            # Construct the specific config for this run
            run_config = copy.deepcopy(base_config)
            
            run_config["run_settings"]["model_id"] = model_path
            run_config["run_settings"]["run_prefix"] = f"Bench_{model_name}_{task['label']}"
            
            run_config["constraint_settings"]["dataset_path"] = task["file"]
            run_config["constraint_settings"]["index"] = task["idx"]
            run_config["constraint_settings"]["phase"] = task["phase"]

            # Run the pipeline and catch the output folder path
            try:
                output_folder = run_inference(run_config)
                results_tracker.append((model_name, task['label'], output_folder, "SUCCESS"))
            except Exception as e:
                print(f"❌ ERROR ON TASK {task['label']}: {e}")
                results_tracker.append((model_name, task['label'], "N/A", f"FAILED: {e}"))

    # 5. Print the Final Report
    print(f"\n\n{'='*70}")
    print(f"🏆 BENCHMARK COMPLETE 🏆")
    print(f"{'='*70}")
    print(f"{'Model':<12} | {'Task':<18} | {'Status':<8} | {'Output Directory'}")
    print("-" * 70)
    for res in results_tracker:
        print(f"{res[0]:<12} | {res[1]:<18} | {res[3]:<8} | {res[2]}")
    
    print("\nBenchmark finished. Compare the 'run_summary.txt' files in the directories above to evaluate performance!")

if __name__ == "__main__":
    main()