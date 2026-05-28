import copy
from run_inference import run_inference

def main():
    # 1. Define the universal config dict
    base_config = {
        "run_settings": {
            "n_batches": 15,
            "sampled_states_per_batch": 2,
            "candidates_per_prompt": 4,
            "max_tokens": 2048,
            "update_plots_per_batch": False, # Faster to only plot at the end for benchmarks
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "run_prefix": "Bench_Base" 
        },
        "mcts_settings": {
            "temperature": 0.05,
            "top_k": 15,
            "epsilon": 0.15
        },
        "constraint_settings": {
            "dataset_path": "pipeline/data/datasets/constraints.json",
            "index": 4,
            "phase": "Phase1"
        },
        "weights": {
            "v_out": 10.0, 
            "efficiency": 20.0,
            "volume": 2.0, 
            "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
        }
    }

    print("\n" + "="*60)
    print("🚀 PHASE 1: RUNNING BASE MODEL BENCHMARK")
    print("="*60)
    base_folder = run_inference(base_config)

    # 2. Duplicate config and swap the model ID to your Finetuned version
    finetuned_config = copy.deepcopy(base_config)
    
    # --- CHANGE THIS TO YOUR ACTUAL FINETUNED MODEL PATH/HUGGINGFACE ID ---
    finetuned_config["run_settings"]["model_id"] = "path/to/your/finetuned/model"
    finetuned_config["run_settings"]["run_prefix"] = "Bench_FineTuned"

    print("\n" + "="*60)
    print("🚀 PHASE 2: RUNNING FINE-TUNED MODEL BENCHMARK")
    print("="*60)
    finetuned_folder = run_inference(finetuned_config)

    print("\n" + "="*60)
    print("🏆 BENCHMARK COMPLETE")
    print("="*60)
    print(f"Base Model Results saved to:       {base_folder}")
    print(f"Fine-tuned Model Results saved to: {finetuned_folder}")
    print("Check the 'run_summary.txt' inside both folders to compare overall fitness!")

if __name__ == "__main__":
    main()