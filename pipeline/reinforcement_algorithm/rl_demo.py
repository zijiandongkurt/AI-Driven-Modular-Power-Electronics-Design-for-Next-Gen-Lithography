import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from pipeline.llm_topology_generation.llm_engine_minimal import LLMEngine, Constraint
from pipeline.llm_topology_generation.rl_updater import RLUpdater, RLConfig
from pipeline.llm_topology_generation.net_writer import write_netlists
from pipeline.llm_topology_generation.prompt_input import slug

from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import TopologySimulator
from pipeline.reward_evaluation.reward_function import RewardFunction


def load_rewards(batch_id: str) -> dict:
    reward_path = PROJECT_ROOT / "pipeline" / "data" / batch_id / "reward_results.json"

    if not reward_path.exists():
        raise FileNotFoundError(f"Missing reward file: {reward_path}")

    with reward_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_one_iteration(engine, updater, constraint, constraint_dict, batch_id):
    generations = engine.generate_batch(
        constraint,
        n=2,
        temperature=0.7,
        top_p=0.9,
    )

    prompts = []
    completions = []

    for gen in generations:
        prompts.append(constraint.to_prompt())
        completions.append(gen.netlist)

    label = slug(constraint_dict, 0)

    written_paths = write_netlists(
        netlists=completions,
        constraint=constraint_dict,
        label=label,
        batchID=batch_id,
    )

    val = validator()
    val.validate(batch_id)

    simulator = TopologySimulator()
    simulator.simulate(batch_id)

    reward_fn = RewardFunction()
    reward_fn.process_batch(
        batchID=batch_id,
        constraints=constraint_dict,
        weights={
            "v_out": 10.0,
            "efficiency": 20.0,
            "volume": 2.0,
            "component_cost": 1.0,
            "components": {
                "mosfet": 1.0,
                "diode": 1.0,
                "inductor": 1.0,
                "capacitor": 1.0,
            },
        },
        include_detailed_metrics=True,
    )

    reward_data = load_rewards(batch_id)
    circuits = reward_data.get("circuits", {})

    final_prompts = []
    final_completions = []
    rewards = []

    for path, prompt, completion in zip(written_paths, prompts, completions):
        topology_id = path.stem

        if topology_id not in circuits:
            print(f"Skipping {topology_id}: no reward found.")
            continue

        final_prompts.append(prompt)
        final_completions.append(completion)
        rewards.append(float(circuits[topology_id]["fitness_score"]))

    if not rewards:
        print(f"No valid rewards for {batch_id}. Skipping RL update.")
        return None

    metrics = updater.update(
        prompts=final_prompts,
        completions=final_completions,
        rewards=rewards,
    )

    return metrics

    

def main():
    constraint_dict = {
        "vin": 12,
        "vout_target": 5,
        "efficiency_target": 0.90,
        "converter_type": "buck",
        "power_out_w": 120,
        "component_preference": "minimal",
    }

    constraint = Constraint.from_dict(constraint_dict)

    # 1. Load LLM only once
    engine = LLMEngine(
        model_id="Qwen/Qwen2.5-Coder-7B",
        quantization="4bit",
        max_new_tokens=64,
    )

    # 2. Create RL updater only once
    updater = RLUpdater(
        engine,
        RLConfig(
            learning_rate=1e-5,
            kl_beta=0.0,
            save_every=5,
            lora_r=8,
            lora_alpha=16,
        )
    )

    # 3. Run multiple RL iterations
    num_iterations = 2

    for step in range(num_iterations):
        batch_id = f"batch_rl_{step + 1}"

        print(f"\n=== RL Iteration {step + 1}/{num_iterations} ===")

        metrics = run_one_iteration(
            engine=engine,
            updater=updater,
            constraint=constraint,
            constraint_dict=constraint_dict,
            batch_id=batch_id,
        )

        if metrics is None:
            continue

        print("=== RL Update Metrics ===")
        print(json.dumps(metrics, indent=2))

    # 4. Save final LoRA adapter after all iterations
    updater.save("./checkpoints/grpo-lora/final")

    print("=== RL Training Loop Done ===")
    

if __name__ == "__main__":
    main()