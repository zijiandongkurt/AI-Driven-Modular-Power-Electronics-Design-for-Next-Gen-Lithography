import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from pipeline.llm_topology_generation.llm_engine_minimal import LLMEngine, Constraint
from pipeline.llm_topology_generation.rl_updater import RLUpdater, RLConfig

#This is a temporary replacement for file grpo_trainer.py

def fake_simulator_reward(netlist: str) -> float:
    """
    Temporary fake reward function.

    Later replace this with your teammate's real simulator / evaluator.
    """
    if ".end" not in netlist.lower():
        return -1.0

    return 0.7


def main():
    # 1. Load LLM engine
    engine = LLMEngine(
        model_id="Qwen/Qwen2.5-Coder-7B",
        quantization=None,  # None for now, bcs no bitsandbytes in environment.
    )

    # Optional: load SFT adapter if you already have one
    # engine.load_adapter("sft", "./checkpoints/sft-lora/final")

    # 2. Create RL updater
    updater = RLUpdater(
        engine,
        RLConfig(
            learning_rate=1e-5,
            kl_beta=0.1,
            save_every=5,
        )
    )

    # 3. Define one constraint
    constraint_dict = {
        "vin": 12,
        "vout_target": 5,
        "efficiency_target": 0.90,
        "converter_type": "buck",
        "power_out_w": 120,
        "component_preference": "minimal",
    }

    constraint = Constraint.from_dict(constraint_dict)

    # 4. Generate multiple candidate topologies
    generations = engine.generate_batch(
        constraint,
        n=4,
        temperature=0.7,
        top_p=0.9,
    )

    prompts = []
    completions = []
    rewards = []

    for gen in generations:
        prompt_text = constraint.to_prompt()
        topology_text = gen.netlist

        reward = fake_simulator_reward(topology_text)

        prompts.append(prompt_text)
        completions.append(topology_text)
        rewards.append(reward)

    # 5. Here update the weights in Lora layer.
    metrics = updater.update(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
    )

    # 6. Save RL adapter
    updater.save("./checkpoints/grpo-lora/final")

    print("=== RL Update Done ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()