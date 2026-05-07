from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import TopologySimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer


def main():
    llm = TopologyLLM(model_id="Qwen/Qwen3-14B")
    #llm = TopologyLLM(model_id="Qwen/Qwen2.5-Coder-7B")
    
    val = validator()
    simulator = TopologySimulator()
    reward_fn = RewardFunctionNorm()
    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=0)

    grpo = GRPOTrainer(
        llm=llm,
        validator=val,
        simulator=simulator,
        reward_fn=reward_fn,
        constraint=constraint,
    )

#    grpo.train(batch_id="batch_1", n=4)
    grpo.train_from_existing_batch(batch_id="batch_1")

if __name__ == "__main__":
    main()