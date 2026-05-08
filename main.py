from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.ltspice_runner import LTSpiceSimulator
from pipeline.simulation.ngspice_runner import NGSpiceSimulator
from pipeline.reward_evaluation.reward_function import RewardFunction
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
from pipeline.reinforcement_algorithm.new_rl_updater import RLUpdater, RLConfig
from pathlib import Path

def main():
    # llm        = TopologyLLM(model_id="Qwen/Qwen3-14B")
    val        = validator()
    simulator = NGSpiceSimulator(ngspice_command=str(Path.home() / ".local/bin/ngspice"))
    reward_fn  = RewardFunctionNorm()
    constraint = load_constraint("pipeline/data/datasets/constraints.json", idx=0)
    batch_id   = "batch_1"

# to run GRPOTrainer:
    # the GRPOTrainer includes the steps 1-4.
    """
    grpo = GRPOTrainer(
        llm=llm,
        validator=val,
        simulator=simulator,
        reward_fn=reward_fn,
        constraint=constraint,
    )
    grpo.train_from_existing_batch(batch_id="batch_1")
    """ 
#
    # # 1. Generate — writes .net files to data/batch_7/llm_output/
    # written = llm.generate_for_batch(constraint, batchID=batch_id, n=4)
    # print(f"Generated {len(written)} netlists")

    # 2. Validate — reads llm_output/, writes validation_results.json
    val.validate(batch_id)

    # 3. Simulate — reads validation_results.json, writes simulation_results.csv
    simulation_results = simulator.simulate(batch_id)
    print("Simulation Results:\n", simulation_results)

    # 4. Evaluate rewards — reads simulation_results.csv, writes reward_results.json
    reward_fn.process_batch(batch_id, constraint, weights={
        "v_out": 10.0, "efficiency": 20.0,
        "volume": 2.0, "component_cost": 1.0,
        "components": {"mosfet": 1.0, "diode": 1.0,
                    "inductor": 1.0, "capacitor": 1.0}
    })

    

    


if __name__ == "__main__":
    main()