from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator as Validator
from pipeline.simulation.ltspice_runner import TopologySimulator
from pipeline.reward_evaluation.reward_function import RewardFunction

def main():
    llm        = TopologyLLM()
    val        = Validator()
    simulator  = TopologySimulator()
    reward_fn  = RewardFunction()

    constraint = {
        "vin_min": 12,
        "vin_max": 100,
        "vout_target": 5,
        "efficiency_target": 0.90,
        "power_in": 100,
    }
    batch_id = "batch_001"

    # 1. Generate — writes .net files to data/batch_001/llm_output/
    written = llm.generate_for_batch(constraint, batchID=batch_id, n=4)
    print(f"Generated {len(written)} netlists")

    # 2. Validate — reads llm_output/, writes validation_results.json
    val.validate(batch_id)

    # 3. Simulate — reads validation_results.json, writes simulation_results.csv
    simulation_results = simulator.simulate(batch_id)
    print("Simulation Results:\n", simulation_results)

    # 4. Evaluate rewards — reads simulation_results.csv, writes reward_results.json
    reward_fn.process_csv_to_json(
        csv_file_path  = f"data/{batch_id}/simulation_results.csv",
        output_json_path = f"data/{batch_id}/reward_results.json",
        constraints    = constraint,
        weights        = {
            "v_out": 10.0, "efficiency": 20.0,
            "volume": 2.0, "component_cost": 1.0,
            "components": {"mosfet": 1.0, "diode": 1.0,
                           "inductor": 1.0, "capacitor": 1.0}
        },
        include_detailed_metrics=True,
    )

if __name__ == "__main__":
    main()