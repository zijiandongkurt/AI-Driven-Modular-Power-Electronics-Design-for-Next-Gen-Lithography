from pipeline.llm_topology_generation.llm_engine_minimal import LLMEngine
from pipeline.simulation.ltspice_runner import TopologySimulator
from pipeline.reward_evaluation.reward_function import RewardFunction
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer

def main():
    llm = LLMEngine()
    simulator = TopologySimulator()
    reward_function = RewardFunction()
    grpo_trainer = GRPOTrainer()

    # sample constraint for testing
    constraint = {
        "vin_min": 12,
        "vin_max": 100,
        "vout_target": 5,
        "efficiency_target": 0.90,
        "power_in": 100,
    }

    # 1. Generate topologies
    generation_output = llm.generate(constraint)
    print("Generated Netlist:\n", generation_output.netlist)

    # 2. Simulate topologies
    batch_id = "sample_batch_001"
    simulation_results = simulator.simulate(batch_id)
    print("Simulation Results:\n", simulation_results)

    # 3. Evaluate rewards    
    reward_results = reward_function.process_csv_and_calculate_fitness(
        csv_file_path=f"../simulation/output/{batch_id}/results.csv", 
        constraints=constraint, 
        weights={"efficiency": 0.5})
    print("Reward Results:\n", reward_results)

    # 4. Train policy with GRPO
    grpo_trainer.train(
        constraint_id=batch_id,
        prompt_text=llm.generate(constraint).to_prompt(),
        topologies=simulation_results,
        rewards=reward_results,
    )

if __name__ == "__main__":
    main()



