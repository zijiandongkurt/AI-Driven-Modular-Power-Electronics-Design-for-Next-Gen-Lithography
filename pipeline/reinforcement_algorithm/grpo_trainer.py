import json
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.reinforcement_algorithm.new_rl_updater import RLUpdater, RLConfig

class GRPOTrainer:
    """
    GRPO trainer:

    It uses the existing project pipeline:
        LLM generation
        -> validation
        -> simulation
        -> reward evaluation
        -> RLUpdater.update()

    The trainer then aligns:
        prompt      = system_prompt.txt + constraint
        completion  = pipeline/data/<batch_id>/LLM_output/*.net
        reward      = pipeline/data/<batch_id>/reward_results.json
    """

    def __init__(
        self,
        llm,
        validator,
        simulator,
        reward_fn,
        constraint: Dict,
        rl_config: Optional[RLConfig] = None,
        output_dir: str = "./checkpoints/grpo-lora/final",
        system_prompt_path: str = "system_prompt.txt",
    ):
        self.llm = llm
        self.validator = validator
        self.simulator = simulator
        self.reward_fn = reward_fn
        self.constraint_dict = constraint
        self.output_dir = output_dir
        self.system_prompt_path = Path(system_prompt_path)

        # Reuse the already-loaded LLM engine from TopologyLLM.
        self.rl_updater = RLUpdater(
            self.llm.engine,
            rl_config or RLConfig(
                learning_rate=1e-5,
                kl_beta=0.0,
                save_every=5,
                lora_r=8,
                lora_alpha=16,
            ),
        )

    def _batch_dir(self, batch_id: str) -> Path:
        return Path("pipeline") / "data" / batch_id

    def _llm_output_dir(self, batch_id: str) -> Path:
        # Current project uses uppercase LLM_output.
        return self._batch_dir(batch_id) / "LLM_output"

    def _reward_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "reward_results.json"

    def _simulation_output_dir(self, batch_id: str) -> Path:
        return Path("pipeline") / "simulation" / "output" / batch_id

    def _load_rewards(self, batch_id: str) -> Dict:
        reward_path = self._reward_path(batch_id)

        if not reward_path.exists():
            raise FileNotFoundError(f"Missing reward file: {reward_path}")

        with reward_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """
        Build the prompt used for RL log-prob calculation.

        The prompt is reconstructed from:
            system_prompt.txt + current constraint
        """
        if self.system_prompt_path.exists():
            system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        else:
            print(f"Warning: {self.system_prompt_path} not found. Using constraint only.")
            system_prompt = ""

        return (
            system_prompt.strip()
            + "\n\n### Constraint:\n"
            + json.dumps(self.constraint_dict, indent=2)
            + "\n\n### SPICE Netlist:\n"
        )

    def _load_netlist(self, batch_id: str, topology_id: str) -> str:
        netlist_path = self._llm_output_dir(batch_id) / f"{topology_id}.net"

        if not netlist_path.exists():
            raise FileNotFoundError(f"Missing netlist file: {netlist_path}")

        return netlist_path.read_text(encoding="utf-8")

    def _load_failure_log(self, batch_id: str, topology_id: str) -> Optional[str]:
        fail_path = self._simulation_output_dir(batch_id) / f"{topology_id}.fail"

        if not fail_path.exists():
            return None

        return fail_path.read_text(encoding="utf-8")

    def _build_training_batch(self, batch_id: str):
        """
        Build prompt/completion/reward lists for RLUpdater.

        reward_results.json structure:
            {
                "active_constraints": {...},
                "circuits": {
                    "top1": {
                        "fitness_score": -0.2135,
                        "grpo_reward": 0.3078,
                        "loss_breakdown": {...},
                        "raw_metrics": {...}
                    }
                }
            }

        RL uses:
            prompt      = system_prompt.txt + constraint
            completion  = pipeline/data/<batch_id>/LLM_output/<topology_id>.net
            reward      = grpo_reward if available, otherwise fitness_score
        """
        reward_data = self._load_rewards(batch_id)
        circuits = reward_data.get("circuits", {})

        if not circuits:
            raise RuntimeError(f"No circuits found in reward_results.json for {batch_id}")

        prompt_text = self._load_prompt()

        prompts: List[str] = []
        completions: List[str] = []
        rewards: List[float] = []

        for topology_id, info in circuits.items():
            try:
                completion_text = self._load_netlist(batch_id, topology_id)
            except FileNotFoundError as e:
                print(f"Skipping {topology_id}: {e}")
                continue

            # Use normalized GRPO reward when available.
            if "grpo_reward" in info:
                reward = float(info["grpo_reward"])
                reward_source = "grpo_reward"
            elif "fitness_score" in info:
                reward = float(info["fitness_score"])
                reward_source = "fitness_score"
            else:
                print(f"Skipping {topology_id}: no grpo_reward or fitness_score found.")
                continue

            prompts.append(prompt_text)
            completions.append(completion_text)
            rewards.append(reward)

            print(f"Loaded {topology_id}: reward={reward:.4f} ({reward_source})")

        if not rewards:
            raise RuntimeError("No valid prompt/completion/reward pairs found.")

        return prompts, completions, rewards

    def _save_metrics(self, batch_id: str, metrics: Dict):
        save_path = self._batch_dir(batch_id) / "grpo_metrics.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {save_path}")

#Tesr RL loop with this method, incase the LTspice not working on HPC:
    def train_from_existing_batch(self, batch_id: str = "batch_1"):
        """
        Run RL update using existing LLM_output and reward_results.json.
        This does not run generation, validation, simulation, or reward evaluation.
        """
        prompts, completions, rewards = self._build_training_batch(batch_id)
        #Added these bcs of Out of Memory:
        prompts = prompts[:2]
        completions = completions[:2]
        rewards = rewards[:2]
        ####

        print(f"RL samples: {len(rewards)}")
        print(f"Rewards: {rewards}")

        metrics = self.rl_updater.update(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
        )
        self._save_metrics(batch_id, metrics)   
        self.rl_updater.save(self.output_dir)

        print("=== GRPO Training From Existing Batch Done ===")
        print(json.dumps(metrics, indent=2))

        return metrics
# The real train method, once the LTspice works in HPC:
    def train(self, batch_id: str = "batch_1", n: int = 4):
        """
        Run one GRPO training iteration.

        This function still runs the full pipeline:
            generate -> validate -> simulate -> reward -> RL update
        """

        # 1. Generate netlists.
        written = self.llm.generate_for_batch(
            self.constraint_dict,
            batchID=batch_id,
            n=n,
        )
        print(f"Generated {len(written)} netlists")

        # 2. Validate generated netlists.
        self.validator.validate(batch_id)

        # 3. Simulate valid netlists.
        simulation_results = self.simulator.simulate(batch_id)
        print("Simulation Results:")
        print(simulation_results)

        # 4. Compute rewards.
        self.reward_fn.process_batch(
            batch_id,
            self.constraint_dict,
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
        )

        # 5. Build RL training data from saved files.
        prompts, completions, rewards = self._build_training_batch(batch_id)

        print(f"RL samples: {len(rewards)}")    # for debugging
        print(f"Rewards: {rewards}")            # for debugging

        # print failure logs if available, for debugging
        for topology_id in self._load_rewards(batch_id).get("circuits", {}).keys():
            fail_log = self._load_failure_log(batch_id, topology_id)
            if fail_log:
                print(f"\nFailure log for {topology_id}:")
                print(fail_log[:1000])

        # 6. Update LoRA policy.
        metrics = self.rl_updater.update(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
        )

        # 7. Save adapter.
        self.rl_updater.save(self.output_dir)

        print("=== GRPO Training Done ===")
        print(json.dumps(metrics, indent=2))

        return metrics