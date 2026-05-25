import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.reinforcement_algorithm.new_rl_updater import RLUpdater, RLConfig


class GRPOTrainer:
    """
    GRPO trainer.

    New grouped-GRPO behavior:
        group g1: g1_cand1, g1_cand2, g1_cand3, g1_cand4
        group g2: g2_cand1, g2_cand2, g2_cand3, g2_cand4

    Advantages are normalized within each group, not across the whole batch.
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
        """Fallback prompt if group-specific prompt file is missing."""
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

    def _load_group_prompt(self, batch_id: str, group_id: str) -> str:
        """
        Load prompt for a specific GRPO group.

        Expected files:
            prompt_g1.txt
            prompt_g2.txt

        Falls back to reconstructed prompt.
        """
        prompt_path = self._batch_dir(batch_id) / f"prompt_{group_id}.txt"

        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        print(f"Warning: missing {prompt_path}. Using fallback prompt.")
        return self._load_prompt()

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

    def _extract_group_id(self, topology_id: str) -> str:
        """
        Extract group id from grouped filename.

        Example:
            00_xxx_g1_cand3_b2 -> g1

        If no group id exists, fallback to one shared group.
        """
        match = re.search(r"_(g\d+)_cand\d+", topology_id)
        if match:
            return match.group(1)

        # Fallback for legacy non-grouped batches.
        return "g1"

    def _get_reward(self, topology_id: str, info: Dict) -> Tuple[Optional[float], Optional[str]]:
        """Read reward from reward_results.json."""
        if "grpo_reward" in info:
            return float(info["grpo_reward"]), "grpo_reward"

        if "fitness_score" in info:
            return float(info["fitness_score"]), "fitness_score"

        print(f"Skipping {topology_id}: no grpo_reward or fitness_score found.")
        return None, None

    def _normalize_group_rewards(self, rewards: List[float]) -> List[float]:
        """
        Normalize rewards inside one prompt group.

        This is the key grouped-GRPO step.
        """
        if len(rewards) < 2:
            raise RuntimeError(
                "Each GRPO group needs at least 2 samples to compute relative advantage."
            )

        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = max(var ** 0.5, 1e-8)

        return [(r - mean) / std for r in rewards]

    def _build_grouped_training_batch(self, batch_id: str):
        """
        Build prompts, completions, rewards, and grouped advantages.

        Important:
            advantages are normalized within each group_id.
        """
        reward_data = self._load_rewards(batch_id)
        circuits = reward_data.get("circuits", {})

        if not circuits:
            raise RuntimeError(f"No circuits found in reward_results.json for {batch_id}")

        groups: Dict[str, List[Dict]] = {}

        # Collect samples by group_id.
        for topology_id, info in circuits.items():
            group_id = self._extract_group_id(topology_id)

            try:
                completion_text = self._load_netlist(batch_id, topology_id)
            except FileNotFoundError as e:
                print(f"Skipping {topology_id}: {e}")
                continue

            reward, reward_source = self._get_reward(topology_id, info)
            if reward is None:
                continue

            prompt_text = self._load_group_prompt(batch_id, group_id)

            groups.setdefault(group_id, []).append({
                "topology_id": topology_id,
                "prompt": prompt_text,
                "completion": completion_text,
                "reward": reward,
                "reward_source": reward_source,
            })

            print(
                f"Loaded {topology_id}: "
                f"group={group_id}, reward={reward:.4f} ({reward_source})"
            )

        if not groups:
            raise RuntimeError("No valid grouped prompt/completion/reward samples found.")

        prompts: List[str] = []
        completions: List[str] = []
        rewards: List[float] = []
        advantages: List[float] = []
        group_ids: List[str] = []
        topology_ids: List[str] = []

        # Compute advantages separately within each group.
        for group_id, samples in groups.items():
            if len(samples) < 2:
                print(f"Skipping group {group_id}: only {len(samples)} sample.")
                continue

            group_rewards = [s["reward"] for s in samples]
            group_advantages = self._normalize_group_rewards(group_rewards)

            for sample, advantage in zip(samples, group_advantages):
                prompts.append(sample["prompt"])
                completions.append(sample["completion"])
                rewards.append(sample["reward"])
                advantages.append(advantage)
                group_ids.append(group_id)
                topology_ids.append(sample["topology_id"])

        if not rewards:
            raise RuntimeError("No valid GRPO groups found after grouping.")

        return prompts, completions, rewards, advantages, group_ids, topology_ids

    def _save_metrics(self, batch_id: str, metrics: Dict):
        save_path = self._batch_dir(batch_id) / "grpo_metrics.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {save_path}")

    def update_from_batch(self, batch_id: str, max_samples: Optional[int] = None) -> Dict:
        """
        Run grouped GRPO update from an already evaluated batch.

        This assumes:
            generate -> validate -> simulate -> reward

        It then:
            groups candidates by g1/g2/...
            normalizes rewards within each group
            passes external advantages to RLUpdater
        """
        (
            prompts,
            completions,
            rewards,
            advantages,
            group_ids,
            topology_ids,
        ) = self._build_grouped_training_batch(batch_id)

        # Optional debug limit.
        # Warning: slicing may cut groups, so use max_samples=None for real training.
        if max_samples is not None:
            prompts = prompts[:max_samples]
            completions = completions[:max_samples]
            rewards = rewards[:max_samples]
            advantages = advantages[:max_samples]
            group_ids = group_ids[:max_samples]
            topology_ids = topology_ids[:max_samples]

        if len(rewards) < 2:
            raise RuntimeError(
                f"GRPO update requires at least 2 samples, but got {len(rewards)}."
            )

        print(f"Running grouped GRPO update from batch: {batch_id}")
        print(f"RL samples: {len(rewards)}")
        print(f"Groups: {group_ids}")
        print(f"Rewards: {rewards}")
        print(f"Advantages: {advantages}")

        metrics = self.rl_updater.update(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
        )

        # Add grouped-GRPO metadata to metrics.
        metrics["group_ids"] = group_ids
        metrics["topology_ids"] = topology_ids
        metrics["grouped_grpo"] = True

        self._save_metrics(batch_id, metrics)
        self.rl_updater.save(self.output_dir)

        print("=== Grouped GRPO Update From Batch Done ===")
        print(json.dumps(metrics, indent=2))

        return metrics

    def train(self, batch_id: str = "batch_1", n: int = 4, max_samples: Optional[int] = None):
        """
        Legacy full-pipeline GRPO iteration.

        In the new project setup, training_loop.py is the main orchestrator.
        This method is kept only for compatibility.
        """
        written = self.llm.generate_for_batch(
            self.constraint_dict,
            batchID=batch_id,
            n=n,
        )
        print(f"Generated {len(written)} netlists")

        self.validator.validate(batch_id)

        simulation_results = self.simulator.simulate(batch_id)
        print("Simulation Results:")
        print(simulation_results)

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

        return self.update_from_batch(
            batch_id=batch_id,
            max_samples=max_samples,
        )