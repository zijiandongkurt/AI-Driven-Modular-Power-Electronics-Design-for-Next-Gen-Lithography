"""
grpo_trainer.py  —  HPC-tuned drop-in replacement.

Differences from the original pipeline/reinforcement_algorithm/grpo_trainer.py:

  1. The `[:2]` OOM workaround in `train_from_existing_batch` is removed.
     On 80GB / 4×40GB HPC nodes we can afford the full 4-sample batch,
     which is required for GRPO's group-relative reward normalization to
     produce a meaningful std estimate.

  2. `system_prompt_path` is resolved to an absolute path relative to the
     repo root, so the script works regardless of the cwd SLURM hands you.

  3. Loud, structured logging at each pipeline stage (we lose interactive
     stdout on SLURM, so we print explicit START/END markers).

Algorithm itself is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.reinforcement_algorithm.new_rl_updater import RLUpdater, RLConfig


# Repo root — used to resolve paths regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class GRPOTrainer:
    """
    GRPO trainer:

        LLM generation
        -> validation
        -> simulation
        -> reward evaluation
        -> RLUpdater.update()
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

        # Resolve system_prompt_path absolutely — SLURM might cd elsewhere.
        sp = Path(system_prompt_path)
        self.system_prompt_path = sp if sp.is_absolute() else (_REPO_ROOT / sp)

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

    # ── Path helpers ────────────────────────────────────────────────────

    def _batch_dir(self, batch_id: str) -> Path:
        return _REPO_ROOT / "pipeline" / "data" / batch_id

    def _llm_output_dir(self, batch_id: str) -> Path:
        # Project convention: uppercase LLM_output.
        return self._batch_dir(batch_id) / "LLM_output"

    def _reward_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "reward_results.json"

    def _simulation_output_dir(self, batch_id: str) -> Path:
        return _REPO_ROOT / "pipeline" / "simulation" / "output" / batch_id

    # ── Loaders ─────────────────────────────────────────────────────────

    def _load_rewards(self, batch_id: str) -> Dict:
        reward_path = self._reward_path(batch_id)

        if not reward_path.exists():
            raise FileNotFoundError(f"Missing reward file: {reward_path}")

        with reward_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """
        Build the prompt used for RL log-prob calculation:
            system_prompt.txt + current constraint
        """
        if self.system_prompt_path.exists():
            system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        else:
            print(f"WARNING: {self.system_prompt_path} not found. Using constraint only.",
                  file=sys.stderr)
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

    # ── Batch builder ───────────────────────────────────────────────────

    def _build_training_batch(self, batch_id: str):
        """
        Build prompt/completion/reward lists for RLUpdater.

        reward_results.json structure:
            {
                "active_constraints": {...},
                "circuits": {
                    "top1": {
                        "fitness_score": -0.2135,
                        "grpo_reward":     0.3078,
                        "loss_breakdown": {...},
                        "raw_metrics":     {...}
                    },
                    ...
                }
            }
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

    # ── Entry points ────────────────────────────────────────────────────

    def train_from_existing_batch(self, batch_id: str = "batch_1"):
        """
        Run RL update using existing LLM_output + reward_results.json.
        Does not run generation/validation/simulation/reward — useful when
        ngspice isn't available and you just want to test the RL path.

        NOTE: The original code capped this to 2 samples ([:2]) to avoid
        OOM on a workstation GPU.  On the HPC nodes (80GB single OR
        4×40GB sharded) we restore the full 4-sample batch, which is what
        GRPO's group-relative normalization needs to estimate variance.
        """
        print(f"=== [GRPO/existing-batch] {batch_id} ===", flush=True)
        prompts, completions, rewards = self._build_training_batch(batch_id)

        print(f"RL samples: {len(rewards)}")
        print(f"Rewards:    {rewards}")

        metrics = self.rl_updater.update(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
        )

        self._save_metrics(batch_id, metrics)
        self.rl_updater.save(self.output_dir)

        print("=== GRPO Training From Existing Batch Done ===", flush=True)
        print(json.dumps(metrics, indent=2))

        return metrics

    def train(self, batch_id: str = "batch_1", n: int = 4):
        """
        Run one full GRPO training iteration:
            generate -> validate -> simulate -> reward -> RL update
        """
        print(f"=== [GRPO/full] {batch_id} (n={n}) ===", flush=True)

        # 1. Generate
        print("[stage] 1/5  generate", flush=True)
        written = self.llm.generate_for_batch(self.constraint_dict, batchID=batch_id, n=n)
        print(f"  generated {len(written)} netlists", flush=True)

        # 2. Validate
        print("[stage] 2/5  validate", flush=True)
        self.validator.validate(batch_id)

        # 3. Simulate
        print("[stage] 3/5  simulate", flush=True)
        simulation_results = self.simulator.simulate(batch_id)
        print(simulation_results)

        # 4. Reward
        print("[stage] 4/5  reward", flush=True)
        self.reward_fn.process_batch(
            batch_id,
            self.constraint_dict,
            weights={
                "v_out":          10.0,
                "efficiency":     20.0,
                "volume":          2.0,
                "component_cost":  1.0,
                "components": {
                    "mosfet":    1.0,
                    "diode":     1.0,
                    "inductor":  1.0,
                    "capacitor": 1.0,
                },
            },
        )

        # 5. RL update
        print("[stage] 5/5  RL update", flush=True)
        prompts, completions, rewards = self._build_training_batch(batch_id)
        print(f"  RL samples: {len(rewards)}")
        print(f"  Rewards:    {rewards}")

        for topology_id in self._load_rewards(batch_id).get("circuits", {}).keys():
            fail_log = self._load_failure_log(batch_id, topology_id)
            if fail_log:
                print(f"\nFailure log for {topology_id}:")
                print(fail_log[:1000])

        metrics = self.rl_updater.update(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
        )
        self._save_metrics(batch_id, metrics)
        self.rl_updater.save(self.output_dir)

        print("=== GRPO Training Done ===", flush=True)
        print(json.dumps(metrics, indent=2))

        return metrics
