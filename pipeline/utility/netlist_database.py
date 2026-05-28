import json
import numpy as np
from pathlib import Path

class NetlistDatabase:
    """
    In-memory database to track all generated netlists and their fitness.
    Implements Epsilon-Greedy Top-K Softmax sampling.
    """
    def __init__(self, temperature=0.05): # Lowered temperature for greedy exploitation
        self.records = {}
        self.temperature = temperature
        
        # Track validity statistics
        self.total_valid = 0
        self.total_invalid = 0

    def add_record(self, candidate_id, netlist_text, fitness, feedback_text, is_valid=False, metrics=None):
        self.records[candidate_id] = {
            "netlist_text": netlist_text,
            "fitness": fitness,
            "feedback": feedback_text,
            "is_valid": is_valid,
            "metrics": metrics or {} 
        }
        
        if is_valid:
            self.total_valid += 1
        else:
            self.total_invalid += 1

    def ingest_batch_data(self, batch_id: str, database_dir: Path):
        """
        Parses the results of a batch from disk and pushes all candidates into the database.
        Also saves a physical copy of the netlist to the global database folder.
        """
        data_dir = Path("pipeline/data") / batch_id
        reward_path = data_dir / "reward_results.json"
        val_path = data_dir / "validation_results.json"
        llm_out_dir = data_dir / "LLM_output"

        if not reward_path.exists():
            return

        with open(reward_path, "r", encoding="utf-8") as f:
            rewards = json.load(f).get("circuits", {})
            
        with open(val_path, "r", encoding="utf-8") as f:
            validations = json.load(f) if val_path.exists() else {}

        for cand_id, metrics in rewards.items():
            # 1. Get Netlist Text
            net_file = llm_out_dir / f"{cand_id}.net"
            if not net_file.exists():
                continue
            netlist_text = net_file.read_text(encoding="utf-8")

            # ---> Save copy to the central database folder <---
            db_file = database_dir / f"{cand_id}.net"
            db_file.write_text(netlist_text, encoding="utf-8")

            # 2. Get Fitness
            fitness = metrics.get("fitness_score", -1.0)
            is_valid = validations.get(cand_id, {}).get("passed", False)

            # 3. Construct Feedback String
            feedback_text = f"Fitness Score: {fitness:.4f}\n"
            if metrics.get("source") == "validation_penalty":
                failed_checks = [k for k, v in validations.get(cand_id, {}).get("checks", {}).items() if not v]
                feedback_text += f"FAILED TESTS: {', '.join(failed_checks)}\n"
            else:
                raw = metrics.get("raw_metrics", {})
                feedback_text += f"V_out: {raw.get('simulation_output_voltage', 0):.2f}V, Efficiency: {raw.get('efficiency', 0):.2f}\n"

            # 4. Push to Database
            self.add_record(cand_id, netlist_text, fitness, feedback_text, is_valid=is_valid, metrics=metrics)

    def sample_states(self, n=2, top_k=15, epsilon=0.15):
        """
        Samples 'n' states using Epsilon-Greedy Top-K Softmax.
        - (1 - epsilon) chance: Exploits the Top-K elite circuits using Softmax.
        - (epsilon) chance: Explores the "Long Tail" uniformly to rescue unoptimized topologies.
        """
        if not self.records:
            return []
        
        keys = list(self.records.keys())
        
        # 1. Score is now purely the raw fitness
        scored_candidates = []
        for k in keys:
            score = self.records[k]["fitness"]
            scored_candidates.append((k, score))
            
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Split into Elite (Top-K) and Long Tail
        elite_candidates = scored_candidates[:top_k]
        elite_keys = [x[0] for x in elite_candidates]
        elite_scores = np.array([x[1] for x in elite_candidates])
        
        long_tail_keys = [x[0] for x in scored_candidates[top_k:]]
        
        # 3. Calculate Softmax probabilities for the Elite group
        exp_scores = np.exp((elite_scores - np.max(elite_scores)) / self.temperature)
        elite_probs = exp_scores / np.sum(exp_scores)
        
        chosen_keys = []
        
        # 4. Sample 'n' times
        for _ in range(n):
            if np.random.rand() < epsilon and len(long_tail_keys) > 0:
                chosen = np.random.choice(long_tail_keys)
                chosen_keys.append(chosen)
                long_tail_keys.remove(chosen)
            else:
                chosen = np.random.choice(elite_keys, p=elite_probs)
                chosen_keys.append(chosen)

        return [{"id": k, **self.records[k]} for k in chosen_keys]