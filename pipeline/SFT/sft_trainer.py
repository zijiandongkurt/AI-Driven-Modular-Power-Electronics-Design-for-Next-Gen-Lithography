"""
sft_trainer.py — Supervised fine-tuning of Qwen with LoRA.

Designed to be a sibling of `new_rl_updater.py` (GRPO):
  - same LoRA wrapping recipe
  - same AdamW optimizer scoped to LoRA params only
  - same prompt/completion tokenization logic
  - SFT-specific loss: next-token cross-entropy with prompt-masking
    (i.e. labels[:prompt_len] = -100 so only completion tokens contribute)

Pipeline integration:
    from pipeline.llm_topology_generation.llm_api import TopologyLLM
    from pipeline.reinforcement_algorithm.sft_trainer import SFTTrainer, SFTConfig

    llm = TopologyLLM(model_id="Qwen/Qwen3-14B")
    sft = SFTTrainer(llm.engine, SFTConfig())
    sft.train(train_path="data/sft/sft_train.jsonl",
              val_path="data/sft/sft_val.jsonl",
              epochs=5)
    sft.save("./checkpoints/sft-lora/final")

After this:
  - `llm` still works (it wraps the same engine; LoRA is now attached)
  - Subsequent GRPO with `GRPOTrainer(llm=llm, ...)` starts from the
    SFT-trained LoRA weights, exactly the "SFT → GRPO" recipe.

Data format (each line of the JSONL):
    {
      "prompt":     "<full prompt incl. NAMING_RULES + Constraint + ### SPICE Netlist:>",
      "completion": "<target SPICE netlist text>",
      ...                # other keys ignored
    }

Memory budget on H100 80GB with default SFTConfig (LoRA r=16, bs=4,
seq=2048):  ~75 GB.  On 40GB cards, set quantization="4bit" on the
underlying TopologyLLM and the same SFTConfig works.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from peft import LoraConfig, TaskType, get_peft_model

import torch
# Note: we don't need torch.nn.functional explicitly — HF's
# `model(input_ids=..., labels=...)` auto-computes cross-entropy with
# ignore_index=-100, so we just unpack `outputs.loss`.


# 
# Config
# 

@dataclass
class SFTConfig:
    """Hyperparameters for supervised fine-tuning on (prompt, netlist) pairs."""

    #  LoRA (same as RLConfig) 
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    #  Training (SFT-specific) 
    learning_rate: float = 2e-4    # ← bigger than GRPO's 2e-5 (SFT signal is stronger)
    max_grad_norm: float = 1.0
    batch_size: int = 4            # samples per optimizer step
    epochs: int = 5
    eval_every_epochs: int = 1     # run val pass every N epochs

    #  Sequence (same as RLConfig) 
    max_length: int = 2048
    max_prompt_length: int = 1536
    max_completion_length: int = 1024

    #  Runtime 
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    output_dir: str = "./checkpoints/sft-lora"


# 
# Trainer
# 

class SFTTrainer:
    """Fine-tune a model with next-token cross-entropy on (prompt, netlist) pairs.

    LoRA is applied directly to the shared ``LLMEngine`` model, so the
    trained weights are visible to any downstream GRPO update without
    reloading the model.
    """

    #  Initialization 

    def __init__(self, engine, config: Optional[SFTConfig] = None):

        self.cfg = config or SFTConfig()
        self.engine = engine
        self.step_id = 0

        # Apply LoRA if the model is not already a PEFT model.
        if not getattr(self.engine, "_is_peft", False):
            lora_config = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.cfg.lora_targets,
            )
            self.engine._model = get_peft_model(self.engine._model, lora_config)
            self.engine._is_peft = True
            self.engine._model.print_trainable_parameters()

        if self.cfg.gradient_checkpointing:
            self.engine._model.enable_input_require_grads()
            self.engine._model.gradient_checkpointing_enable()

        trainable_params = [
            p for p in self.engine.model.parameters() if p.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for SFT.")

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.cfg.learning_rate,
        )

        # Make tokenizer's pad token explicit (needed for batching).
        if self.engine.tokenizer.pad_token is None:
            self.engine.tokenizer.pad_token = self.engine.tokenizer.eos_token
            self.engine.tokenizer.pad_token_id = self.engine.tokenizer.eos_token_id

    #  Helpers 

    def _device(self):
        return self.engine.model.device

    def _tokenizer(self):
        return self.engine.tokenizer

    def _encode_one(self, prompt: str, completion: str) -> Optional[dict]:
        """Tokenize one (prompt, completion) pair into ids and labels.

        Completion tokens get their real ids as labels; prompt tokens are
        masked with -100 so they do not contribute to the CE loss.
        Truncates the prompt from the head and the completion from the tail
        when the combined length exceeds ``max_length``.

        Args:
            prompt (str): Full prompt string (system message + constraint).
            completion (str): Target SPICE netlist text.

        Returns:
            dict | None: Dict with keys ``"ids"`` (list[int]), ``"labels"``
                (list[int]), and ``"prompt_len"`` (int), or None if either
                input is empty after stripping.
        """
        tok = self._tokenizer()

        prompt = prompt.strip()
        completion = completion.strip()
        if not prompt or not completion:
            return None

        p_ids = tok(prompt, add_special_tokens=False).input_ids
        c_ids = tok(completion, add_special_tokens=False).input_ids
        # Make sure completion ends with EOS so the model learns to stop.
        if c_ids and c_ids[-1] != tok.eos_token_id:
            c_ids = c_ids + [tok.eos_token_id]

        # Per-side caps first
        p_ids = p_ids[-self.cfg.max_prompt_length:]   # keep tail of prompt
        c_ids = c_ids[: self.cfg.max_completion_length]

        # Global cap: keep full completion, then prompt from tail
        if len(p_ids) + len(c_ids) > self.cfg.max_length:
            keep_c = min(len(c_ids), self.cfg.max_completion_length)
            keep_p = max(self.cfg.max_length - keep_c, 1)
            p_ids = p_ids[-keep_p:]
            c_ids = c_ids[:keep_c]

        ids = p_ids + c_ids
        # Prompt positions masked with -100 → ignored by CE loss
        labels = [-100] * len(p_ids) + list(c_ids)
        return {"ids": ids, "labels": labels, "prompt_len": len(p_ids)}

    def _collate(self, encoded_batch: List[dict]) -> dict:
        """Right-pad a batch of encoded samples to the longest sequence length.

        Args:
            encoded_batch (list[dict]): Output dicts from ``_encode_one``,
                each with ``"ids"`` and ``"labels"`` keys.

        Returns:
            dict: Tensors ready for the model: ``"input_ids"``,
                ``"labels"``, and ``"attention_mask"``, all on the engine
                device.
        """
        tok = self._tokenizer()
        device = self._device()
        pad_id = tok.pad_token_id

        max_len = max(len(e["ids"]) for e in encoded_batch)
        ids, labels, masks = [], [], []
        for e in encoded_batch:
            n = len(e["ids"])
            pad = max_len - n
            ids.append(e["ids"] + [pad_id] * pad)
            # Padding positions are also -100 so they don't enter loss
            labels.append(e["labels"] + [-100] * pad)
            masks.append([1] * n + [0] * pad)
        return {
            "input_ids":      torch.tensor(ids,    dtype=torch.long, device=device),
            "labels":         torch.tensor(labels, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks,  dtype=torch.long, device=device),
        }

    #  Single optimizer step 

    def _step(self, batch_samples: List[dict]) -> dict:
        """Run one AdamW step on a list of prompt/completion dicts.

        Args:
            batch_samples (list[dict]): Each dict must have ``"prompt"`` and
                ``"completion"`` keys.

        Returns:
            dict: Step metrics with keys ``"loss"``, ``"n_samples"``, and
                ``"n_loss_tokens"``.  ``"loss"`` is None when all samples
                encode to empty.
        """
        # Encode + drop any sample that came out empty
        encoded = [self._encode_one(s["prompt"], s["completion"])
                   for s in batch_samples]
        encoded = [e for e in encoded if e is not None]
        if not encoded:
            return {"loss": None, "n_samples": 0}

        b = self._collate(encoded)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.cfg.bf16 and torch.cuda.is_available(),
        ):
            # HF auto-computes next-token CE with ignore_index=-100
            outputs = self.engine.model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                labels=b["labels"],
            )
            loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.engine.model.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        self.step_id += 1

        # Count how many actual completion tokens contributed to loss
        n_loss_tokens = int((b["labels"] != -100).sum().item())
        return {
            "loss":          float(loss.detach().cpu()),
            "n_samples":     len(encoded),
            "n_loss_tokens": n_loss_tokens,
        }

    #  Eval (no grad) 

    @torch.no_grad()
    def evaluate(self, samples: List[dict]) -> float:
        """Return mean next-token cross-entropy loss over held-out samples.

        Args:
            samples (list[dict]): Each dict must have ``"prompt"`` and
                ``"completion"`` keys.

        Returns:
            float: Token-weighted mean cross-entropy loss.
        """
        self.engine.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for s in samples:
            enc = self._encode_one(s["prompt"], s["completion"])
            if enc is None:
                continue
            b = self._collate([enc])
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.cfg.bf16 and torch.cuda.is_available(),
            ):
                out = self.engine.model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    labels=b["labels"],
                )
            n_tok = int((b["labels"] != -100).sum().item())
            total_loss += float(out.loss.detach().cpu()) * n_tok
            total_tokens += n_tok
        self.engine.model.train()
        return total_loss / max(total_tokens, 1)

    #  Main training loop 

    def train(
        self,
        train_path: str | Path,
        val_path: Optional[str | Path] = None,
        epochs: Optional[int] = None,
    ) -> List[dict]:
        """Train on a JSONL file where each line has ``prompt`` and ``completion``.

        Args:
            train_path (str | Path): Path to the training JSONL file.
            val_path (str | Path | None): Optional path to a validation JSONL
                file.  Validation is skipped when None.
            epochs (int | None): Number of training epochs; falls back to
                ``SFTConfig.epochs`` when None.

        Returns:
            list[dict]: Per-epoch history entries with ``"epoch"``,
                ``"train_loss"``, ``"tokens"``, ``"wall_s"``, and optionally
                ``"val_loss"``.

        Raises:
            RuntimeError: If no training samples are found in ``train_path``.
        """
        train_path = Path(train_path)
        train_samples = self._load_jsonl(train_path)
        if not train_samples:
            raise RuntimeError(f"No samples found in {train_path}")

        val_samples = (self._load_jsonl(Path(val_path)) if val_path else [])

        n_epochs = epochs or self.cfg.epochs
        rng = random.Random(self.cfg.seed)
        print(f"[sft] training on {len(train_samples)} samples, "
              f"val={len(val_samples)}, epochs={n_epochs}, "
              f"batch_size={self.cfg.batch_size}, lr={self.cfg.learning_rate}")

        self.engine.model.train()
        history: list[dict] = []
        for epoch in range(1, n_epochs + 1):
            rng.shuffle(train_samples)

            epoch_loss_sum = 0.0
            epoch_tokens = 0
            t0 = time.time()
            for i in range(0, len(train_samples), self.cfg.batch_size):
                batch = train_samples[i : i + self.cfg.batch_size]
                m = self._step(batch)
                if m["loss"] is None:
                    continue
                epoch_loss_sum += m["loss"] * m["n_loss_tokens"]
                epoch_tokens   += m["n_loss_tokens"]

                if self.step_id % 5 == 0:
                    print(f"  epoch={epoch} step={self.step_id} "
                          f"loss={m['loss']:.4f} "
                          f"bs={m['n_samples']} tokens={m['n_loss_tokens']}",
                          flush=True)

            dt = time.time() - t0
            train_loss = epoch_loss_sum / max(epoch_tokens, 1)
            entry = {"epoch": epoch, "train_loss": train_loss,
                     "tokens": epoch_tokens, "wall_s": round(dt, 1)}

            if val_samples and (epoch % self.cfg.eval_every_epochs == 0):
                val_loss = self.evaluate(val_samples)
                entry["val_loss"] = val_loss
                print(f"[sft]  epoch {epoch}/{n_epochs}  "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                      f"({dt:.1f}s)", flush=True)
            else:
                print(f"[sft]  epoch {epoch}/{n_epochs}  "
                      f"train_loss={train_loss:.4f} ({dt:.1f}s)", flush=True)

            history.append(entry)
            self.save(f"{self.cfg.output_dir}/epoch-{epoch}")

        return history

    #  Save / load helpers 

    def save(self, path: Optional[str] = None) -> Path:
        """Save the LoRA adapter and tokenizer to disk.

        Args:
            path (str | None): Destination directory.  Falls back to
                ``SFTConfig.output_dir/final`` when None.

        Returns:
            Path: Absolute path to the saved checkpoint directory.
        """
        save_path = Path(path or f"{self.cfg.output_dir}/final")
        save_path.mkdir(parents=True, exist_ok=True)
        self.engine.model.save_pretrained(str(save_path))
        self.engine.tokenizer.save_pretrained(str(save_path))
        print(f"[sft] saved LoRA adapter to: {save_path}", flush=True)
        return save_path

    @staticmethod
    def _load_jsonl(path: Path) -> List[dict]:
        """Read a JSONL file of prompt/completion records.

        Args:
            path (Path): Path to the JSONL file.

        Returns:
            list[dict]: Dicts with ``"prompt"`` and ``"completion"`` keys;
                lines missing either key are silently dropped.
        """
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "prompt" in obj and "completion" in obj:
                    out.append({"prompt": obj["prompt"], "completion": obj["completion"]})
        return out


# 
# CLI entry point: `python -m pipeline.reinforcement_algorithm.sft_trainer ...`
# 

def _main():
    import argparse, os
    from pipeline.llm_topology_generation.llm_api import TopologyLLM

    ap = argparse.ArgumentParser(description="SFT trainer for Qwen + LoRA")
    ap.add_argument("--train", type=str, default="data/sft/sft_train.jsonl",
                    help="Training JSONL (default: the canonical training file)")
    ap.add_argument("--val", type=str, default="data/sft/sft_val.jsonl",
                    help="Validation JSONL (set empty to skip eval)")
    ap.add_argument("--model-id", type=str,
                    default=os.environ.get("GRPO_MODEL_ID", "Qwen/Qwen3-14B"))
    ap.add_argument("--quantization", type=str,
                    default=os.environ.get("GRPO_QUANTIZATION", "") or None,
                    choices=["4bit", None])
    ap.add_argument("--epochs",     type=int,   default=5)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--lora-r",     type=int,   default=16)
    ap.add_argument("--lora-alpha", type=int,   default=32)
    ap.add_argument("--max-length", type=int,   default=2048)
    ap.add_argument("--output-dir", type=str,
                    default="./checkpoints/sft-lora")
    args = ap.parse_args()

    cfg = SFTConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )

    print(f"[sft] loading TopologyLLM(model_id={args.model_id!r}, "
          f"quantization={args.quantization!r})")
    llm = TopologyLLM(model_id=args.model_id, quantization=args.quantization)

    trainer = SFTTrainer(llm.engine, cfg)
    history = trainer.train(
        train_path=args.train,
        val_path=args.val if args.val else None,
    )
    final_path = trainer.save(f"{args.output_dir}/final")

    # Dump training history alongside the checkpoint
    with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[sft] history written to {args.output_dir}/history.json")
    print(f"[sft] final adapter: {final_path}")


if __name__ == "__main__":
    _main()
