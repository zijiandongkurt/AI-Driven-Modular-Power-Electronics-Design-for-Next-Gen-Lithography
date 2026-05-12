"""
llm_engine.py — Minimal LLM topology generation module.

Only two jobs: load model, generate netlists.
Everything else (training, validation, simulation, reward) lives elsewhere.

Usage:
    engine = LLMEngine()  # loads Qwen-3.5 27B from Snellius shared cache
    engine.load_adapter("sft", "./checkpoints/sft-lora/final")
    netlist = engine.generate({"vin": 12, "vout_target": 5})
"""

from __future__ import annotations

import json
import os
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import TextStreamer

import torch

from .prompt_input import NAMING_RULES

logger = logging.getLogger(__name__)


# ── Snellius model config ────────────────────────────────────────────────

# Shared HuggingFace cache on Snellius — no download needed.
SNELLIUS_HF_CACHE = "/projects/2/managed_datasets/hf_cache_dir"

# Qwen-3 14B — free access, Apache 2.0, 52GB on disk.
# Verify the exact folder name with:
#   ls /projects/2/managed_datasets/hf_cache_dir | grep -i qwen
DEFAULT_MODEL_ID = "Qwen/Qwen3-14B"


# ── Data contracts ───────────────────────────────────────────────────────

@dataclass
class Constraint:
    """Design constraint spec. Shared contract with all other modules."""

    vin: float
    vout_target: float
    efficiency_target: float = 0.90
    ripple_max: float = 0.05
    converter_type: str = "auto"
    switching_freq_khz: float = 100.0
    power_out_w: float = 10.0
    component_preference: str = "minimal"
    i_switch_max: Optional[float] = None
    v_stress_max: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> Constraint:
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_prompt(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return (
            f"{NAMING_RULES}\n"
            f"### Constraint:\n{json.dumps(d, indent=2)}\n\n"
            f"### SPICE Netlist:\n"
        )


@dataclass
class GenerationOutput:
    """What this module returns. Other modules consume this."""
    netlist: str
    constraint: Constraint
    tokens: int = 0
    time_sec: float = 0.0


# ── LLM Engine ─────────────────────────────────────────────────────────

class LLMEngine:
    """
    Loads a model, manages adapters, generates netlists.

    Public interface (what other modules call):
        engine.generate(constraint)        -> GenerationOutput
        engine.generate_batch(constraint)  -> list[GenerationOutput]
        engine.load_adapter(name, path)
        engine.switch_adapter(name)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_cache_dir: str = SNELLIUS_HF_CACHE,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Point HuggingFace at the Snellius shared cache so no download occurs.
        os.environ["HF_HUB_CACHE"] = hf_cache_dir
        os.environ["HF_HUB_OFFLINE"] = "1"  # never attempt a download
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Store generation tuning hyperparameters.
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

        # Adapter management state (name -> folder path).
        self._adapters: dict[str, str] = {}
        self._active: Optional[str] = None
        self._is_peft = False

        logger.info(f"Loading tokenizer from cache: {hf_cache_dir}")

        # Load tokenizer
        self._tok = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
            local_files_only=False,
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        # Build model load kwargs
        kw: dict = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
            local_files_only=False,
        )

        # 4-bit quantisation — useful if VRAM is tight even on Snellius A100s.
        # For Qwen 27B on a single 80GB A100, bfloat16 fits without quantisation.
        # Enable only if you are running on smaller GPUs or want to fit more
        # parallel workers on the same node.
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Try flash-attention-2 first (available on Snellius A100 nodes),
        # fall back to standard attention if not installed.
        try:
            kw["attn_implementation"] = "flash_attention_2"
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except Exception:
            kw.pop("attn_implementation", None)
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kw)

        logger.info(f"Model loaded: {model_id} ({self._model.num_parameters()/1e9:.1f}B params)")

    # ── Adapter management ───────────────────────────────────────────

    def load_adapter(self, name: str, path: str) -> None:
        """Load a LoRA adapter from disk. First call wraps model with PEFT."""
        from peft import PeftModel
        if not self._is_peft:
            self._model = PeftModel.from_pretrained(self._model, path, adapter_name=name)
            self._is_peft = True
        else:
            self._model.load_adapter(path, adapter_name=name)
        self._adapters[name] = path
        self._active = name
        self._model.set_adapter(name)

    def switch_adapter(self, name: str) -> None:
        """Zero-latency adapter swap."""
        if name not in self._adapters:
            raise KeyError(f"Unknown adapter '{name}'. Loaded: {list(self._adapters)}")
        self._model.set_adapter(name)
        self._active = name

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active

    @property
    def model(self):
        """Raw model reference — for external trainers that need direct access."""
        return self._model

    @property
    def tokenizer(self):
        """Raw tokenizer reference — for external trainers that need direct access."""
        return self._tok

    # ── Generation ───────────────────────────────────────────────────

    def generate(
        self,
        constraint: dict | Constraint,
        feedback: str = "",
        **overrides,
    ) -> GenerationOutput:
        """
        Generate one SPICE netlist from a constraint spec.

        Args:
            constraint: Design constraints (dict or Constraint).
            feedback:   Optional error text from a failed prior attempt.
            **overrides: temperature, top_p, max_new_tokens overrides.

        Returns:
            GenerationOutput with the netlist string.
        """
        import time
        t0 = time.time()

        if isinstance(constraint, dict):
            constraint = Constraint.from_dict(constraint)

        prompt = self._build_prompt(constraint, feedback)
        ids = self._tok(prompt, return_tensors="pt")
        ids = {k: v.to(self._model.device) for k, v in ids.items()}

        with torch.no_grad():
            out = self._model.generate(
                **ids,
                max_new_tokens=overrides.get("max_new_tokens", self._max_new_tokens),
                do_sample=True,
                temperature=overrides.get("temperature", self._temperature),
                top_p=overrides.get("top_p", self._top_p),
                pad_token_id=self._tok.pad_token_id,
            )

        gen_ids = out[0][ids["input_ids"].shape[1]:]
        raw = self._tok.decode(gen_ids, skip_special_tokens=True)
        netlist = self._clean(raw)

        return GenerationOutput(
            netlist=netlist,
            constraint=constraint,
            tokens=len(gen_ids),
            time_sec=round(time.time() - t0, 3),
        )

    def generate_batch(
        self,
        constraint: dict | Constraint,
        n: int = 8,
        **overrides,
    ) -> list[GenerationOutput]:
        """Generate n candidate netlists for the same constraint."""
        return [self.generate(constraint, **overrides) for _ in range(n)]

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(c: Constraint, feedback: str) -> str:
        prompt = c.to_prompt()
        if feedback:
            prompt += f"\n### Feedback:\n{feedback}\n\n### Corrected SPICE Netlist:\n"
        return prompt

    @staticmethod
    def _clean(raw: str) -> str:
        """Clean generated text and keep only the first full netlist.

        - Trim whitespace
        - Remove Markdown code fences
        - Stop at .end if present (SPICE convention)
        """
        lines = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("```"):
                continue
            lines.append(line)
            if line.lower() == ".end":
                break
        return "\n".join(lines)

    @staticmethod
    def parse_constraint_from_prompt(prompt: str) -> Constraint:
        """Utility for external modules that receive raw prompt strings."""
        m = re.search(r"### Constraint:\s*\n(.*?)### SPICE Netlist:", prompt, re.DOTALL)
        if m:
            return Constraint.from_dict(json.loads(m.group(1).strip()))
        raise ValueError("Cannot parse constraint from prompt")