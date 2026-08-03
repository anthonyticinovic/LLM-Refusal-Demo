"""Model loading and layer-indexing conventions.

Everything downstream depends on one convention, so it is stated once here and
referred to everywhere else:

    LAYER INDEX `i` MEANS THE OUTPUT OF DECODER BLOCK `i`.

That is, `model.model.layers[i]`'s output, which is identically
`output_hidden_states=True`'s `hidden_states[i + 1]`. Valid range is
`0 .. n_layers - 1`. The off-by-one between block index and hidden-state index
is the single easiest way to extract a direction from one layer and inject it
into a different one, so both accessors live here and nothing else indexes
`hidden_states` directly.

No remote inference: weights are pulled from the HF hub once and cached, and
every forward pass after that runs locally. See checks/check_offline.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Overridable so the same code can be smoke-tested against a small model on a
# slow machine and run for real against a larger one.
DEFAULT_MODEL = os.environ.get("LATENTSPACE_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # bfloat16 on Apple Silicon is a large speed/memory win. On CPU it is a
    # pessimisation in torch 2.2 (many ops fall back to fp32 anyway, via a
    # conversion), so CPU stays in float32 and simply runs slower.
    return torch.bfloat16 if device in ("mps", "cuda") else torch.float32


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    model_id: str
    device: str
    dtype: torch.dtype

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def blocks(self) -> torch.nn.ModuleList:
        """The decoder blocks. Llama- and Qwen-style models both expose these
        at `model.model.layers`; this is the one place that assumption lives."""
        return self.model.model.layers

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    def hidden_state_for_layer(self, hidden_states, layer: int) -> torch.Tensor:
        """Residual stream after decoder block `layer`.

        `hidden_states[0]` is the embedding output, so block `layer` maps to
        index `layer + 1`. See the module docstring.

        ONE EXCEPTION, and it matters. The final entry is not the last block's
        output — transformers has already applied the model's final norm to it,
        so for `layer == n_layers - 1` this returns a normed vector. Measured on
        Qwen2.5-3B at the last prompt position: ‖h‖ 248.9 from a forward hook on
        the block against 180.2 here. Every other index matches the block output
        exactly.

        Harmless for extraction, which uses a middle layer, and for the cached
        sweep at any layer but the last. Do NOT use it to compare the final block
        against the others — register a forward hook instead, as scan.py does.
        """
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range for {self.n_layers}-block model")
        return hidden_states[layer + 1]

    def format_prompt(self, user_text: str) -> str:
        """Apply the model's chat template with the assistant turn opened.

        Extraction reads the activation at the final token of this string. That
        position is the one the paper uses: the model has ingested the
        instruction and is about to commit to a first response token, which is
        where the decision to refuse is most legible.
        """
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode(self, user_text: str) -> torch.Tensor:
        ids = self.tokenizer(self.format_prompt(user_text), return_tensors="pt").input_ids
        return ids.to(self.model.device)


_CACHE: dict = {}


def load(model_id: str | None = None, device: str | None = None) -> LoadedModel:
    """Load (and process-cache) a model. Repeated calls are free, which is what
    lets the server switch alpha between generations without a reload."""
    model_id = model_id or DEFAULT_MODEL
    device = device or pick_device()
    key = (model_id, device)
    if key in _CACHE:
        return _CACHE[key]

    dtype = pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    loaded = LoadedModel(model=model, tokenizer=tok, model_id=model_id, device=device, dtype=dtype)
    _CACHE[key] = loaded
    return loaded


def suggested_layer(n_layers: int) -> int:
    """~60% depth, per the paper's finding that middle-to-late layers carry the
    refusal direction most cleanly. A starting point for the sweep, not a
    conclusion — checks/sweep_layers.py is what actually decides."""
    return int(round(0.6 * n_layers)) - 1
