"""Generation with the intervention applied, streaming tokens and projections.

A hand-written decode loop rather than `model.generate` + a streamer thread.
The reason is synchronisation: every emitted token must carry the projection
recorded on the forward pass that produced it, and gluing a background thread's
token stream to a hook's side-effect list is a race waiting to happen. The loop
below is ~30 lines and the pairing is structural.

Greedy by default. Evals need to be re-runnable and comparable across
conditions; sampling would put noise on exactly the measurement the demo rests
on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from . import hooks as hooks_mod
from . import model as model_mod

ARTIFACTS = Path(__file__).parent / "artifacts"


@dataclass
class Direction:
    rhat: torch.Tensor
    layer: int
    diff_norm: float
    meta: dict

    @property
    def model_id(self) -> str:
        return self.meta["model_id"]


def load_direction(
    name: str = "direction",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Direction:
    npz_path = ARTIFACTS / f"{name}.npz"
    json_path = ARTIFACTS / f"{name}.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found — run `python -m backend.extract` first."
        )
    with open(json_path) as f:
        meta = json.load(f)
    rhat = np.load(npz_path)["rhat"]
    # r̂ is kept in float32 regardless of the model's dtype. It is used inside
    # fp32 projection arithmetic (see hooks.project_out) and rounding it to
    # bf16 first would defeat that.
    tensor = torch.from_numpy(rhat.astype(np.float32)).to(device)
    return Direction(rhat=tensor, layer=meta["layer"], diff_norm=meta["diff_norm"], meta=meta)


def direction_for_layer(
    layer: int,
    name: str = "direction",
    device: str = "cpu",
) -> Direction:
    """Rebuild r̂ at an arbitrary layer from the artifact's cached activations.

    The extraction artifact stores every layer's activations, so sweeping the
    intervention layer costs no forward passes — only the generation needed to
    measure the causal effect. Without this, a layer sweep would mean re-running
    extraction once per candidate.
    """
    import numpy as _np

    npz = _np.load(ARTIFACTS / f"{name}.npz")
    with open(ARTIFACTS / f"{name}.json") as f:
        meta = dict(json.load(f))
    diff = npz["acts_harmful"][:, layer, :].mean(axis=0) - npz["acts_harmless"][:, layer, :].mean(axis=0)
    norm = float(_np.linalg.norm(diff))
    meta["layer"] = layer
    meta["diff_norm"] = norm
    return Direction(
        rhat=torch.from_numpy((diff / norm).astype(_np.float32)).to(device),
        layer=layer,
        diff_norm=norm,
        meta=meta,
    )


def _stop_ids(lm: model_mod.LoadedModel) -> set:
    ids = set()
    for src in (lm.tokenizer.eos_token_id, getattr(lm.model.generation_config, "eos_token_id", None)):
        if src is None:
            continue
        ids.update(src if isinstance(src, (list, tuple)) else [src])
    return ids


@torch.no_grad()
def stream_generate(
    lm: model_mod.LoadedModel,
    prompt: str,
    alpha: float = 0.0,
    direction: Optional[Direction] = None,
    max_new_tokens: int = 48,
) -> Iterator[dict]:
    """Yield one event per generated token.

    Each event: {"token": str, "index": int, "projection": float,
                 "projection_post": float}

    `projection` is the model's own pre-intervention projection onto r̂ at the
    extraction layer, for the forward pass that produced this token. Under full
    ablation it stays informative (the model still tries to write the direction
    back); `projection_post` is what survived, and is ~0 there by construction.
    """
    iv = None
    trace = None
    if direction is not None:
        trace = hooks_mod.ProjectionTrace()
        iv = hooks_mod.resolve_alpha(alpha, direction.rhat, direction.layer, direction.diff_norm)

    ids = lm.encode(prompt)
    stop = _stop_ids(lm)
    generated: list[int] = []
    prev_text = ""

    with hooks_mod.apply(lm.blocks, iv, trace):
        past = None
        cur = ids
        for step in range(max_new_tokens):
            out = lm.model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            if next_id in stop:
                break
            generated.append(next_id)

            # Decode the whole suffix each step and emit the delta. Decoding
            # single ids in isolation splits multi-byte characters and emits
            # replacement chars mid-word.
            text = lm.tokenizer.decode(generated, skip_special_tokens=True)
            piece, prev_text = text[len(prev_text):], text

            yield {
                "token": piece,
                "index": step,
                "projection": trace.pre[step] if trace and step < len(trace.pre) else 0.0,
                "projection_post": trace.post[step] if trace and step < len(trace.post) else 0.0,
            }
            cur = torch.tensor([[next_id]], device=lm.model.device)


def generate_text(
    lm: model_mod.LoadedModel,
    prompt: str,
    alpha: float = 0.0,
    direction: Optional[Direction] = None,
    max_new_tokens: int = 48,
) -> str:
    """Collected form of `stream_generate`, for eval scripts."""
    return "".join(
        ev["token"]
        for ev in stream_generate(lm, prompt, alpha, direction, max_new_tokens)
    )
