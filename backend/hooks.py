"""The two interventions, and the exact meaning of the UI's single `alpha`.

    ALPHA MAPPING (one slider, one number, both directions)

    alpha ∈ [-1, +1]

      alpha <  0   ablation.  strength t = |alpha|, interpolating 0 → 1.
                   Applied at EVERY decoder block:   h ← h − t·(h·r̂)r̂
                   At t = 1 the residual stream carries no component along r̂
                   at any depth.

      alpha == 0   baseline. No hooks fire. This is the untouched model.

      alpha >  0   injection. Applied at the EXTRACTION LAYER ONLY:
                   h ← h + c·r̂,  where  c = alpha · INJECT_MAX_MULT · ‖d‖
                   and ‖d‖ is the norm of the raw difference-of-means vector
                   before normalisation.

Two notes on that mapping, both deliberate.

Injection is scaled in units of ‖d‖ rather than raw activation units. A bare
"alpha = 3.0" would mean something different at every layer and every model,
because residual stream norms grow with depth; expressing it as a multiple of
the harmful-minus-harmless difference makes alpha = 1.0 mean "add one full
class separation" and stay comparable across layers. INJECT_MAX_MULT is the
one free constant here and sweep_alpha.py is what sets it.

The two operations are exclusive, not composed. Ablation lives on the negative
side and injection on the positive side, so no slider position both removes the
direction and adds it back. That is a UI decision, not a mathematical
restriction — `Intervention` will happily do both at once if constructed
directly, which the sweep scripts rely on.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import List, Optional

import torch

# Maximum injection strength at alpha = +1, in multiples of ‖d‖.
# Calibrated per direction: at alpha = +1 the coefficient is INJECT_MAX_MULT·‖d‖,
# and residual norms — hence the coefficient a given multiple buys — vary with
# layer and model. For Qwen2.5-3B, layer 21, ‖d‖ ≈ 14.7, a sweep showed 2.0
# (coefficient ≈ 29) is the largest value that still yields coherent refusal on
# benign prompts; beyond it the output degrades into hyperbolic condemnation and
# then word-salad. 0.5B, with a much smaller ‖d‖, wanted a larger multiple.
# Re-sweep when the extraction layer or model changes.
INJECT_MAX_MULT = 2.0


@dataclass
class Intervention:
    """A resolved intervention: what to do, at which layers, how strongly."""

    rhat: torch.Tensor          # (d_model,) unit vector
    layer: int                  # extraction/injection block index
    ablation: float = 0.0       # 0..1, applied at every block
    injection: float = 0.0      # activation-space coefficient at `layer`

    @property
    def is_noop(self) -> bool:
        return self.ablation == 0.0 and self.injection == 0.0


def resolve_alpha(
    alpha: float,
    rhat: torch.Tensor,
    layer: int,
    diff_norm: float,
    inject_max_mult: float = INJECT_MAX_MULT,
) -> Intervention:
    """Turn the UI's single alpha into a concrete intervention.

    See the module docstring for the mapping. Kept as a pure function so the
    server, the eval scripts and the tests all agree on what a given slider
    position means.
    """
    alpha = float(max(-1.0, min(1.0, alpha)))
    if alpha < 0:
        return Intervention(rhat=rhat, layer=layer, ablation=-alpha, injection=0.0)
    if alpha > 0:
        return Intervention(
            rhat=rhat, layer=layer, ablation=0.0,
            injection=alpha * inject_max_mult * diff_norm,
        )
    return Intervention(rhat=rhat, layer=layer)


def _split_output(out):
    """Decoder blocks return either a tensor or a tuple whose first element is
    the hidden state. Normalise, and give back a rebuilder."""
    if isinstance(out, tuple):
        return out[0], lambda h: (h,) + out[1:]
    return out, lambda h: h


def project_out(h: torch.Tensor, rhat: torch.Tensor, strength: float) -> torch.Tensor:
    """h − t·(h·r̂)r̂, computed in float32.

    The cast is not incidental. In bfloat16 the residual stream has roughly
    three significant decimal digits, and `h − (h·r̂)r̂` is a cancellation: the
    result is the small part left after subtracting most of a large number. Done
    in bf16 the "ablated" stream retains a visible component along r̂, which
    shows up downstream as ablation that mysteriously fails to suppress
    refusal. Compute in fp32, store back in the model's dtype.
    """
    dtype = h.dtype
    h32 = h.float()
    r32 = rhat.float()
    proj = h32 @ r32                       # (batch, seq)
    h32 = h32 - strength * proj.unsqueeze(-1) * r32
    return h32.to(dtype)


@dataclass
class ProjectionTrace:
    """Per-forward-call projection onto r̂ at the extraction layer.

    Two series are recorded because they answer different questions:

      `pre`  — the projection the model itself computed, before we touched it.
               Under full ablation this is NOT zero: earlier blocks were
               cleaned, so this measures how much refusal direction the model
               writes back at this layer for this token. That is the
               interesting one.
      `post` — what actually flowed onward after the intervention. Under full
               ablation it is ~0 by construction, which is the useful sanity
               check that the hooks fired at all.

    `uhat`, when supplied, is the presenter demo's second plane axis (see
    projection.py). `pre_y` then tracks the token's position along it, so the
    live point can be drawn in the same plane as the cached activations. It is
    recorded pre-intervention to match `pre` — and because both interventions
    move the stream along r̂, which is orthogonal to u, so the intervention
    should leave this coordinate alone. Watching it stay put while `pre` moves
    is a free sanity check that the intervention is as surgical as claimed.
    """

    pre: List[float] = field(default_factory=list)
    post: List[float] = field(default_factory=list)
    uhat: Optional[torch.Tensor] = None
    pre_y: List[float] = field(default_factory=list)

    def reset(self) -> None:
        self.pre.clear()
        self.post.clear()
        self.pre_y.clear()


@contextlib.contextmanager
def apply(model_blocks, iv: Optional[Intervention], trace: Optional[ProjectionTrace] = None):
    """Register the intervention's hooks for the duration of the block.

    Hooks are always removed on exit, including on exception. The server holds
    one model for its lifetime and mutates only the hooks, which is what lets
    an alpha change take effect on the next generation with no reload.
    """
    handles = []
    try:
        if iv is not None:
            rhat = iv.rhat

            def make_hook(block_idx: int):
                is_target = block_idx == iv.layer

                def hook(_module, _args, output):
                    h, rebuild = _split_output(output)

                    if is_target and trace is not None:
                        # Last position only: during prefill that is the final
                        # prompt token, during decode it is the token just
                        # produced. One entry per forward call either way.
                        last = h[0, -1].float()
                        trace.pre.append(float((last @ rhat.float()).item()))
                        if trace.uhat is not None:
                            trace.pre_y.append(float((last @ trace.uhat.float()).item()))

                    if iv.ablation > 0.0:
                        h = project_out(h, rhat, iv.ablation)
                    if is_target and iv.injection != 0.0:
                        h = h + (iv.injection * rhat).to(h.dtype)

                    if is_target and trace is not None:
                        trace.post.append(float((h[0, -1].float() @ rhat.float()).item()))

                    return rebuild(h)

                return hook

            # Ablation is applied at every block; injection only at the target.
            # A block that is neither still needs a hook when it is the target
            # (for the trace), hence the membership test rather than a filter.
            for i, block in enumerate(model_blocks):
                needs_hook = (
                    iv.ablation > 0.0
                    or (i == iv.layer and (iv.injection != 0.0 or trace is not None))
                )
                if needs_hook:
                    handles.append(block.register_forward_hook(make_hook(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()
