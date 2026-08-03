"""Alignment with r̂ at every block and every token of a prompt.

One forward pass gives every block's residual stream at every position, so this
costs a single prompt-length forward and no generation at all — cheaper than the
two generations the demo's first slide already runs.

Reported as a COSINE, not a raw dot product. Residual norms grow substantially
with depth, so raw `h · r̂` rises through the stack whether or not anything
refusal-shaped is happening, and the late layers would glow for a trivial
reason. Dividing by ‖h‖ asks the scale-free question the demo actually means:
what fraction of this activation points along r̂?

Position 0 needs no special handling for the same reason. It carries a huge raw
projection — the attention-sink artefact every decoder-only transformer has —
but a huge norm with it, so in cosine terms it sits with everything else.

Block outputs are captured with forward hooks rather than read out of
`output_hidden_states`. The last entry of `hidden_states` is NOT the last
block's output — transformers has already applied the model's final norm to it,
and measurably so: at the final position ‖h‖ is 248.9 from the block itself
against 180.2 from `hidden_states`. Every other index matches exactly. Hooks
therefore give one consistent quantity at every depth, which is the minimum bar
for a picture whose whole point is comparing depths.

Note what this does NOT explain. The top row of the heatmap is nearly blank
because alignment really does collapse at the last block: cosine at the final
position runs 0.186 at block 34 and 0.038 at block 35, and that 0.038 is the
same to two significant figures whichever accessor you use. The cliff is a
property of the model, not of the read — plausibly the stream being turned into
logits once the decision is already made. Do not "fix" it.
"""

from __future__ import annotations

import torch

from . import hooks as hooks_mod
from . import model as model_mod


@torch.no_grad()
def scan_prompt(lm: model_mod.LoadedModel, rhat: torch.Tensor, prompt: str) -> dict:
    """Cosine between each position's residual stream and r̂, at every block.

    Returns `tokens` (decoded, one per position) and `grid` (n_layers rows of
    n_tokens values, row `i` being the output of decoder block `i`).
    """
    r = rhat.float()
    r = r / r.norm()

    grid: list = [None] * lm.n_layers
    handles = []

    def make_hook(i: int):
        def hook(_module, _args, output):
            h, _ = hooks_mod._split_output(output)
            h = h[0].float()
            grid[i] = [round(float(v), 4) for v in (h @ r) / (h.norm(dim=-1) + 1e-6)]
        return hook

    try:
        for i, block in enumerate(lm.blocks):
            handles.append(block.register_forward_hook(make_hook(i)))
        ids = lm.encode(prompt)
        lm.model(ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "tokens": [lm.tokenizer.decode([int(i)]) for i in ids[0]],
        "grid": grid,
        "n_layers": lm.n_layers,
    }
