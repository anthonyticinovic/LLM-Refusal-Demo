"""Alignment with r̂ at every block and every token of a prompt.

One forward pass gives every layer's residual stream at every position, so this
costs a single prompt-length forward and no generation at all — cheaper than the
two generations the demo's first slide already runs.

Reported as a COSINE, not a raw dot product. Residual norms grow substantially
with depth, so raw `h · r̂` rises through the stack whether or not anything
refusal-shaped is happening, and the late layers would glow for a trivial
reason. Dividing by ‖h‖ asks the scale-free question the demo actually means:
what fraction of this activation points along r̂?

Position 0 needs no special handling here for the same reason. It carries a huge
raw projection — the attention-sink artefact every decoder-only transformer has —
but a huge norm with it, so in cosine terms it sits with everything else.
"""

from __future__ import annotations

import torch

from . import model as model_mod


@torch.no_grad()
def scan_prompt(lm: model_mod.LoadedModel, rhat: torch.Tensor, prompt: str) -> dict:
    """Cosine between each position's residual stream and r̂, at every block.

    Returns `tokens` (decoded, one per position) and `grid` (n_layers rows of
    n_tokens values, row `i` being the output of decoder block `i`).
    """
    ids = lm.encode(prompt)
    out = lm.model(ids, output_hidden_states=True, use_cache=False)

    r = rhat.float()
    r = r / r.norm()

    grid = []
    for layer in range(lm.n_layers):
        h = lm.hidden_state_for_layer(out.hidden_states, layer)[0].float()
        cos = (h @ r) / (h.norm(dim=-1) + 1e-6)
        grid.append([round(float(v), 4) for v in cos])

    return {
        "tokens": [lm.tokenizer.decode([int(i)]) for i in ids[0]],
        "grid": grid,
        "n_layers": lm.n_layers,
    }
