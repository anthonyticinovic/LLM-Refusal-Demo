"""Direction-quality checks. No forward passes — runs in under a second.

`extract.py` writes every layer's activations into the artifact, so all of this
is arithmetic on cached arrays. That is deliberate: these are the checks worth
re-running after any change to the prompt sets or the extraction code, and a
check that takes a second gets run while one that takes ten minutes does not.

The per-layer table is also how the extraction layer gets chosen. Split-half
stability and projection separation are free to compute at every depth; only
the causal check (check_causal.py) needs generation.

    python -m backend.checks.check_direction

Exits non-zero if any assertion fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..extract import ARTIFACTS, direction_at_layer, projection_separation, split_half_cosine

# Thresholds from the definition of done.
MIN_SPLIT_HALF_COS = 0.8
MIN_SEPARATION_SDS = 1.0
UNIT_NORM_TOL = 1e-5


def main() -> int:
    npz_path = ARTIFACTS / "direction.npz"
    if not npz_path.exists():
        print(f"FAIL  {npz_path} missing — run `python -m backend.extract` first.")
        return 1

    data = np.load(npz_path)
    with open(ARTIFACTS / "direction.json") as f:
        meta = json.load(f)

    rhat = data["rhat"]
    harmful = data["acts_harmful"]
    harmless = data["acts_harmless"]
    layer = meta["layer"]
    n_layers = meta["n_layers"]

    print(f"model        {meta['model_id']}")
    print(f"layer        {layer} of {n_layers} blocks   ({layer / n_layers:.0%} depth)")
    print(f"pairs        {meta['n_pairs']}")
    print(f"prompt hash  {meta['prompt_set_hash']}")
    if meta.get("refusal_filtered"):
        print(f"filtered     {meta['refusal_filter_note']}")
    print()

    # Per-layer landscape. Cheap, and the most useful single output here.
    print("  layer   split-half cos   separation (SDs)   ‖d‖")
    candidates = []
    for i in range(n_layers):
        r, dn = direction_at_layer(harmful, harmless, i)
        sep = projection_separation(harmful, harmless, i, r)["separation_sds"]
        cos = split_half_cosine(harmful, harmless, i)
        mark = "  <- extraction" if i == layer else ""
        # Restrict sweep candidates to the middle band. The last layers often
        # show the largest separation and are still poor intervention sites:
        # by then the residual stream is being read off as logits, so editing
        # it steers the current token without changing what the model has
        # already committed to. (The ‖d‖ blow-up at the final layer is the
        # tell.) The paper's middle-to-late finding is the band worth sweeping.
        if 0.30 <= i / n_layers <= 0.80 and cos > MIN_SPLIT_HALF_COS:
            candidates.append((sep, i))
        print(f"  {i:5d}   {cos:14.4f}   {sep:16.2f}   {dn:6.2f}{mark}")
    print()
    if candidates:
        top = [i for _, i in sorted(candidates, reverse=True)[:3]]
        print(f"sweep candidates (stable, 30-80% depth, best separation first): {top}")
        print("only the causal check decides — see check_causal.py --layer-sweep")
        print()

    checks = []

    norm_err = abs(float(np.linalg.norm(rhat)) - 1.0)
    checks.append(("‖r̂‖ = 1 within 1e-5", norm_err < UNIT_NORM_TOL, f"|‖r̂‖ − 1| = {norm_err:.2e}"))

    cos = split_half_cosine(harmful, harmless, layer)
    checks.append((
        f"split-half cosine > {MIN_SPLIT_HALF_COS}",
        cos > MIN_SPLIT_HALF_COS,
        f"cos = {cos:.4f}",
    ))

    # Stability across several splits, not just the one seed. A single split
    # clearing the bar can be luck; the spread is the honest summary.
    cosines = [split_half_cosine(harmful, harmless, layer, seed=s) for s in range(10)]
    checks.append((
        f"split-half cosine > {MIN_SPLIT_HALF_COS} across 10 random splits",
        min(cosines) > MIN_SPLIT_HALF_COS,
        f"min = {min(cosines):.4f}, mean = {np.mean(cosines):.4f}, max = {max(cosines):.4f}",
    ))

    sep = projection_separation(harmful, harmless, layer, rhat)
    checks.append((
        f"projection separation > {MIN_SEPARATION_SDS} SD",
        sep["separation_sds"] > MIN_SEPARATION_SDS,
        f"{sep['separation_sds']:.2f} SDs "
        f"(harmless {sep['harmless_mean']:+.3f}, harmful {sep['harmful_mean']:+.3f})",
    ))

    checks.append((
        "artifact records layer and prompt-set hash",
        "layer" in meta and bool(meta.get("prompt_set_hash")),
        f"layer={meta.get('layer')}, hash={meta.get('prompt_set_hash')}",
    ))

    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"        {detail}")
        failed += not ok

    print()
    print(f"{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
