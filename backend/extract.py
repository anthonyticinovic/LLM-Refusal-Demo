"""Difference-of-means extraction of the refusal direction.

    r̂ = normalize( mean(harmful activations) − mean(harmless activations) )

taken at the last prompt token, at one layer. That is the whole method — the
paper's contribution is the claim that this crude estimator finds something
causally load-bearing, not the estimator itself.

One forward pass per prompt caches EVERY layer's activation, and all of them
are written to the artifact. Sweeping layers, checking split-half stability and
measuring projection separation then cost no additional forward passes, which
matters because those are the checks worth re-running after any change.

Usage (from the repo root):

    python -m backend.extract                 # auto layer (~60% depth)
    python -m backend.extract --layer 20
    python -m backend.extract --refusal-filter   # see note below
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from . import model as model_mod
from .refusal import is_refusal

ARTIFACTS = Path(__file__).parent / "artifacts"
PROMPTS = Path(__file__).parent / "prompts"


def load_prompts(name: str) -> List[str]:
    with open(PROMPTS / name) as f:
        return json.load(f)["prompts"]


def prompt_set_hash(*prompt_lists: List[str]) -> str:
    h = hashlib.sha256()
    for lst in prompt_lists:
        for p in lst:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\xff")
    return h.hexdigest()[:16]


@torch.no_grad()
def cache_activations(lm: model_mod.LoadedModel, prompts: List[str], label: str = "") -> np.ndarray:
    """Activation at the final prompt token, for every layer.

    Returns (n_prompts, n_layers, d_model) float32.
    """
    out = np.zeros((len(prompts), lm.n_layers, lm.d_model), dtype=np.float32)
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        ids = lm.encode(prompt)
        res = lm.model(ids, output_hidden_states=True, use_cache=False)
        for layer in range(lm.n_layers):
            h = lm.hidden_state_for_layer(res.hidden_states, layer)
            out[i, layer] = h[0, -1].float().cpu().numpy()
        if label and (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {label}: {i + 1}/{len(prompts)}  ({rate:.1f} prompt/s)", flush=True)
    return out


def direction_at_layer(
    harmful: np.ndarray, harmless: np.ndarray, layer: int
) -> Tuple[np.ndarray, float]:
    """r̂ and ‖d‖ at one layer, from (n, n_layers, d) activation caches."""
    diff = harmful[:, layer, :].mean(axis=0) - harmless[:, layer, :].mean(axis=0)
    norm = float(np.linalg.norm(diff))
    return diff / norm, norm


def projection_separation(
    harmful: np.ndarray, harmless: np.ndarray, layer: int, rhat: np.ndarray
) -> dict:
    """How well the direction separates the two classes at this layer.

    Reported in pooled standard deviations, which is the scale-free version and
    the one the definition-of-done specifies a threshold against.
    """
    pa = harmless[:, layer, :] @ rhat
    pb = harmful[:, layer, :] @ rhat
    pooled = float(np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2))
    return {
        "harmless_mean": float(pa.mean()),
        "harmful_mean": float(pb.mean()),
        "gap": float(pb.mean() - pa.mean()),
        "pooled_sd": pooled,
        "separation_sds": float((pb.mean() - pa.mean()) / pooled) if pooled > 0 else 0.0,
    }


def split_half_cosine(harmful: np.ndarray, harmless: np.ndarray, layer: int, seed: int = 0) -> float:
    """The paper's own stability check: extract r̂ from each half of the prompt
    set independently and compare. A direction that is an artefact of a handful
    of prompts will not survive this."""
    rng = np.random.default_rng(seed)
    n = min(len(harmful), len(harmless))
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2 :]
    r1, _ = direction_at_layer(harmful[a], harmless[a], layer)
    r2, _ = direction_at_layer(harmful[b], harmless[b], layer)
    return float(r1 @ r2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="HF model id (default: $LATENTSPACE_MODEL)")
    ap.add_argument("--layer", type=int, default=None, help="block index; default ~60%% depth")
    ap.add_argument("--out", default=None, help="artifact basename (default: direction)")
    ap.add_argument(
        "--refusal-filter",
        action="store_true",
        help=(
            "Keep only harmful prompts the model actually refuses at baseline. "
            "Without this the extracted direction risks encoding TOPIC rather "
            "than REFUSAL: if the model happily answers the harmful set, the "
            "difference-of-means separates 'security-flavoured' from "
            "'domestic-flavoured' prompts and nothing more. Costs one "
            "generation per harmful prompt."
        ),
    )
    args = ap.parse_args()

    lm = model_mod.load(args.model)
    layer = args.layer if args.layer is not None else model_mod.suggested_layer(lm.n_layers)
    print(f"model={lm.model_id}  device={lm.device}  dtype={lm.dtype}")
    print(f"blocks={lm.n_layers}  d_model={lm.d_model}  extraction layer={layer}")

    harmful_prompts = load_prompts("harmful.json")
    harmless_prompts = load_prompts("harmless.json")

    filtered_note = None
    if args.refusal_filter:
        from .generate import generate_text

        print("filtering harmful set to prompts the model actually refuses...")
        kept_h, kept_b = [], []
        for hp, bp in zip(harmful_prompts, harmless_prompts):
            text = generate_text(lm, hp, alpha=0.0, max_new_tokens=24)
            if is_refusal(text):
                kept_h.append(hp)
                kept_b.append(bp)
        filtered_note = f"{len(kept_h)}/{len(harmful_prompts)} harmful prompts refused at baseline"
        print(f"  {filtered_note}")
        if len(kept_h) < 10:
            raise SystemExit(
                f"Only {len(kept_h)} harmful prompts were refused at baseline. The extracted "
                "direction would be topic, not refusal. Use a stronger model or sharper prompts."
            )
        # Pairs are index-matched, so both sides must be filtered together to
        # keep the form-matching that makes the difference interpretable.
        harmful_prompts, harmless_prompts = kept_h, kept_b

    print(f"caching activations for {len(harmful_prompts)} pairs...")
    acts_harmful = cache_activations(lm, harmful_prompts, "harmful")
    acts_harmless = cache_activations(lm, harmless_prompts, "harmless")

    rhat, diff_norm = direction_at_layer(acts_harmful, acts_harmless, layer)
    sep = projection_separation(acts_harmful, acts_harmless, layer, rhat)
    cos = split_half_cosine(acts_harmful, acts_harmless, layer)

    ARTIFACTS.mkdir(exist_ok=True)
    base = args.out or "direction"

    np.savez(
        ARTIFACTS / f"{base}.npz",
        rhat=rhat.astype(np.float32),
        acts_harmful=acts_harmful,
        acts_harmless=acts_harmless,
    )

    meta = {
        "model_id": lm.model_id,
        "device": lm.device,
        "dtype": str(lm.dtype),
        "layer": layer,
        "n_layers": lm.n_layers,
        "d_model": lm.d_model,
        "n_pairs": len(harmful_prompts),
        "diff_norm": diff_norm,
        "rhat_norm": float(np.linalg.norm(rhat)),
        "prompt_set_hash": prompt_set_hash(harmful_prompts, harmless_prompts),
        "refusal_filtered": bool(args.refusal_filter),
        "refusal_filter_note": filtered_note,
        "split_half_cosine": cos,
        "projection_separation": sep,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(ARTIFACTS / f"{base}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"  ‖r̂‖              {meta['rhat_norm']:.8f}")
    print(f"  ‖d‖              {diff_norm:.3f}")
    print(f"  split-half cos   {cos:.4f}   (want > 0.8)")
    print(f"  separation       {sep['separation_sds']:.2f} SDs   (want > 1.0)")
    print(f"  harmless·r̂       {sep['harmless_mean']:+.3f}")
    print(f"  harmful·r̂        {sep['harmful_mean']:+.3f}")
    print()
    print(f"wrote {ARTIFACTS / (base + '.npz')} and {base}.json")


if __name__ == "__main__":
    main()
