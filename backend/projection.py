"""The fixed 2D plane the presenter demo draws the real activations in.

    x = h · r̂      the refusal direction itself
    y = h · u      the leading direction of what is left once r̂ is gone

u is the first principal component of the activations after r̂ has been projected
out, so it is orthogonal to r̂ by construction. That choice is the honest one
here: the claim being made on screen is "the classes separate along THIS
direction and are otherwise mixed", and picking y as the *strongest* remaining
signal is the hardest available test of that claim in two dimensions. Choosing
any weaker axis would flatter the result.

This is a chosen projection, not a discovered structure, and the page says so.
It is deliberately not a PCA/UMAP/t-SNE switcher — see CLAUDE.md's "explicitly
not built" list. One fixed plane, built to show one direction.

Both coordinates are plain (uncentred) dot products, so a live generation's
projection — which hooks.ProjectionTrace reports the same way — lands on the
same scale as the cached points and can be drawn in the same axes.

Everything here reads the existing extraction artifact. extract.py already
caches every layer's activations, so building this plane costs no forward passes
and needs no re-extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

ARTIFACTS = Path(__file__).parent / "artifacts"


def second_axis(acts: np.ndarray, rhat: np.ndarray) -> np.ndarray:
    """Leading direction of `acts` once the r̂ component is removed.

    `acts` is (n, d) at a single layer. Returns a unit vector orthogonal to r̂.

    The explicit re-orthogonalisation on the last line is not redundant. The
    activations arrive as float32 and the SVD runs in float64; without it the
    residual |u·r̂| sits around 1e-7 rather than 1e-16, which is small but is
    exactly the quantity check_projection asserts on.
    """
    x = acts.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    r = rhat.astype(np.float64)
    x = x - np.outer(x @ r, r)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    u = vt[0]
    u = u - (u @ r) * r
    return u / np.linalg.norm(u)


def load_plane(name: str = "direction", layer: Optional[int] = None) -> dict:
    """Build the plane from the extraction artifact.

    Returns r̂, u, and the plane coordinates of the cached harmful/harmless
    activations, ready to hand to the front end.
    """
    npz_path = ARTIFACTS / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found — run `python -m backend.extract` first."
        )
    with open(ARTIFACTS / f"{name}.json") as f:
        meta = json.load(f)
    layer = meta["layer"] if layer is None else layer

    npz = np.load(npz_path)
    harmful = npz["acts_harmful"][:, layer, :]
    harmless = npz["acts_harmless"][:, layer, :]
    rhat = npz["rhat"].astype(np.float64)

    u = second_axis(np.concatenate([harmful, harmless], axis=0), rhat)

    def coords(a: np.ndarray) -> list:
        return [[float(x), float(y)] for x, y in zip(a @ rhat, a @ u)]

    return {
        "rhat": rhat,
        "u": u,
        "layer": layer,
        "harmful": coords(harmful),
        "harmless": coords(harmless),
        "model_id": meta["model_id"],
        "d_model": meta["d_model"],
    }
