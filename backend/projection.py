"""The fixed 3D frame the demo draws the real activations in.

    x = h · r̂       the refusal direction itself
    y = h · u₁      strongest direction left once r̂ is gone
    z = h · u₂      second strongest

u₁ and u₂ are the leading principal components of the activations after r̂ has
been projected out, so all three axes are mutually orthogonal by construction.

That choice is the honest one. The claim on screen is "the classes separate
along THIS direction and are otherwise mixed", and using the *strongest*
remaining directions as the other two axes is the hardest available test of it:
if the classes were going to separate anywhere else, it would show up here
first. On Qwen2.5-3B layer 21 they separate 3.94 SD along r̂ and 0.00 SD along
both others, while the three axes carry comparable variance (26% / 21% / 14%),
so the cloud reads as a genuine 3D blob rather than a flat sheet.

This is a chosen frame, not a discovered structure, and the page says so. It is
deliberately not a PCA/UMAP/t-SNE switcher — see CLAUDE.md's "explicitly not
built" list. One fixed frame, built to show one direction.

Coordinates are plain (uncentred) dot products, so a live generation's
projection — which hooks.ProjectionTrace reports the same way — lands on the
same scale as the cached points and can be drawn in the same axes.

Everything here reads the existing extraction artifact. extract.py already
caches every layer's activations, so building this frame costs no forward passes
and needs no re-extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

ARTIFACTS = Path(__file__).parent / "artifacts"

# Axes besides r̂. Two gives a rotatable 3D cloud; the demo uses no more.
N_LEFTOVER_AXES = 2


def leftover_axes(acts: np.ndarray, rhat: np.ndarray, k: int = N_LEFTOVER_AXES) -> np.ndarray:
    """Top-`k` directions of `acts` once the r̂ component is removed.

    `acts` is (n, d) at a single layer. Returns (k, d), each row a unit vector
    orthogonal to r̂ and to the others.

    The explicit re-orthogonalisation against r̂ is not redundant: the
    activations arrive as float32 and the SVD runs in float64, and without it
    the residual |u·r̂| sits around 1e-9 rather than 1e-16. That is small, but it
    is exactly the quantity check_projection asserts on.
    """
    x = acts.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    r = rhat.astype(np.float64)
    x = x - np.outer(x @ r, r)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    out = []
    for i in range(k):
        u = vt[i] - (vt[i] @ r) * r
        for prev in out:                      # SVD rows are already orthogonal;
            u = u - (u @ prev) * prev         # this holds it after the r̂ pass.
        out.append(u / np.linalg.norm(u))
    return np.stack(out)


def load_plane(name: str = "direction", layer: Optional[int] = None) -> dict:
    """Build the frame from the extraction artifact.

    Returns r̂, the leftover axes, and the (x, y, z) coordinates of the cached
    harmful/harmless activations, ready to hand to the front end.
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

    axes = leftover_axes(np.concatenate([harmful, harmless], axis=0), rhat)
    frame = np.vstack([rhat[None, :], axes])          # (3, d): r̂, u₁, u₂

    def coords(a: np.ndarray) -> list:
        return [[float(v) for v in row] for row in (a @ frame.T)]

    return {
        "rhat": rhat,
        "axes": axes,
        "layer": layer,
        "harmful": coords(harmful),
        "harmless": coords(harmless),
        "model_id": meta["model_id"],
        "d_model": meta["d_model"],
    }
