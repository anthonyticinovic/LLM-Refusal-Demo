"""Verify the presenter demo's projection plane is what it claims to be.

The demo asserts on screen that x is the refusal direction and y is "everything
else". That is only honest if y genuinely carries no r̂ component, so this checks
orthogonality directly, plus that the plane still separates the two classes
along x — a projection that lost the separation would be showing the room
nothing.

The last check is the interesting one: it asserts the classes separate along r̂
far more than along the strongest remaining direction. That is the claim the
whole demo rests on, stated as an inequality rather than a picture.

    python -m backend.checks.check_projection
"""

from __future__ import annotations

import sys

import numpy as np

from ..projection import load_plane

CHECKS: list = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Mean gap between two samples in pooled standard deviations."""
    pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def main() -> int:
    try:
        plane = load_plane()
    except FileNotFoundError as e:
        print(f"  FAIL  {e}")
        return 1

    rhat, u = plane["rhat"], plane["u"]
    print(f"model        {plane['model_id']}")
    print(f"layer        {plane['layer']}   d_model {plane['d_model']}")
    print(f"points       {len(plane['harmful'])} harmful, {len(plane['harmless'])} harmless\n")

    dot = float(abs(rhat @ u))
    check("u is orthogonal to r-hat", dot < 1e-5, f"|u . rhat| = {dot:.2e}")

    unorm = float(np.linalg.norm(u))
    check("||u|| = 1 within 1e-5", abs(unorm - 1) < 1e-5, f"||u|| = {unorm:.8f}")

    hx = np.array([p[0] for p in plane["harmful"]])
    bx = np.array([p[0] for p in plane["harmless"]])
    xsep = separation(hx, bx)
    check("classes separate along x by > 1 SD", xsep > 1.0,
          f"{xsep:.2f} SDs (harmless {bx.mean():+.2f}, harmful {hx.mean():+.2f})")

    hy = np.array([p[1] for p in plane["harmful"]])
    by = np.array([p[1] for p in plane["harmless"]])
    ysep = abs(separation(hy, by))
    check("y separates the classes far less than x", ysep < xsep,
          f"y = {ysep:.2f} SDs vs x = {xsep:.2f} SDs")

    passed = sum(CHECKS)
    print(f"\n{passed}/{len(CHECKS)} passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
