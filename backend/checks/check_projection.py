"""Verify the demo's 3D frame is what it claims to be.

The demo asserts on screen that one axis is the refusal direction and the other
two are "everything else". That is only honest if the other two carry no r̂
component, so this checks orthogonality directly.

The last check is the one that matters: the classes must separate along r̂ and
essentially not at all along the strongest remaining directions. That is the
"single direction" claim stated as an inequality rather than shown as a picture.
If refusal were spread across several directions, the leftover axes would pick
it up and this would fail.

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
    """Mean gap between two samples, in pooled standard deviations."""
    pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def main() -> int:
    try:
        plane = load_plane()
    except FileNotFoundError as e:
        print(f"  FAIL  {e}")
        return 1

    rhat, axes = plane["rhat"], plane["axes"]
    hf = np.array(plane["harmful"])
    hl = np.array(plane["harmless"])

    print(f"model        {plane['model_id']}")
    print(f"layer        {plane['layer']}   d_model {plane['d_model']}")
    print(f"points       {len(hf)} harmful, {len(hl)} harmless\n")

    worst_r = max(float(abs(u @ rhat)) for u in axes)
    check("leftover axes are orthogonal to r-hat", worst_r < 1e-5,
          f"max |u . rhat| = {worst_r:.2e}")

    worst_uu = 0.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            worst_uu = max(worst_uu, float(abs(axes[i] @ axes[j])))
    check("leftover axes are orthogonal to each other", worst_uu < 1e-5,
          f"max |u_i . u_j| = {worst_uu:.2e}")

    worst_n = max(abs(float(np.linalg.norm(u)) - 1) for u in axes)
    check("leftover axes are unit length", worst_n < 1e-5, f"max |‖u‖ − 1| = {worst_n:.2e}")

    xsep = separation(hf[:, 0], hl[:, 0])
    check("classes separate along r-hat by > 1 SD", xsep > 1.0,
          f"{xsep:+.2f} SDs (harmless {hl[:, 0].mean():+.2f}, harmful {hf[:, 0].mean():+.2f})")

    others = [abs(separation(hf[:, i + 1], hl[:, i + 1])) for i in range(len(axes))]
    worst_other = max(others)
    check("classes barely separate along the leftover axes", worst_other < xsep / 2,
          "r-hat %+.2f SD vs %s" % (xsep, ", ".join(f"u{i+1} {v:.2f} SD"
                                                    for i, v in enumerate(others))))

    passed = sum(CHECKS)
    print(f"\n{passed}/{len(CHECKS)} passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
