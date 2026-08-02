"""Run the whole verification sequence and print one summary.

Intended as the first thing to run on a new machine or after switching models:

    export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct
    python -m backend.checks.run_all --extract

`--extract` re-runs extraction first, which you want whenever the model or the
prompt sets changed. Without it, the existing artifact is reused.

check_stream is not included: it needs a running server, and starting one here
would make this script own a subprocess whose failure modes are more annoying
than the check is worth. Run it separately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(label: str, args: list[str]) -> tuple[str, bool, float]:
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", *args], cwd=ROOT)
    return label, proc.returncode == 0, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="re-run extraction first")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--refusal-filter", action="store_true",
                    help="passed through to extract.py; see its help")
    ap.add_argument("--screen", action="store_true",
                    help="passed through to check_causal.py; see its help")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    results = []

    if args.extract:
        cmd = ["backend.extract"]
        if args.layer is not None:
            cmd += ["--layer", str(args.layer)]
        if args.refusal_filter:
            cmd += ["--refusal-filter"]
        results.append(run("extract direction", cmd))
        if not results[-1][1]:
            print("\nextraction failed — stopping, everything downstream depends on it")
            return 1

    results.append(run("direction quality", ["backend.checks.check_direction"]))

    causal = ["backend.checks.check_causal", "--max-new-tokens", str(args.max_new_tokens)]
    if args.screen:
        causal += ["--screen"]
    results.append(run("causal effect", causal))

    results.append(run("local-only + offline", ["backend.checks.check_local_only"]))

    print(f"\n{'=' * 72}\n  SUMMARY\n{'=' * 72}")
    for label, ok, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<28} {secs:6.1f}s")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} suites passed")
    print("\nstill to run by hand:")
    print("  python -m backend.app                    # then, in another shell:")
    print("  python -m backend.checks.check_stream")
    print("  python -m backend.checks.spot_check --disagree")
    print("  open 'frontend/index.html?selftest=1'")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
