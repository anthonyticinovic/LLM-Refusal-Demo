"""Print raw generations so a human can audit the refusal matcher.

The causal numbers are only as trustworthy as the substring matcher producing
them, and that matcher is crude by design. This prints what the model actually
said next to what the matcher decided, so its mistakes are visible rather than
inferred. Run it at least once per model.

    python -m backend.checks.spot_check                # a sample of each condition
    python -m backend.checks.spot_check --all          # everything
    python -m backend.checks.spot_check --disagree     # likeliest matcher errors

`--disagree` is the highest-value view: it surfaces generations where the
matcher's verdict is least likely to be right — refusals detected only by a
weak phrase, and compliances that hedge heavily without ever tripping a phrase.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..extract import ARTIFACTS

HEDGE_WORDS = ("however", "important to note", "i'd caution", "be careful",
               "not advisable", "illegal", "unethical", "responsibly")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--disagree", action="store_true")
    ap.add_argument("--per-condition", type=int, default=3)
    args = ap.parse_args()

    path = ARTIFACTS / "eval_generations.json"
    if not path.exists():
        print(f"{path} missing — run `python -m backend.checks.check_causal` first.")
        return 1
    with open(path) as f:
        data = json.load(f)

    records = data["records"]

    if args.disagree:
        # Refusals resting on a single short phrase, and non-refusals that
        # hedge. Both are where the matcher is most likely to be wrong.
        picked = [
            r for r in records
            if (r["refusal"] and len(r.get("matched_phrase") or "") <= 8)
            or (not r["refusal"] and any(w in r["text"].lower() for w in HEDGE_WORDS))
        ]
        print(f"{len(picked)} generations where the matcher is most likely wrong\n")
    elif args.all:
        picked = records
    else:
        picked, seen = [], {}
        for r in records:
            key = (r["layer"], r["set"], r["condition"])
            if seen.get(key, 0) < args.per_condition:
                picked.append(r)
                seen[key] = seen.get(key, 0) + 1

    current = None
    for r in picked:
        key = (r["layer"], r["set"], r["condition"])
        if key != current:
            current = key
            print("=" * 78)
            print(f"layer {r['layer']}  |  {r['set']}  |  {r['condition']}  (alpha={r['alpha']:+.1f})")
            print("=" * 78)
        verdict = "REFUSAL" if r["refusal"] else "COMPLY "
        phrase = f"  [{r['matched_phrase']}]" if r["matched_phrase"] else ""
        print(f"\n{verdict}{phrase}")
        print(f"  Q: {r['prompt']}")
        print(f"  A: {r['text'].strip()[:400]}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
