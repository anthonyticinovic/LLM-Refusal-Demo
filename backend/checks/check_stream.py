"""Verify /generate streams incrementally rather than buffering.

The programmatic equivalent of `curl -N http://127.0.0.1:8000/generate ...` and
watching output trickle in. Worth automating because the failure mode is
invisible in the UI on a fast machine and fatal on a slow one: if any layer
buffers, the client receives one blob at the end and the "watch it think" part
of the demo silently stops existing.

The test is arrival TIMING, not content. A response can contain every token and
still have been delivered as a single write.

    python -m backend.app          # in one shell
    python -m backend.checks.check_stream   # in another
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time

MIN_EVENTS = 5
MIN_SPREAD_S = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="Explain how tides work.")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    body = json.dumps({
        "prompt": args.prompt, "alpha": args.alpha, "max_new_tokens": args.max_new_tokens,
    })

    try:
        conn = http.client.HTTPConnection(args.host, args.port, timeout=300)
        t0 = time.time()
        conn.request("POST", "/generate", body, {"Content-Type": "application/json"})
        res = conn.getresponse()
    except OSError as e:
        print(f"FAIL  cannot reach http://{args.host}:{args.port} — is `python -m backend.app` running? ({e})")
        return 1

    if res.status != 200:
        print(f"FAIL  HTTP {res.status}")
        return 1
    ctype = res.getheader("Content-Type", "")
    print(f"HTTP {res.status}  Content-Type: {ctype}")

    arrivals, tokens, kinds = [], [], []
    while True:
        line = res.readline()
        if not line:
            break
        line = line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        arrivals.append(time.time() - t0)
        kinds.append(ev.get("type"))
        if ev.get("type") == "token":
            tokens.append(ev["token"])
        elif ev.get("type") == "done":
            print(f"\ngenerated: {ev['text'][:160]!r}")
            print(f"{ev['n_tokens']} tokens in {ev['seconds']}s ({ev['tokens_per_second']} tok/s)")

    spread = (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]

    print(f"\n{len(arrivals)} SSE events; first at {arrivals[0]:.2f}s, last at {arrivals[-1]:.2f}s")
    print(f"inter-event gaps: min {min(gaps) * 1000:.0f}ms, median {sorted(gaps)[len(gaps) // 2] * 1000:.0f}ms, "
          f"max {max(gaps) * 1000:.0f}ms" if gaps else "")

    checks = [
        ("content type is text/event-stream", ctype.startswith("text/event-stream"), ctype),
        (f"received >= {MIN_EVENTS} events", len(arrivals) >= MIN_EVENTS, f"{len(arrivals)} events"),
        ("stream opens with a meta event", kinds[:1] == ["meta"], f"first = {kinds[:1]}"),
        ("stream ends with a done event", kinds[-1:] == ["done"], f"last = {kinds[-1:]}"),
        (
            f"delivery is incremental (spread >= {MIN_SPREAD_S}s)",
            spread >= MIN_SPREAD_S,
            f"{spread:.2f}s between first and last event"
            + ("  <-- looks buffered into one write" if spread < MIN_SPREAD_S else ""),
        ),
    ]

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")
        failed += not ok
    print()
    print(f"{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
