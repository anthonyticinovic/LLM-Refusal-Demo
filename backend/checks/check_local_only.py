"""Verify the project is local-only: no keys, no cloud endpoints, works offline.

Two parts.

1. A static scan for credentials and remote inference endpoints. Downloading
   weights from the HF hub once is permitted, so huggingface.co is allowlisted;
   anything else that looks like a remote call is reported.

2. A real offline generation. `HF_HUB_OFFLINE` alone only proves the hub client
   was told to stay quiet, so this also points the proxy environment variables
   at a closed port — any process that genuinely attempts an outbound HTTP
   request fails loudly instead of silently succeeding. Passing means
   generation ran with every network path broken.

    python -m backend.checks.check_local_only
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCAN_SUFFIXES = {".py", ".html", ".js", ".json", ".txt", ".md", ".toml", ".yaml", ".yml", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "artifacts"}

# Allowed to appear: the weight host, and loopback.
ALLOW = (
    "huggingface.co",
    "hf.co",
    "127.0.0.1",
    "localhost",
    "0.0.0.0",
)

PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{16,}")),
    ("Anthropic-style key", re.compile(r"\bsk-ant-[A-Za-z0-9\-]{16,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("HF token literal", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("bearer literal", re.compile(r"Authorization[\"']?\s*[:=]\s*[\"']Bearer\s+\S")),
    ("assigned api key", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*[\"'][^\"'\s]{8,}")),
]

URL_RE = re.compile(r"https?://[^\s\"'`<>)\]]+")


def scan() -> tuple[list[str], list[str]]:
    """Returns (problems, templated). Templated URLs contain a format
    placeholder, so they are not literal endpoints and cannot be resolved
    statically. They are surfaced for a human to eyeball rather than failing
    the check — but they are surfaced, not dropped."""
    problems: list[str] = []
    templated: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)

        for label, pattern in PATTERNS:
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                problems.append(f"{rel}:{line}  {label}: {m.group(0)[:40]}")

        for m in URL_RE.finditer(text):
            url = m.group(0)
            if any(a in url for a in ALLOW):
                continue
            line = text[: m.start()].count("\n") + 1
            if "{" in url or "}" in url:
                templated.append(f"{rel}:{line}  {url[:80]}")
                continue
            problems.append(f"{rel}:{line}  non-allowlisted URL: {url[:80]}")
    return problems, templated


def offline_generation() -> tuple[bool, str]:
    """Generate with the hub offline AND all proxies pointed at a dead port."""
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # Port 9 is discard; nothing listens. Any real outbound HTTP dies here.
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "TOKENIZERS_PARALLELISM": "false",
    })
    code = (
        "from backend import model as M;"
        "from backend.generate import generate_text;"
        "lm = M.load();"
        "print('OFFLINE_OK:' + generate_text(lm, 'Why is the sky blue?', max_new_tokens=8).strip()[:60])"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out"
    for line in proc.stdout.splitlines():
        if line.startswith("OFFLINE_OK:"):
            return True, line[len("OFFLINE_OK:"):]
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return False, tail[-1] if tail else "no output"


def main() -> int:
    print(f"scanning {ROOT}")
    problems, templated = scan()
    if problems:
        print(f"\n  FAIL  {len(problems)} finding(s):")
        for p in problems:
            print(f"        {p}")
    else:
        print("\n  PASS  no credentials or non-allowlisted URLs "
              "(huggingface.co and loopback are allowlisted)")
    if templated:
        print(f"\n  note: {len(templated)} templated URL(s), not literal endpoints — "
              f"confirm the defaults are loopback:")
        for t in templated:
            print(f"        {t}")

    print("\n  running generation with hub offline and proxies pointed at a dead port...")
    ok, detail = offline_generation()
    print(f"  {'PASS' if ok else 'FAIL'}  offline generation  ({detail})")

    failed = bool(problems) + (not ok)
    print(f"\n{2 - failed}/2 passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
