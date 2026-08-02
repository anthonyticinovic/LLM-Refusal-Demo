"""Local API + static file server for the Layer 2 demo.

Bound to 127.0.0.1 and nothing else. Serving the frontend from here rather than
opening it over file:// is what makes CORS a non-issue: same origin, no
preflight, no headers to configure.

    python -m backend.app                 # http://127.0.0.1:8000/live.html

The model is loaded once at startup and held for the process lifetime. Changing
alpha only changes which hooks are registered for the next generation, so the
slider takes effect immediately with no reload — see hooks.apply.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import hooks as hooks_mod
from . import model as model_mod
from .generate import load_direction, stream_generate
from .refusal import is_refusal

FRONTEND = Path(__file__).parent.parent / "frontend"

STATE: dict = {}

# One model, one set of hooks, one process. Hooks are registered on shared
# module objects, so two generations running at once would see each other's
# interventions. Serialise rather than pretend otherwise — this is a local
# single-user demo and a queue is the honest design.
GEN_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    lm = model_mod.load()
    STATE["lm"] = lm
    print(f"[startup] {lm.model_id} on {lm.device} ({lm.dtype}) in {time.time() - t0:.1f}s")
    try:
        STATE["direction"] = load_direction(device=str(lm.model.device))
        d = STATE["direction"]
        print(f"[startup] direction: layer {d.layer}, ‖d‖ = {d.diff_norm:.2f}, "
              f"from {d.meta['n_pairs']} pairs of {d.meta['model_id']}")
        if d.meta["model_id"] != lm.model_id:
            print(f"[startup] WARNING: direction was extracted from {d.meta['model_id']} "
                  f"but the loaded model is {lm.model_id}. Re-run extraction.")
    except FileNotFoundError as e:
        STATE["direction"] = None
        print(f"[startup] no direction artifact: {e}")
    yield


app = FastAPI(title="Refusal direction demo", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    alpha: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_new_tokens: int = Field(default=48, ge=1, le=256)


@app.get("/api/meta")
def meta():
    lm = STATE["lm"]
    d = STATE.get("direction")
    return {
        "model_id": lm.model_id,
        "device": lm.device,
        "dtype": str(lm.dtype),
        "n_layers": lm.n_layers,
        "d_model": lm.d_model,
        "direction": None if d is None else {
            "layer": d.layer,
            "diff_norm": d.diff_norm,
            "n_pairs": d.meta["n_pairs"],
            "split_half_cosine": d.meta.get("split_half_cosine"),
            "separation_sds": (d.meta.get("projection_separation") or {}).get("separation_sds"),
            "extracted_from": d.meta["model_id"],
        },
        "alpha_mapping": {
            "negative": "ablation, strength |alpha|, applied at every block: h - t(h.r)r",
            "zero": "baseline, no hooks",
            "positive": f"injection at layer {d.layer if d else '?'}: "
                        f"h + alpha * {hooks_mod.INJECT_MAX_MULT} * ||d|| * r",
            "inject_max_mult": hooks_mod.INJECT_MAX_MULT,
        },
    }


@app.get("/api/prompts")
def prompts():
    """Sample prompts for the UI, so a first-time viewer has somewhere to start."""
    out = {}
    for key, fname in (("harmful", "heldout_harmful.json"), ("harmless", "heldout_harmless.json")):
        with open(Path(__file__).parent / "prompts" / fname) as f:
            out[key] = json.load(f)["prompts"][:8]
    return out


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/generate")
def generate(req: GenerateRequest):
    """Stream {token, projection} events as Server-Sent Events."""

    def event_stream():
        lm = STATE["lm"]
        direction = STATE.get("direction")
        acquired = GEN_LOCK.acquire(timeout=180)
        if not acquired:
            yield _sse({"type": "error", "message": "generation busy, try again"})
            return
        try:
            iv_desc = "baseline"
            if direction is not None and req.alpha != 0:
                iv = hooks_mod.resolve_alpha(req.alpha, direction.rhat, direction.layer, direction.diff_norm)
                iv_desc = (f"ablate t={iv.ablation:.2f} (all blocks)" if iv.ablation
                           else f"inject c={iv.injection:+.2f} (layer {iv.layer})")

            yield _sse({
                "type": "meta",
                "alpha": req.alpha,
                "intervention": iv_desc,
                "layer": direction.layer if direction else None,
            })

            t0 = time.time()
            text, n = "", 0
            for ev in stream_generate(lm, req.prompt, req.alpha, direction, req.max_new_tokens):
                text += ev["token"]
                n += 1
                yield _sse({"type": "token", **ev})

            elapsed = time.time() - t0
            yield _sse({
                "type": "done",
                "text": text,
                "n_tokens": n,
                "seconds": round(elapsed, 2),
                "tokens_per_second": round(n / elapsed, 2) if elapsed > 0 else None,
                # Reported so the UI can label it, with the caveat it deserves:
                # this is the same crude substring matcher the evals use.
                "refusal_heuristic": is_refusal(text),
            })
        finally:
            GEN_LOCK.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Defeats proxy buffering, which would otherwise deliver the whole
            # stream as one blob at the end and quietly break the demo.
            "X-Accel-Buffering": "no",
        },
    )


# Mounted last: a catch-all mount would shadow the API routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="local only by default, and it should stay that way")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"serving http://{args.host}:{args.port}/live.html")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
