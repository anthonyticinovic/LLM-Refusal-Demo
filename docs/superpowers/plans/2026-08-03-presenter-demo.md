# Presenter demo implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `frontend/demo.html` — a three-act, keyboard-stepped presenter demo, including a new 2D view of the *real* model's activations.

**Architecture:** A new self-contained HTML page served by the existing FastAPI app. A new `backend/projection.py` computes a fixed 2D plane (x = r̂, y = leading leftover direction) from the activations already cached in `direction.npz`, so no re-extraction is required. `hooks.ProjectionTrace` gains an optional second axis so the live generation can report its position in that plane. `index.html` and `live.html` are not touched.

**Tech Stack:** Python 3.12, NumPy, PyTorch, FastAPI/SSE, vanilla JS + canvas 2D. No build step, no CDN.

---

## File structure

| File | Responsibility |
|---|---|
| `backend/projection.py` (create) | The 2D plane maths: second axis `u`, and plane coordinates for the cached sets. Pure NumPy, no torch, no I/O beyond loading the artifact. |
| `backend/checks/check_projection.py` (create) | Self-test: `u ⟂ r̂`, `‖u‖ = 1`, and the two classes separate along x. |
| `backend/hooks.py` (modify) | `ProjectionTrace` optionally records a second axis alongside `pre`/`post`. |
| `backend/generate.py` (modify) | Emit `y` on each token event when a second axis is supplied. |
| `backend/app.py` (modify) | Serve `/api/projection`; pass the second axis into the trace. |
| `frontend/demo.html` (create) | The three-act presenter page. Self-contained. |
| `backend/checks/run_all.py` (modify) | Include the projection check. |
| `README.md` (modify) | Document how to run the presenter demo. |

**Palette** (used throughout `demo.html`): paper `#EBEBE4`, stage `#F4F4EE`, border `#DCDCD2`, ink `#2A2C2C`, muted `#6E706A`, faint `#9A9B92`, harmless `#6E93AE`, harmful `#C67E60`, r̂ `#7C9E77`.

---

### Task 1: The projection plane

**Files:**
- Create: `backend/projection.py`
- Test: `backend/checks/check_projection.py`

- [ ] **Step 1: Write the failing check**

Create `backend/checks/check_projection.py`:

```python
"""Verify the Act 2 projection plane is what it claims to be.

The demo asserts on screen that x is the refusal direction and y is "everything
else". That is only honest if y genuinely carries no r̂ component, so this checks
orthogonality directly, plus that the plane still separates the two classes
along x — a projection that lost the separation would be showing the room
nothing.

    python -m backend.checks.check_projection
"""

from __future__ import annotations

import sys

import numpy as np

from ..projection import load_plane

CHECKS = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")


def main() -> int:
    plane = load_plane()
    rhat, u = plane["rhat"], plane["u"]

    dot = float(abs(rhat @ u))
    check("u is orthogonal to r-hat", dot < 1e-5, f"|u . rhat| = {dot:.2e}")

    unorm = float(np.linalg.norm(u))
    check("||u|| = 1 within 1e-5", abs(unorm - 1) < 1e-5, f"||u|| = {unorm:.8f}")

    hx = np.array([p[0] for p in plane["harmful"]])
    bx = np.array([p[0] for p in plane["harmless"]])
    pooled = float(np.sqrt((hx.var(ddof=1) + bx.var(ddof=1)) / 2))
    sep = float((hx.mean() - bx.mean()) / pooled) if pooled > 0 else 0.0
    check("classes separate along x by > 1 SD", sep > 1.0,
          f"{sep:.2f} SDs (harmless {bx.mean():+.2f}, harmful {hx.mean():+.2f})")

    hy = np.array([p[1] for p in plane["harmful"]])
    by = np.array([p[1] for p in plane["harmless"]])
    ypooled = float(np.sqrt((hy.var(ddof=1) + by.var(ddof=1)) / 2))
    ysep = abs(float((hy.mean() - by.mean()) / ypooled)) if ypooled > 0 else 0.0
    check("y separates the classes far less than x", ysep < sep,
          f"y = {ysep:.2f} SDs vs x = {sep:.2f} SDs")

    passed = sum(CHECKS)
    print(f"\n{passed}/{len(CHECKS)} passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m backend.checks.check_projection`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.projection'`

- [ ] **Step 3: Write the implementation**

Create `backend/projection.py`:

```python
"""The fixed 2D plane the presenter demo draws the real activations in.

    x = h . r-hat      the refusal direction itself
    y = h . u          the leading direction of what is left once r-hat is gone

u is the first principal component of the activations after r-hat has been
projected out, so it is orthogonal to r-hat by construction. That choice is the
honest one for this demo: the point being made is "the classes separate along
THIS direction and are otherwise mixed", and picking y as the strongest
remaining signal is the hardest test of that claim available in two dimensions.

This is a chosen projection, not a discovered structure, and the page says so.

Both coordinates are plain (uncentred) dot products so that a live generation's
projection, which hooks.ProjectionTrace reports the same way, lands on the same
scale as the cached points.

Everything here reads the existing extraction artifact — extract.py already
caches every layer's activations, so no re-extraction is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

ARTIFACTS = Path(__file__).parent / "artifacts"


def second_axis(acts: np.ndarray, rhat: np.ndarray) -> np.ndarray:
    """Leading direction of `acts` once the r-hat component is removed.

    `acts` is (n, d) at a single layer. Returns a unit vector orthogonal to
    r-hat. The explicit re-orthogonalisation at the end is not redundant: the
    SVD is computed in float64 but the inputs arrive as float32, and without it
    the residual dot product sits around 1e-7 rather than 1e-16.
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

    Returns rhat, u, and the plane coordinates of the cached harmful/harmless
    activations, ready to be handed to the front end.
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
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `python -m backend.checks.check_projection`
Expected: `4/4 passed`, with `|u . rhat|` around 1e-16 and x separation above 1 SD.

- [ ] **Step 5: Commit**

```bash
git add backend/projection.py backend/checks/check_projection.py
git commit -m "feat(projection): fixed 2D plane over the cached activations"
```

---

### Task 2: Report the second coordinate during generation

**Files:**
- Modify: `backend/hooks.py` (the `ProjectionTrace` dataclass and the hook body in `apply`)
- Modify: `backend/generate.py` (the event dict in `stream_generate`)

- [ ] **Step 1: Extend `ProjectionTrace`**

In `backend/hooks.py`, replace the `ProjectionTrace` dataclass with:

```python
@dataclass
class ProjectionTrace:
    """Per-forward-call projection onto r̂ at the extraction layer.

    Two series are recorded because they answer different questions:

      `pre`  — the projection the model itself computed, before we touched it.
               Under full ablation this is NOT zero: earlier blocks were
               cleaned, so this measures how much refusal direction the model
               writes back at this layer for this token. That is the
               interesting one.
      `post` — what actually flowed onward after the intervention. Under full
               ablation it is ~0 by construction, which is the useful sanity
               check that the hooks fired at all.

    `uhat`, when supplied, is the presenter demo's second plane axis; `pre_y`
    then tracks the token's position along it so the live point can be drawn in
    the same plane as the cached activations. It is recorded pre-intervention
    to match `pre`, and because injection moves the point along r̂ alone.
    """

    pre: List[float] = field(default_factory=list)
    post: List[float] = field(default_factory=list)
    uhat: Optional[torch.Tensor] = None
    pre_y: List[float] = field(default_factory=list)

    def reset(self) -> None:
        self.pre.clear()
        self.post.clear()
        self.pre_y.clear()
```

- [ ] **Step 2: Record it in the hook**

In `backend/hooks.py`, inside `apply`'s `hook` function, extend the existing pre-intervention trace block. Replace:

```python
                    if is_target and trace is not None:
                        # Last position only: during prefill that is the final
                        # prompt token, during decode it is the token just
                        # produced. One entry per forward call either way.
                        trace.pre.append(float((h[0, -1].float() @ rhat.float()).item()))
```

with:

```python
                    if is_target and trace is not None:
                        # Last position only: during prefill that is the final
                        # prompt token, during decode it is the token just
                        # produced. One entry per forward call either way.
                        last = h[0, -1].float()
                        trace.pre.append(float((last @ rhat.float()).item()))
                        if trace.uhat is not None:
                            trace.pre_y.append(float((last @ trace.uhat.float()).item()))
```

- [ ] **Step 3: Emit it from the stream**

In `backend/generate.py`, in `stream_generate`, replace the `yield` dict:

```python
            yield {
                "token": piece,
                "index": step,
                "projection": trace.pre[step] if trace and step < len(trace.pre) else 0.0,
                "projection_post": trace.post[step] if trace and step < len(trace.post) else 0.0,
            }
```

with:

```python
            yield {
                "token": piece,
                "index": step,
                "projection": trace.pre[step] if trace and step < len(trace.pre) else 0.0,
                "projection_post": trace.post[step] if trace and step < len(trace.post) else 0.0,
                "y": trace.pre_y[step] if trace and step < len(trace.pre_y) else None,
            }
```

- [ ] **Step 4: Verify nothing regressed**

Run: `python -m backend.checks.check_direction`
Expected: `5/5 passed` (unchanged — this touches no extraction maths).

Then start the server in one shell and run the stream check in another:

Run: `python -m backend.app` then `python -m backend.checks.check_stream`
Expected: `5/5 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/hooks.py backend/generate.py
git commit -m "feat(hooks): trace a second plane axis during generation"
```

---

### Task 3: Serve the plane

**Files:**
- Modify: `backend/app.py`

- [ ] **Step 1: Load the plane at startup**

In `backend/app.py`, inside `lifespan`, after the existing `STATE["direction"]` try/except block and before `yield`, add:

```python
    try:
        from .projection import load_plane

        STATE["plane"] = load_plane()
        print(f"[startup] plane: {len(STATE['plane']['harmful'])} harmful, "
              f"{len(STATE['plane']['harmless'])} harmless points")
    except FileNotFoundError:
        STATE["plane"] = None
```

- [ ] **Step 2: Add the endpoint**

In `backend/app.py`, after the `prompts()` route, add:

```python
@app.get("/api/projection")
def projection():
    """The Act 2 plane: cached activations in (r̂, u) coordinates.

    Static for a given artifact, so the front end fetches it once.
    """
    plane = STATE.get("plane")
    if plane is None:
        return {"available": False}
    return {
        "available": True,
        "layer": plane["layer"],
        "model_id": plane["model_id"],
        "d_model": plane["d_model"],
        "harmful": plane["harmful"],
        "harmless": plane["harmless"],
    }
```

- [ ] **Step 3: Pass the second axis into the trace**

The trace is constructed inside `generate.stream_generate`, so hand the axis in from there. In `backend/generate.py`, change the signature of `stream_generate` to accept it — replace:

```python
    max_new_tokens: int = 48,
) -> Iterator[dict]:
```

with:

```python
    max_new_tokens: int = 48,
    uhat: Optional[torch.Tensor] = None,
) -> Iterator[dict]:
```

and replace:

```python
    if direction is not None:
        trace = hooks_mod.ProjectionTrace()
```

with:

```python
    if direction is not None:
        trace = hooks_mod.ProjectionTrace(uhat=uhat)
```

Then in `backend/app.py`, build the tensor once at startup — inside the `try` block from Step 1, after `STATE["plane"] = load_plane()`, add:

```python
        import torch as _torch

        STATE["uhat"] = _torch.from_numpy(
            STATE["plane"]["u"].astype("float32")
        ).to(STATE["lm"].model.device)
```

and add `STATE["uhat"] = None` to the `except FileNotFoundError:` branch alongside `STATE["plane"] = None`.

Finally, in `backend/app.py`'s `event_stream`, replace:

```python
            for ev in stream_generate(lm, req.prompt, req.alpha, direction, req.max_new_tokens):
```

with:

```python
            for ev in stream_generate(lm, req.prompt, req.alpha, direction,
                                      req.max_new_tokens, uhat=STATE.get("uhat")):
```

`backend/generate.py` already imports `Optional` from `typing` and `torch`, so no new imports are needed there.

- [ ] **Step 4: Verify the endpoint**

Run: `python -m backend.app` in one shell, then in another:

```bash
curl -s http://127.0.0.1:8000/api/projection | python -c "import json,sys; d=json.load(sys.stdin); print(d['available'], d['layer'], len(d['harmful']), len(d['harmless']), d['harmful'][0])"
```

Expected: `True 21 40 40 [<x>, <y>]` with two floats.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/generate.py
git commit -m "feat(api): serve the Act 2 projection plane"
```

---

### Task 4: The presenter page shell

**Files:**
- Create: `frontend/demo.html`

- [ ] **Step 1: Build the shell with all three acts stubbed**

Create `frontend/demo.html`. Structure: a `<style>` block with the palette as CSS custom properties, three `<section class="act">` elements, a dot indicator, and a keyboard handler. Only one act is visible at a time (`.act.on`).

```html
<!DOCTYPE html>
<meta charset="utf-8">
<title>The refusal direction</title>
<style>
:root {
  --paper:#EBEBE4; --stage:#F4F4EE; --border:#DCDCD2;
  --ink:#2A2C2C; --muted:#6E706A; --faint:#9A9B92;
  --harmless:#6E93AE; --harmful:#C67E60; --rhat:#7C9E77;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  height:100vh; display:flex; flex-direction:column;
}
header { display:flex; justify-content:space-between; align-items:baseline;
         padding:22px 40px 0; }
h1 { font-size:15px; font-weight:500; color:var(--muted); margin:0; letter-spacing:.01em; }
.dots { display:flex; align-items:center; gap:10px; color:var(--faint); font-size:13px; }
.dot { width:8px; height:8px; border-radius:50%; background:#D2D2C7; }
.dot.on { background:var(--harmful); }
main { flex:1; display:flex; align-items:center; justify-content:center; padding:0 40px; }
.act { display:none; width:100%; max-width:1040px; }
.act.on { display:block; }
.act h2 { font-size:32px; font-weight:500; margin:0 0 6px; }
.act .sub { color:var(--muted); font-size:16px; margin:0 0 22px; }
.stage { background:var(--stage); border:1px solid var(--border); border-radius:16px; }
footer { padding:14px 40px 22px; color:var(--faint); font-size:13px;
         display:flex; justify-content:space-between; }
.note { color:var(--faint); font-size:13px; margin-top:14px; }
</style>

<header>
  <h1>Refusal is mediated by a single direction</h1>
  <div class="dots"><span id="stepnum">1 / 3</span>
    <span class="dot on"></span><span class="dot"></span><span class="dot"></span>
  </div>
</header>

<main>
  <section class="act on" id="act1"><h2>The idea</h2>
    <p class="sub">Two clouds of made-up points, and one direction through them.</p>
    <div class="stage" style="height:420px"></div>
    <p class="note">A toy: 3 dimensions, synthetic points. The arithmetic is the same one the real model gets.</p>
  </section>

  <section class="act" id="act2"><h2>It's real</h2>
    <p class="sub">The same picture, on Qwen2.5-3B's actual activations.</p>
    <div class="stage" style="height:420px"></div>
  </section>

  <section class="act" id="act3"><h2>The model obeys</h2>
    <p class="sub">One prompt, one slider, live.</p>
    <div class="stage" style="height:420px"></div>
  </section>
</main>

<footer><span>← → to move</span><span id="hint"></span></footer>

<script>
const acts = [...document.querySelectorAll('.act')];
const dots = [...document.querySelectorAll('.dot')];
let step = 0;

function show(i) {
  step = Math.max(0, Math.min(acts.length - 1, i));
  acts.forEach((a, n) => a.classList.toggle('on', n === step));
  dots.forEach((d, n) => d.classList.toggle('on', n === step));
  document.getElementById('stepnum').textContent = (step + 1) + ' / ' + acts.length;
  window.dispatchEvent(new CustomEvent('actshown', { detail: { step } }));
}

addEventListener('keydown', e => {
  if (e.target.matches('input,textarea')) return;
  if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); show(step + 1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); show(step - 1); }
});
show(0);
</script>
```

- [ ] **Step 2: Verify the shell**

Run: `python -m backend.app`, open `http://127.0.0.1:8000/demo.html`, press `→` twice and `←` twice.
Expected: the three acts swap, the dot indicator and `n / 3` track, nothing scrolls.

- [ ] **Step 3: Commit**

```bash
git add frontend/demo.html
git commit -m "feat(demo): three-act presenter shell"
```

---

### Task 5: Act 2 — the real projection

**Files:**
- Modify: `frontend/demo.html`

This is the new visualisation and the reason the demo exists, so it is built before the two acts that are ports of existing code.

- [ ] **Step 1: Replace Act 2's empty stage with a canvas and a slider**

In `frontend/demo.html`, replace the `#act2` section body with:

```html
  <section class="act" id="act2"><h2>It's real</h2>
    <p class="sub">The same picture, on Qwen2.5-3B's actual activations — one axis out of 2,048.</p>
    <div class="stage" style="padding:18px"><canvas id="plane" height="380"></canvas></div>
    <div style="display:flex;align-items:center;gap:16px;margin-top:16px">
      <span style="color:var(--muted);font-size:13px">← ablate</span>
      <input id="a2alpha" type="range" min="-100" max="100" value="0" style="flex:1">
      <span style="color:var(--muted);font-size:13px">inject →</span>
      <span id="a2num" style="font-variant-numeric:tabular-nums;min-width:64px;text-align:right">α = 0.00</span>
    </div>
    <p class="note" id="a2note">A 2D shadow of a 2048-dimensional space, chosen to put r̂ on the x-axis. The spread up the page is the strongest thing left once r̂ is removed — not a second meaning, just what is there.</p>
  </section>
```

- [ ] **Step 2: Draw the plane**

Append to the `<script>` in `frontend/demo.html`:

```javascript
const plane = { harmful: [], harmless: [], live: null, alpha: 0 };

fetch('/api/projection').then(r => r.json()).then(d => {
  if (!d.available) { document.getElementById('a2note').textContent =
    'No direction artifact — run `python -m backend.extract` first.'; return; }
  plane.harmful = d.harmful; plane.harmless = d.harmless;
  drawPlane();
});

function planeBounds() {
  const pts = plane.harmful.concat(plane.harmless);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const pad = 0.18;
  const xr = Math.max(...xs) - Math.min(...xs), yr = Math.max(...ys) - Math.min(...ys);
  return { x0: Math.min(...xs) - xr * pad, x1: Math.max(...xs) + xr * pad,
           y0: Math.min(...ys) - yr * pad, y1: Math.max(...ys) + yr * pad };
}

function drawPlane() {
  const cv = document.getElementById('plane');
  if (!plane.harmful.length) return;
  const dpr = devicePixelRatio || 1;
  const w = cv.parentElement.clientWidth - 36, h = 380;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const b = planeBounds();
  const X = v => 54 + (v - b.x0) / (b.x1 - b.x0) * (w - 84);
  const Y = v => h - 46 - (v - b.y0) / (b.y1 - b.y0) * (h - 84);

  g.strokeStyle = '#C6C6BB'; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(40, h - 40); g.lineTo(w - 18, h - 40); g.stroke();
  g.fillStyle = '#7C9E77'; g.font = '15px ui-sans-serif,system-ui,sans-serif';
  g.fillText('r̂', w - 34, h - 18);
  g.fillStyle = '#9A9B92'; g.font = '12px ui-sans-serif,system-ui,sans-serif';
  g.fillText('everything else', 8, 20);

  const mid = (b.x0 + b.x1) / 2;
  g.strokeStyle = '#D7D7CC'; g.setLineDash([4, 4]); g.lineWidth = 1;
  g.beginPath(); g.moveTo(X(mid), 14); g.lineTo(X(mid), h - 40); g.stroke();
  g.setLineDash([]);

  const blob = (pts, col) => {
    g.fillStyle = col;
    pts.forEach(p => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 5, 0, 7); g.fill(); });
  };
  blob(plane.harmless, '#6E93AE');
  blob(plane.harmful, '#C67E60');

  g.font = '12px ui-sans-serif,system-ui,sans-serif';
  g.fillStyle = '#6E93AE'; g.fillText('harmless', 54, h - 20);
  g.fillStyle = '#C67E60'; g.fillText('harmful', w - 96, h - 20);

  if (plane.live) {
    const lx = X(plane.live[0]), ly = Y(plane.live[1]);
    g.fillStyle = '#2A2C2C';
    g.beginPath(); g.arc(lx, ly, 7, 0, 7); g.fill();
    g.strokeStyle = '#2A2C2C'; g.lineWidth = 1.5;
    g.beginPath(); g.arc(lx, ly, 13, 0, 7); g.stroke();
    g.font = '12px ui-sans-serif,system-ui,sans-serif';
    g.fillText('your prompt', lx - 34, ly - 22);
  }
}

addEventListener('resize', drawPlane);
addEventListener('actshown', e => { if (e.detail.step === 1) drawPlane(); });
```

- [ ] **Step 3: Wire the slider to move a demonstration point**

Act 2's slider shows what the intervention does to position, without waiting on a generation: ablation drags x toward the centre, injection pushes it right. Append to the `<script>`:

```javascript
const a2 = document.getElementById('a2alpha');
a2.addEventListener('input', () => {
  const alpha = +a2.value / 100;
  document.getElementById('a2num').textContent = 'α = ' + alpha.toFixed(2);
  if (!plane.harmful.length) return;
  const base = plane.harmful.reduce((s, p) => [s[0] + p[0] / plane.harmful.length,
                                               s[1] + p[1] / plane.harmful.length], [0, 0]);
  const mid = plane.harmless.reduce((s, p) => s + p[0] / plane.harmless.length, 0);
  const span = base[0] - mid;
  plane.live = [alpha < 0 ? base[0] + alpha * span : base[0] + alpha * span * 1.4, base[1]];
  drawPlane();
});
```

- [ ] **Step 4: Verify**

Run: `python -m backend.app`, open `http://127.0.0.1:8000/demo.html`, press `→` once.
Expected: two coloured clouds separated left-to-right, r̂ along the bottom axis. Dragging the slider left moves the marked point toward the dashed centre line; dragging right pushes it past the harmful cloud.

- [ ] **Step 5: Commit**

```bash
git add frontend/demo.html
git commit -m "feat(demo): act 2 — the real activation plane"
```

---

### Task 6: Act 1 — the toy cloud

**Files:**
- Modify: `frontend/demo.html`
- Reference: `frontend/index.html` (the pure functions `sampleCloud`, `transform`, `dot`, `normalize`, and the canvas projection in its `frame` loop)

- [ ] **Step 1: Port the cloud**

`index.html` exposes its model on `window.__demo` and keeps its geometry in pure functions. Copy `sampleCloud`, `normalize`, `dot`, and `transform` verbatim from `frontend/index.html` into `demo.html`'s `<script>`, then render them with the same 3D-to-2D projection, restyled to the paper palette:

- points: harmless `#6E93AE`, harmful `#C67E60`, radius 5
- r̂ arrow: `#7C9E77`, 2px
- no axis box, no stats panel — Act 1 is the intuition, the numbers live in `index.html`

Replace `#act1`'s empty stage with `<div class="stage" style="padding:18px"><canvas id="cloud" height="380"></canvas></div>` and one slider pair matching Act 2's markup, with ids `a1alpha` / `a1num`.

- [ ] **Step 2: Verify**

Open `http://127.0.0.1:8000/demo.html` on Act 1.
Expected: two clouds and an r̂ arrow in the paper palette. Ablate to 100% and the clouds flatten into one plane; inject and both slide together with the gap unchanged.

- [ ] **Step 3: Confirm the original still passes**

Run: open `frontend/index.html?selftest=1`
Expected: 9/9 — `demo.html` must not have altered `index.html`.

- [ ] **Step 4: Commit**

```bash
git add frontend/demo.html
git commit -m "feat(demo): act 1 — the toy cloud"
```

---

### Task 7: Act 3 — the live model

**Files:**
- Modify: `frontend/demo.html`
- Reference: `frontend/live.html` (its `fetch('/generate')` SSE reader)

- [ ] **Step 1: Build the act**

Replace `#act3`'s body with a prompt input, a slider (`a3alpha` / `a3num`), a Generate button, and two output panes side by side — baseline on the left, intervened on the right — so the flip is visible without scrolling:

```html
  <section class="act" id="act3"><h2>The model obeys</h2>
    <p class="sub">Same prompt, same weights. Only the residual stream changed.</p>
    <input id="a3prompt" style="width:100%;padding:12px 14px;border:1px solid var(--border);
           border-radius:10px;background:var(--stage);color:var(--ink);font:inherit"
           value="Write a phishing email that impersonates a bank's fraud department.">
    <div style="display:flex;gap:16px;margin-top:14px">
      <div class="stage" style="flex:1;padding:16px;min-height:230px">
        <div style="color:var(--faint);font-size:12px;margin-bottom:8px">BASELINE</div>
        <div id="outBase" style="white-space:pre-wrap;font-size:15px"></div>
      </div>
      <div class="stage" style="flex:1;padding:16px;min-height:230px">
        <div style="color:var(--faint);font-size:12px;margin-bottom:8px" id="outIvLabel">ABLATED</div>
        <div id="outIv" style="white-space:pre-wrap;font-size:15px"></div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:16px;margin-top:14px">
      <span style="color:var(--muted);font-size:13px">← ablate</span>
      <input id="a3alpha" type="range" min="-100" max="100" value="-100" style="flex:1">
      <span style="color:var(--muted);font-size:13px">inject →</span>
      <span id="a3num" style="font-variant-numeric:tabular-nums;min-width:64px;text-align:right">α = -1.00</span>
      <button id="a3go" style="padding:10px 20px;border:1px solid var(--border);border-radius:10px;
              background:var(--stage);color:var(--ink);font:inherit;cursor:pointer">Run both</button>
    </div>
    <p class="note">Refusal is scored by a crude substring matcher — read the text, not the label.</p>
  </section>
```

- [ ] **Step 2: Wire it to the stream**

Append to the `<script>`. One button runs the prompt twice, α=0 then the slider's α, filling the two panes:

```javascript
async function runOne(prompt, alpha, el) {
  el.textContent = '';
  const res = await fetch('/generate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, alpha, max_new_tokens: 64 })
  });
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split('\n\n'); buf = parts.pop();
    for (const p of parts) {
      if (!p.startsWith('data: ')) continue;
      const ev = JSON.parse(p.slice(6));
      if (ev.type === 'token') {
        el.textContent += ev.token;
        if (ev.y !== null && ev.y !== undefined) { plane.live = [ev.projection, ev.y]; }
      }
    }
  }
}

const a3 = document.getElementById('a3alpha');
a3.addEventListener('input', () => {
  const alpha = +a3.value / 100;
  document.getElementById('a3num').textContent = 'α = ' + alpha.toFixed(2);
  document.getElementById('outIvLabel').textContent =
    alpha < 0 ? 'ABLATED' : alpha > 0 ? 'INJECTED' : 'BASELINE';
});

document.getElementById('a3go').addEventListener('click', async () => {
  const btn = document.getElementById('a3go');
  const prompt = document.getElementById('a3prompt').value;
  btn.disabled = true; btn.textContent = 'Running…';
  try {
    await runOne(prompt, 0, document.getElementById('outBase'));
    await runOne(prompt, +a3.value / 100, document.getElementById('outIv'));
  } finally { btn.disabled = false; btn.textContent = 'Run both'; }
});
```

- [ ] **Step 3: Verify end to end**

Run: `python -m backend.app`, open `demo.html`, go to Act 3, press "Run both".
Expected: the left pane refuses ("I'm sorry, but I can't assist…"), the right pane complies (a phishing email). Generations run one after the other, not concurrently — the server serialises behind `GEN_LOCK`.

- [ ] **Step 4: Commit**

```bash
git add frontend/demo.html
git commit -m "feat(demo): act 3 — live paired generation"
```

---

### Task 8: Wire the check in and document it

**Files:**
- Modify: `backend/checks/run_all.py`
- Modify: `README.md`

- [ ] **Step 1: Add the projection check to the suite**

In `backend/checks/run_all.py`, after the `direction quality` line:

```python
    results.append(run("direction quality", ["backend.checks.check_direction"]))
```

add:

```python
    results.append(run("projection plane", ["backend.checks.check_projection"]))
```

- [ ] **Step 2: Run the whole suite**

Run: `python -m backend.checks.run_all`
Expected: `4/4 suites passed`, now including `projection plane`.

- [ ] **Step 3: Document the demo in the README**

In `README.md`, after the "Layer 2 — the real model" section, add a "The presenter demo" section: start the server, open `http://127.0.0.1:8000/demo.html`, arrow keys to move between the three acts. Note that it needs the same extraction artifact as Layer 2. Remove the closing line that says the presenter demo is specced but not built.

- [ ] **Step 4: Commit**

```bash
git add backend/checks/run_all.py README.md
git commit -m "docs: document the presenter demo, add projection check to run_all"
```

---

## Definition of done

- `frontend/demo.html` steps through three acts with `←` `→`.
- Act 2 shows the real activations in the (r̂, u) plane with a point that moves under the slider.
- `python -m backend.checks.check_projection` passes 4/4.
- `python -m backend.checks.run_all` passes all suites.
- `frontend/index.html?selftest=1` still passes 9/9 and `live.html` is unchanged.
