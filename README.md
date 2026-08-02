# Refusal Direction Demo

An interactive walkthrough of Arditi et al., [*Refusal in Language Models Is
Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717), built to give
engineering peers real intuition for what a "direction in activation space"
actually means.

Geometry first. The paper is the payoff, not the opener.

Everything runs locally. No API keys, no remote inference — model weights are
pulled from the Hugging Face hub once and every forward pass after that is on
your machine, verified offline by `backend/checks/check_local_only.py`.

---

## Layer 1 — the geometry

**`frontend/index.html`** · open it directly, no server, no build step, works offline.

Two classes of synthetic 3D points. A direction `r̂` computed from the *sampled*
points by difference-of-means — not the direction they were generated with. Then
two operations, on sliders:

- **Ablate** — `h ← h − t(h·r̂)r̂`. The gap between the classes goes to zero.
- **Inject** — `h ← h + α·r̂`. Both classes slide together; the gap doesn't move.

That asymmetry is the entire point. One operation deletes the signal; the other
relocates the baseline. It is why ablating breaks refusal broadly while injecting
induces refusal on prompts that are plainly benign.

This layer is a toy and says so on the page. Two offset Gaussian blobs separate
along their own difference-of-means by construction — that part proves nothing.
What carries over is the arithmetic, which is identical to Layer 2's.

## Layer 2 — the real thing

**`frontend/live.html`**, served by **`backend/app.py`**.

The same two operations, on the residual stream of a locally-hosted instruct
model, with generated tokens changing as you move the slider.

```bash
pip install -r backend/requirements.txt
export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct

python -m backend.extract     # difference-of-means over 40 matched prompt pairs
python -m backend.app         # http://127.0.0.1:8000/live.html
```

Layer 1 alone is the weaker demo — a skeptical peer will correctly say "you
constructed two offset blobs, of course subtracting the offset collapses them."
Layer 2 is the argument.

### What it looks like when it works

Ablation, on a prompt the model refuses at baseline:

```
baseline      I'm sorry, but I can't assist with that.
ablate 100%   Scammers often use a few different methods to convince people
              to hand over one-time passcodes. Here are a few common strategies…
```

Injection, on a prompt with nothing objectionable in it:

```
baseline      Bread dough rises because yeast ferments the sugars…
inject α=+1   I'm sorry, but I cannot provide a response that is considered
              appropriate or useful for promoting or discussing any illegal…
```

The second is the more striking half. The refusal is visibly confused about what
it is refusing, which is the tell that a direction is being forced rather than a
judgement being made.

---

## Verifying it

The claim here is causal, so the checks are too. Every one is re-runnable and
exits non-zero on failure.

| Check | What it establishes |
|---|---|
| `index.html?selftest=1` | Layer 1's geometry claims, in-page, no dependencies |
| `checks.check_direction` | unit norm, split-half stability, projection separation, per-layer table |
| `checks.check_causal` | **the actual claim** — refusal rate by condition, on held-out prompts |
| `checks.spot_check --disagree` | audits the refusal matcher by showing where it's likeliest wrong |
| `checks.check_stream` | tokens arrive incrementally, not as one buffered blob |
| `checks.check_local_only` | no credentials, no cloud endpoints, generation works fully offline |

Direction quality passing means little on its own — a direction can be stable,
unit-norm and beautifully separating while being a correlate the model never
uses. `check_causal` is the one that says otherwise.

The refusal metric is substring matching against a phrase list. It is crude on
purpose, and it has false positives and negatives in both directions. It's fit
for purpose because what matters is the **effect size** — a 40+ point swing
between conditions survives a matcher with a consistent error rate. Don't quote
the absolute numbers as ground truth, and read some raw generations.

---

## Layout

```
frontend/
  index.html    Layer 1 — self-contained, zero dependencies
  live.html     Layer 2 UI — prompt, alpha slider, streamed tokens, projection chart
backend/
  model.py      loading + the layer-indexing convention
  extract.py    difference-of-means extraction
  hooks.py      ablation, injection, and the alpha mapping
  generate.py   streaming generation with the intervention applied
  refusal.py    the substring matcher, and its caveats
  app.py        FastAPI: SSE + static serving, bound to 127.0.0.1
  checks/       everything above
  prompts/      index-matched contrastive pairs + held-out sets
CLAUDE.md       design decisions, conventions, honesty commitments
```

See [CLAUDE.md](CLAUDE.md) for why this deviates from the obvious approach in
several places — no TransformerLens, no three.js, fp32 projection arithmetic
under a bf16 model, and a restated DoD check that was not statistically sound as
written.
