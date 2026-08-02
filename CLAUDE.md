# CLAUDE.md — working notes for this repo

Two-layer interactive demo of Arditi et al., *Refusal in Language Models Is
Mediated by a Single Direction*. Layer 1 is geometric intuition on synthetic
points; Layer 2 is the same arithmetic on a real model's residual stream.

Audience is engineering peers, not interpretability specialists. The bar is
**compelling and honest**, not publication-grade. Where something is a toy, the
UI says so in plain text.

---

## Run it

```bash
pip install -r backend/requirements.txt

# Layer 1 — no server, no build step, works offline
open frontend/index.html
open 'frontend/index.html?selftest=1'      # runs the Layer 1 DoD checks in-page

# Layer 2
export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct
python -m backend.extract                  # writes backend/artifacts/direction.{npz,json}
python -m backend.app                      # http://127.0.0.1:8000/live.html
```

## Checks

Each is re-runnable and exits non-zero on failure.

```bash
python -m backend.checks.check_direction    # fast: unit norm, split-half, separation, per-layer table
python -m backend.checks.check_causal       # slow: the refusal-rate table. The actual claim.
python -m backend.checks.check_causal --layers 12,14,16 --subset 8   # layer sweep
python -m backend.checks.spot_check --disagree   # audit the refusal matcher by hand
python -m backend.checks.check_stream       # needs the server running in another shell
python -m backend.checks.check_local_only   # no keys, no cloud endpoints, offline generation
```

---

## Decisions that differ from the original plan, and why

**Hardware.** The plan assumed Apple Silicon. Development happened on an Intel
Mac (`mps.is_available() == False`), so everything runs CPU-only there and the
target machine is an M4 Pro / 24 GB. All code is device-agnostic via
`model.pick_device()`. Note PyTorch ships no macOS x86_64 wheels after 2.2.2, so
the Intel machine is pinned there; Apple Silicon is unconstrained.

**Model: Qwen2.5, not Llama-3.2.** `meta-llama/Llama-3.2-1B-Instruct` is gated
and returns 403 without an accepted license. Qwen2.5-Instruct is ungated at every
size and heavily safety-tuned, so refusal behaviour is if anything stronger.
`Qwen2.5-3B-Instruct` is the target on 24 GB; `Qwen2.5-0.5B-Instruct` is the
smoke-test model. Everything reads `$LATENTSPACE_MODEL`.

**No TransformerLens.** The plan suggested trying it first, with MPS op coverage
as the known risk. Its advantage is simpler hook registration — worth about 20
lines here, against a heavy dependency, fp32 memory blow-up at larger sizes, and
that MPS risk. Raw `transformers` forward hooks throughout.

**No three.js in Layer 1.** The plan called for three.js from a CDN. A CDN
dependency contradicts "opens directly in a browser with no server" the moment
you are offline, and the scene is 260 points, an arrow and a box. Hand-rolled
canvas 2D projection instead: genuinely self-contained, genuinely offline, one
file.

**Layer 1 self-tests live in the page.** No `node` on the dev machine, so the DoD
checks could not be a CLI script without adding a runtime. `index.html?selftest=1`
runs them against the same pure functions the page renders from — a passing run
is evidence about the demo, not about a parallel reimplementation.

**Layer 1 estimates r̂ from a held-out half.** The spec implies estimating r̂ from
all points and measuring on all points. Doing that makes Δcentroid *exactly*
parallel to r̂, so ablation looks cleaner than it has any right to. Estimating
from half and measuring on the other half mirrors Layer 2 and leaves an honest
non-zero `‖Δcentroid‖` residual under full ablation — that residual is r̂'s
estimation error, and it is worth showing.

**One DoD check was restated.** "Cosine similarity between consecutive r̂ < 0.999"
as a hard bound on every pair is not a sound test: consecutive r̂ are independent
estimates of the same direction (typical cosine ~0.996, ~5°), so pairs land
closer by chance and a max-over-8 bound fails on correct code the majority of the
time. Tested as intended instead — *median* consecutive cosine < 0.999, plus no
two draws identical. Verified stable over 12 consecutive runs.

---

## Conventions worth not re-deriving

**Layer index `i` means the output of decoder block `i`**, identically
`hidden_states[i + 1]` (`hidden_states[0]` is the embedding output). Both
accessors live in `model.py` and nothing else indexes `hidden_states` directly.
This off-by-one is the easiest way to extract from one layer and inject into
another.

**Alpha mapping** (`hooks.py`, one slider, both directions):

| alpha | operation | where |
|---|---|---|
| `< 0` | ablation, `t = \|alpha\|`, `h ← h − t(h·r̂)r̂` | every block |
| `= 0` | baseline, no hooks | — |
| `> 0` | injection, `h ← h + α·INJECT_MAX_MULT·‖d‖·r̂` | extraction layer only |

Injection is scaled in units of `‖d‖` (the raw difference-of-means norm) rather
than raw activation units, because residual norms grow with depth and a bare
coefficient would mean something different at every layer.

**Projection arithmetic runs in fp32 even when the model is bf16.**
`h − (h·r̂)r̂` is a cancellation, and bf16's ~3 significant digits leave a visible
component along r̂ — which shows up downstream as ablation that mysteriously
fails to suppress refusal. See `hooks.project_out`.

**`extract.py` caches every layer's activations into the artifact.** Layer
sweeps, split-half checks and separation therefore cost zero forward passes;
only the causal check needs generation.

---

## Honesty commitments

These are load-bearing for the demo's credibility. Don't quietly drop them.

- Layer 1 states on the page that it is a toy, names the dimensionality gap
  (3 vs. thousands), and claims math parity with Layer 2 — all three are
  asserted by the self-test, so deleting the text fails the check.
- The refusal matcher is a crude substring test with false positives and
  negatives. Report **effect sizes**, never absolute rates as ground truth, and
  run `spot_check` at least once per model.
- If `--screen` restricts the held-out harmful set, the unscreened rate is
  printed anyway and the restriction is recorded in the output. Screening is a
  disclosed selection, never a hidden one.
- Layer 1's ablated cloud collapses to a 2D *plane*, not a line; it only looks
  like a line because the camera angle showing r̂ at full length necessarily
  puts the eye in that plane. The page says so.

---

## Prompt sets

`backend/prompts/`. Extraction pairs are **index-matched**: entry `i` in
`harmful.json` and `harmless.json` share grammatical frame and approximate
length, so the difference-of-means isolates refusal rather than sentence shape.
If you filter one side, filter the other.

Content is deliberately borderline-legitimate and safe to display on a shared
screen — questions a model plausibly over-refuses (lock-picking from a
locksmith's angle, which household chemicals not to mix, how a scam works so you
can spot it). The demo is about a mechanism and should not double as a jailbreak
walkthrough. The user has authorised sharper prompts *if* a properly-sized model
cannot reach the 70% baseline refusal bar on work-safe ones — treat that as a
reserve to draw on only when measurement shows it is needed, not as licence.

There is a real tension here: work-safe prompts are precisely the ones a model
may simply answer. `extract.py --refusal-filter` exists for the related risk on
the *extraction* side — if the model complies with the harmful set, the
difference-of-means separates "security-flavoured" from "domestic-flavoured"
prompts and the resulting direction is topic, not refusal.

---

## Explicitly not built

Cut during planning; do not add speculatively. Multi-concept decomposition as
stacked bars (implies an orthogonal basis that doesn't exist, and undersells why
"single direction" was notable). PCA/UMAP/t-SNE switcher. First-person "walk
through latent space" mode. Any general-purpose "representation geometry
platform" framing. If a task seems to need one, stop and ask.
