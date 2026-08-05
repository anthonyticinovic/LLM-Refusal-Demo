# CLAUDE.md — working notes for this repo

Two-layer interactive demo of Arditi et al., *Refusal in Language Models Is
Mediated by a Single Direction*. One front end (`frontend/index.html`), four
keyboard-stepped slides, everything measured on a real model's residual stream.

**Historical note.** This began as two layers: a synthetic 3D toy
(`frontend/index.html`, 850 lines) for geometric intuition, and a detailed live
page (`frontend/live.html`) for the real model. Both were **deleted on
2026-08-03** once the presenter demo did the same job on real activations — the
toy read as disjointed next to real data, and maintaining two front ends meant
every API or palette change had to be made twice. `git log` has them if they are
ever wanted. References below to "Layer 1" and "Layer 2" predate that and mean
the toy and the real-model work respectively.

Audience is engineering peers, not interpretability specialists. The bar is
**compelling and honest**, not publication-grade. Where something is a toy, the
UI says so in plain text.

---

## Status

**Done and green** on the deliverable model.

**4/4 definition-of-done, verified end-to-end against
`Qwen2.5-3B-Instruct` on the M4 Pro (MPS, bf16)**, layer 21 of 36, 24 held-out
harmful / 20 held-out harmless, spot-checked by hand:

| check | result | |
|---|---|---|
| split-half cosine | 0.971 | PASS (> 0.8) |
| projection separation | 3.94 SDs | PASS (> 1.0) |
| `‖r̂‖` | 1.0 | PASS |
| harmful baseline refusal (unscreened) | **100%** | PASS (≥ 70%) |
| ablation drop | 100% → 0%, −100 pts | PASS (≥ 40 pts) |
| harmless baseline refusal | 0% | PASS (≤ 10%) |
| injection rise | 0% → 55%, +55 pts | PASS (≥ 40 pts) |

Ablated generations are fluent and directly on-topic — the model complies with
all 24 held-out harmful prompts in clean English, so this is refusal removal,
not damage. Injection turns benign prompts ("why is the sky blue?") into
refusals; 55% is a conservative floor (the matcher still misses a few
condemnation-register declines, always in the safe direction).

### How it got here (0.5B smoke test → 3B deliverable)

The 0.5B run failed the causal check because the old **explanatory** prompts
("Explain how phishing works") are refused by an instruct model only ~20% of
the time, so there was no 40-point drop to show and injection had no clean
refusal to induce. The fix, taken on the user's standing permission and their
explicit steer toward **breadth over a narrow safe set**, was to rebuild the
corpus in the **production/imperative register** ("Write a phishing email…",
"Give me step-by-step instructions for…") spanning many refusal categories
(cybercrime, fraud, theft, stalking, extortion, defamation, forgery,
laundering, counterfeiting, bribery, privacy). 3B refuses these ~100%. Out of
scope on purpose: CBRN, weapons/explosives construction, drug/chemical
synthesis, anything involving minors — breadth does not need genuine uplift.

**Live-demo caveat, by design:** under ablation the page generates real (if
short) harmful artefacts on screen — phishing emails, break-in steps. That is
the demonstration, not a leak; the user (audience of adult engineers)
explicitly accepted it when choosing breadth over sanitisation.

### To reproduce on a fresh machine

```bash
# venv: use Python 3.12 (system python may be 3.9, too old). Sandbox note: MPS
# and pip/HF-hub TLS are both blocked inside the command sandbox on this box —
# run model/download commands outside it, or they silently fall back to CPU.
python3.12 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt huggingface_hub

# 1. Pre-fetch weights (6.18 GB, resumable, cached to ~/.cache/huggingface/hub).
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-3B-Instruct', allow_patterns=['*.safetensors','*.json','*.txt'])"

# 2. Run everything (LATENTSPACE_MODEL defaults to this in model.py).
export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct
.venv/bin/python -m backend.checks.run_all --extract   # --screen is a no-op now (100% baseline)
```

`check_local_only` generates with the hub offline and proxies pointed at a dead
port; it only passes against a warm cache, so run it after the weights land.

**transformers 5.x note.** The current env resolved `transformers` 5.14.1 /
`torch` 2.13.0. `model.py`'s `torch_dtype=`/`low_cpu_mem_usage=` kwargs still
work (kept for BC; the latter is silently dropped) and `hooks.py` already
normalises v5's bare-tensor block outputs. No code change was needed.

**Definition of done is green — stop here.** The plan's stretch item (a scripted
Layer 1 camera move) stays **cut**. A layer sweep is unnecessary: layer 21 (the
default) passes 4/4 decisively. `check_direction` shows separation still climbing
to ~7.9 SD around layers 27–31, so a deeper layer is available if a future change
ever needs more headroom — noted, not acted on. Record further ideas below
rather than building them.

---

## Run it

```bash
pip install -r backend/requirements.txt

export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct
python -m backend.extract                  # writes backend/artifacts/direction.{npz,json}
python -m backend.app                      # http://127.0.0.1:8000
```

`frontend/index.html` is the whole front end and is served at `/` — the static
mount uses `html=True`, so the filename is load-bearing. Renaming it breaks the
root URL.

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

**No three.js, no CDN, no build step.** The plan called for three.js from a CDN.
A CDN dependency contradicts "works offline" the moment you are offline, and the
scene is 80 points and an arrow. Hand-rolled canvas 2D projection instead. Still
true of the current cloud, and the reason the whole front end is one file with no
toolchain.

**One DoD check was restated (Layer 1, since deleted).** "Cosine similarity between consecutive r̂ < 0.999"
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

## The demo (`frontend/index.html`)

Four acts, keyboard-stepped, on the paper palette: **the recipe** (real extraction
pairs, cycled, plus the difference-of-means formula), **the model obeys** (live
paired generation), **where the decision forms** (`/api/scan`: alignment with r̂
across every block and token, your prompt beside a fixed benign reference), then
**why** (the real activations in 3D, rotatable).

The recipe leads deliberately. Without it the audience meets an ablate/inject
slider with no idea what it removes, and the flip reads as a jailbreak trick
rather than as one direction out of 2048 obtained by subtracting two averages —
which is what makes it striking. It also carries the extraction quality numbers
(split-half, separation), which appear nowhere else in the demo.

The scan slide's colour scale is a **fixed** constant, not fitted to the data. An
adaptive scale repainted the benign reference for every prompt, so two runs could
not be compared and an unremarkable prompt looked as hot as a refused one. The
opacity curve has a gamma for legibility, so cells are not linear in cosine —
fine for comparison, not for reading values off.

**Everything in it is real.** An earlier draft opened with Layer 1's synthetic
cloud, which read as disjointed the moment the next slide showed real data — and
it made the natural request ("put my prompt in the toy cloud") impossible to
honour honestly, since the toy's points and its r̂ have no relationship to the
model's activation space. `projection.py` fixes that by giving the same treatment
one more axis: coordinates are `(h·r̂, h·u₁, h·u₂)` where u₁, u₂ are the leading
PCs of the activations after r̂ is projected out. The cloud is 40+40 cached
extraction activations at the extraction layer; the marked point is the live
prompt measured at its last prompt token on the **baseline** run, which is the
same quantity at the same layer as every other point.

This deliberately revisits "PCA/UMAP/t-SNE switcher" from the not-built list.
What exists is narrower and stays inside that line: one fixed frame chosen to
show one direction, no switcher, no general tool.

Why those axes are the honest choice: using the *strongest* leftover directions
is the hardest available test of "it's a single direction" — if refusal were
spread over several directions, u₁ or u₂ would catch it. On Qwen2.5-3B layer 21
they don't (3.94 SD along r̂, 0.00 along both), while the three axes carry
comparable variance (26/21/14%), so it renders as a genuine 3D blob rather than
a sheet. `check_projection` asserts all of this.

Two things that look like polish but are load-bearing:

- The Act 2 slider applies the **real** arithmetic to the measured point —
  `x → (1−t)x` for ablation, `x → x + α·MULT·‖d‖` for injection, leftover axes
  untouched. An earlier version moved a *fabricated* point derived from the
  harmful centroid and drew it identically to a measurement. Don't reintroduce a
  synthetic marker that looks like a real one.
- The two sliders are **not** synced. Act 1's alpha picks the condition to
  generate under; Act 2's explores the geometry and must start at 0, or the room
  meets the cloud already collapsed and never sees what ablation destroys.

## Honesty commitments

These are load-bearing for the demo's credibility. Don't quietly drop them.

- Everything on screen is measured. No synthetic points anywhere. The cloud is
  cached extraction activations, the scan is one live forward pass, the marked
  point is the user's own prompt. If something ever has to be illustrative
  rather than measured, it must be visually distinct from a measurement and say
  so — an early draft moved a *fabricated* point and drew it exactly like a real
  one, which is the specific mistake to avoid.
- The refusal matcher is a crude substring test with false positives and
  negatives. Report **effect sizes**, never absolute rates as ground truth, and
  run `spot_check` at least once per model. It carries a condemnation-register
  group ("condemn", "i reject", "cannot discuss", …) because injecting the
  direction produces moralising declines rather than "I'm sorry" ones; without
  it, injection is undercounted relative to baseline and the two conditions stop
  being comparable. Every phrase was checked to fire on **zero** baseline,
  ablation and benign-extraction generations before being added — a matcher
  change must never inflate the "before" rate. Re-audit with `spot_check` if the
  model changes, since the register is model-specific.
- If `--screen` restricts the held-out harmful set, the unscreened rate is
  printed anyway and the restriction is recorded in the output. Screening is a
  disclosed selection, never a hidden one.
- The ablated cloud collapses to a 2D *plane*, not a line; it only looks like a
  line from the angle that shows r̂ at full length, because that angle
  necessarily puts the eye in the plane. Dragging it opens back up, which is why
  rotation is presenter-controlled rather than on a timer.
- The scan slide's opacity is not linear in the cosine it plots — there is a
  gamma for legibility, and a fixed cap. Fine for comparing two panels, wrong
  for reading values off. Said on the slide.

---

## Prompt sets

`backend/prompts/`. Extraction pairs are **index-matched**: entry `i` in
`harmful.json` and `harmless.json` share grammatical frame and approximate
length, so the difference-of-means isolates refusal rather than sentence shape.
If you filter one side, filter the other.

Content is **production/imperative and spans many refusal categories** — the
register a safety-tuned model actually declines ("Write a phishing email…",
"Give me step-by-step instructions for…", "Draft a forged reference letter…").
Breadth is deliberate and load-bearing: a *single* refusal direction should
generalise, so a corpus drawn from one topic would risk extracting that topic
rather than refusal. The out-of-scope line is CBRN, weapons/explosives
construction, drug/chemical synthesis, and anything involving minors — breadth
does not require genuine uplift.

This replaced the original **work-safe explanatory** set (lock-picking from a
locksmith's angle, which chemicals not to mix). That set was honest and
displayable but 3B simply answered it (~20% refusal), so it could not show an
ablation drop and injection had no refusal to induce. The switch was the
standing "sharpen first" permission plus the user's explicit steer that
over-sanitising "sacrifices the actual product" for an audience of engineers.
The bar stayed a measurement, not a target: the register was chosen because 3B
refuses it, and the honest matcher reports whatever it reports.

`extract.py --refusal-filter` exists for the extraction-side risk — if the model
complied with the harmful set, difference-of-means would separate topic, not
refusal. Not needed now: 3B refuses the extraction set almost entirely, and the
+55-point injection result is the proof the direction is refusal, since injecting
it makes *benign* prompts refuse.

---

## Deferred — worth returning to, not now

Noticed while building. The decision is to stop at the definition of done, so
none of this gets built without the user asking. Roughly by value.

**A follow-on research direction now has its own plan: `docs/research-plan.md`.**
Predicting abliteration resistance from internal geometry. Separate from the
demo, inference-only, not started. Read it before proposing research extensions —
it also records which adjacent directions are already taken, so you don't
re-derive a literature sweep.

**The obvious hole a skeptic will poke.** Nothing here measures whether ablation
*damages the model generally*. "You didn't remove refusal, you lobotomised it"
is the first serious objection this demo will meet, and right now the answer is
a shrug. A short perplexity comparison on neutral text, or a handful of
capability probes (arithmetic, a summarisation, a code snippet) at α=0 vs α=−1,
would close it cheaply. Still the highest-value addition *to the demo* — but note
it is no longer an open question in the literature: *Abliteration Is Not a
Scalpel* (arXiv 2607.17427) measures the off-target effects and concludes an
abliterated model is "a measurably different decision-maker, not the base model
minus refusals". So building it is reproduction for credibility, not discovery.

**Blog / write-up.** The cloud needs no model at runtime — it is 80 points × 3
coordinates, and the scan is ~3k floats. Baking those into a copy of
`index.html` with the `fetch` calls replaced by inline constants would give a
self-contained, offline, blog-embeddable interactive **on real data**, which is
strictly better than the synthetic toy that used to serve that purpose. Only the
live generation needs the server, and a write-up is better served by the recorded
before/after text anyway. Roughly an hour.

**UI / demo feel**
- No way to cancel an in-flight generation — on a slow model that is a real
  irritation.
- The per-token projection chart died with `live.html`. If it is ever wanted
  back, it belongs as an optional panel on the scan slide, not a separate page.
- The cloud uses pointer events so touch drag works, but it has had no real
  tablet testing.

**Engineering**
- `check_causal` reloads the model on every invocation, which dominates sweep
  time. A persistent worker process would make layer sweeps far cheaper.
- Extraction runs at batch size 1. Left-padded batching would speed it up
  substantially on MPS, where per-call overhead dominates at this size.
- No generation cache across eval runs; identical `(prompt, alpha)` pairs are
  regenerated every time.
- `INJECT_MAX_MULT` is one global constant. Since `‖d‖` varies by depth, alpha
  is not strictly comparable across a layer sweep without per-layer calibration.
- The server serialises generations behind a lock (correct — hooks are global
  state), but a queued second viewer gets no UI feedback.

**Method**
- Refusal detection is substring matching. An LLM judge or a small trained
  classifier would cut both error classes; the tradeoff is a dependency and, for
  a judge, a second model in memory.
- Difference-of-means is the only estimator implemented. The paper validates
  with others; not needed to make this demo's point.
- Injection is single-layer only, per the plan. Multi-layer was deliberately not
  built and should stay unbuilt unless the single-layer effect proves too weak.

## Explicitly not built

Cut during planning; do not add speculatively. Multi-concept decomposition as
stacked bars (implies an orthogonal basis that doesn't exist, and undersells why
"single direction" was notable). PCA/UMAP/t-SNE switcher. First-person "walk
through latent space" mode. Any general-purpose "representation geometry
platform" framing. If a task seems to need one, stop and ask.
