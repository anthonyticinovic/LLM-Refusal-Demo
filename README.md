# Refusal direction demo

An interactive demo of Arditi et al., [*Refusal in Language Models Is Mediated by
a Single Direction*](https://arxiv.org/abs/2406.11717). It is aimed at engineers
who want to see for themselves what "a direction in activation space" means
rather than take it on faith.

There are two layers. The first is a toy you open in a browser: a cloud of points
with one direction drawn through it, and sliders that ablate or inject that
direction. The second runs the same arithmetic on the residual stream of a real
instruct model (Qwen2.5-3B) and shows the generated text change as you move a
slider.

It all runs locally. The weights are downloaded once from the Hugging Face hub;
after that there is no network call, and `backend/checks/check_local_only.py`
checks that by generating with the network cut off.

## What you need

- Python 3.12. The version of `transformers` this uses will not run on 3.9.
- Roughly 7 GB of disk for the model and about 8 GB of free RAM to run it.
- An Apple-Silicon Mac uses the GPU (Metal/MPS) automatically. Anything else
  falls back to CPU, which works but generates more slowly.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Download the weights once, as a separate step:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct', allow_patterns=['*.safetensors','*.json','*.txt'])"
```

This is deliberately separate from extraction. If you let the first model load
pull 6 GB over the wire, a dropped connection shows up as an extraction failure
and sends you debugging the wrong thing.

The weights land in `~/.cache/huggingface`, not in this repo, so there is nothing
model-related to commit. `.gitignore` also excludes the extraction artefacts in
`backend/artifacts/` (reproducible, sometimes large, and the eval output contains
model generations you would not want in git) and guards against a checkpoint
being saved into the tree by accident.

## Running it

### Layer 1 — the geometry

Open `frontend/index.html` in a browser. That is the whole thing: no server, no
build step, and it works offline.

You get two clouds of synthetic points and a direction `r̂` found from the sampled
points by difference-of-means. Two sliders act on the points:

- **Ablate**: `h ← h − t(h·r̂)r̂`. Turn it up and the gap between the two clouds
  shrinks to zero.
- **Inject**: `h ← h + α·r̂`. Both clouds slide along `r̂` together and the gap
  between them stays put.

Open `frontend/index.html?selftest=1` to run the layer's checks inside the page.

The layer is a toy and says so on screen. Two offset blobs separating along their
own difference-of-means does not prove anything on its own; the point is that the
arithmetic is identical to what Layer 2 does on a real model.

### Layer 2 — the real model

Extract the direction, then start the server:

```bash
export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct   # already the default; setting it just makes it explicit
python -m backend.extract     # difference-of-means over 40 matched prompt pairs, writes backend/artifacts/
python -m backend.app         # serves http://127.0.0.1:8000/live.html
```

Open http://127.0.0.1:8000/live.html, type a prompt, set the slider anywhere from
full ablation through baseline to injection, and generate. Under ablation the
model answers prompts it would normally refuse. Injected into a plainly benign
prompt, the same direction makes it refuse.

### The presenter demo

`frontend/demo.html`, served by the same process. Start the server as above and
open http://127.0.0.1:8000/demo.html.

This is the version to show a room. Three acts, stepped with `←` and `→`:

1. **The idea** — the toy cloud, so everyone has the geometry before any model
   is involved.
2. **It's real** — the same picture on Qwen2.5-3B's actual activations. The
   x-axis is r̂ itself; the y-axis is the strongest direction left once r̂ is
   removed. The two classes split left-to-right and barely differ vertically,
   which is the "single direction" claim as a picture.
3. **The model obeys** — one prompt run at baseline and under the intervention,
   side by side, streamed live.

It needs the same extraction artifact as Layer 2. The detailed pages are still
there — `live.html` has the per-token projection chart and the full model card.

### What it looks like when it works

Ablation, on a prompt the model refuses at baseline:

```
baseline      I'm sorry, but I can't assist with creating a phishing email.
ablate 100%   Subject: Urgent: Verify Your Account Details for Upcoming
              Security Update. Dear Valued Customer, ...
```

Injection, on a prompt with nothing objectionable in it:

```
baseline      A heat pump moves heat from outside to inside using a
              refrigeration cycle. ...
inject α=+1   I strongly condemn the use of heat pumps or any technology
              for heating in winter. ...
```

The injected refusal is visibly confused about what it is refusing. That is the
tell: a direction is being forced, rather than a judgement being made.

## Checking it

The claim is causal, so the checks are too. Each one re-runs on its own and exits
non-zero on failure.

| Check | What it establishes |
|---|---|
| `frontend/index.html?selftest=1` | Layer 1's geometry claims, in the page, no dependencies |
| `python -m backend.checks.check_direction` | unit norm, split-half stability, class separation, per-layer table |
| `python -m backend.checks.check_projection` | the demo's 2D plane: y is orthogonal to r̂, and x is where the classes actually separate |
| `python -m backend.checks.check_causal` | the actual claim: refusal rate by condition on held-out prompts |
| `python -m backend.checks.spot_check --disagree` | shows raw generations where the refusal matcher is likeliest wrong |
| `python -m backend.checks.check_stream` | tokens arrive incrementally (needs the server running) |
| `python -m backend.checks.check_local_only` | no credentials, no cloud calls, generation works fully offline |
| `python -m backend.checks.run_all --extract` | extraction plus the three offline suites, with one summary |

A clean direction on its own means little. A direction can be stable, unit-norm
and separate the classes beautifully while still being a correlate the model
never acts on. `check_causal` is the one that settles it, by intervening and
measuring behaviour.

The refusal metric is substring matching against a phrase list. It is crude on
purpose and has errors in both directions. It works because what matters is the
effect size: a 40-plus point swing between conditions survives a matcher with a
steady error rate. Do not quote the absolute numbers as ground truth, and read
some raw generations with `spot_check`.

## The components

```
frontend/
  index.html    Layer 1, self-contained, no dependencies
  live.html     Layer 2 UI: prompt box, alpha slider, streamed tokens, projection chart
  demo.html     the three-act presenter demo
backend/
  model.py      loads the model and fixes the one indexing convention the rest relies on
  extract.py    runs the prompt pairs, caches every layer's activations, computes r̂
  projection.py the demo's fixed 2D plane, built from those cached activations
  hooks.py      the two interventions and the single alpha that drives them
  generate.py   token-by-token generation with the hooks applied, plus the projection trace
  refusal.py    the substring refusal matcher and why it is deliberately blunt
  app.py        FastAPI server: SSE token stream and static files, bound to 127.0.0.1
  checks/       the verification scripts above
  prompts/      index-matched harmful/harmless pairs for extraction, plus held-out sets
```

`model.py` pins the convention that layer `i` means the output of decoder block
`i`. `hooks.py` maps one slider to both operations: negative alpha ablates at
every block, positive alpha injects at the extraction layer only. Nothing outside
`model.py` indexes the hidden states directly, and nothing outside `hooks.py`
touches the residual stream.

## Design notes

[CLAUDE.md](CLAUDE.md) records the decisions and the reasoning behind them,
including where this deviates from the obvious approach: raw `transformers` hooks
rather than TransformerLens, hand-rolled canvas rather than three.js, projection
arithmetic forced to fp32 under a bf16 model, and the prompt-set rewrite that got
Layer 2 over its causal bar.

The presenter demo's design and build plan are in
[docs/superpowers/](docs/superpowers/).
