# Refusal direction demo

An interactive demo of Arditi et al., [*Refusal in Language Models Is Mediated by
a Single Direction*](https://arxiv.org/abs/2406.11717). It is aimed at engineers
who want to see for themselves what "a direction in activation space" means
rather than take it on faith.

It is a four-slide demo you step through with the arrow keys. You see how the
refusal direction is built from forty prompt pairs, watch a real model
(Qwen2.5-3B) refuse a prompt and then answer it once that one direction is
removed, see where in the 36 blocks the decision gets made, and end in the
model's own activation space with your prompt marked in it.

Everything on screen is measured. There is no synthetic data anywhere.

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

Extract the direction, then start the server:

```bash
export LATENTSPACE_MODEL=Qwen/Qwen2.5-3B-Instruct   # already the default; setting it just makes it explicit
python -m backend.extract     # difference-of-means over 40 matched prompt pairs, writes backend/artifacts/
python -m backend.app         # serves http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 and step through with `←` and `→`. The two operations
on the slider are:

- **Ablate** (`α < 0`): `h ← h − t(h·r̂)r̂`, at every block. The residual stream
  loses its component along r̂ entirely.
- **Inject** (`α > 0`): `h ← h + α·k·r̂` at the extraction block, where `k` is
  scaled in units of the raw difference-of-means norm.

The four slides:

1. **One direction, from forty pairs** — the extraction corpus, three pairs at a
   time, with the difference-of-means formula. Click through all forty to show
   the breadth: fraud, lock-picking, surveillance, forgery. Each pair holds the
   grammar fixed, so the direction is refusal rather than "security words".
2. **The model obeys** — your prompt run at baseline and under the intervention,
   side by side, streamed live. Refusal on the left, compliance on the right.
3. **Where the decision forms** — alignment with r̂ at every block and every
   token, for your prompt and a fixed benign reference on one shared scale. Flat
   and identical through the system prompt, igniting mid-stack once the request
   turns harmful, hottest at the boxed final token — the position whose residual
   stream becomes the first word the model says, and the one r̂ was extracted
   from.
4. **Why** — the model's real activations in 3D, rotatable. One axis is r̂; the
   other two are the strongest directions left once r̂ is removed. The classes
   split by 3.94 SD along r̂ and 0.00 SD along both others, which is the "single
   direction" claim as a picture. The prompt you just ran appears as a marked
   point, and the slider applies the real arithmetic to it: ablation slides it
   to the middle of the harmless cloud, injection pushes it past the harmful
   one. "Line up with r̂" snaps the camera side-on so the split reads directly.

Nothing on any slide is synthetic. The cloud is 40 harmful and 40 harmless
prompts' activations at block 21, read straight out of the extraction artifact,
and the marked point is measured at your prompt's last token on the baseline run
— the same quantity, at the same block, as every other point on screen.

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
| `python -m backend.checks.check_direction` | unit norm, split-half stability, class separation, per-layer table |
| `python -m backend.checks.check_projection` | the demo's 3D frame: the leftover axes are orthonormal and orthogonal to r̂, and the classes separate along r̂ and not along them |
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
  index.html    the whole front end: four slides, no build step, no dependencies
backend/
  model.py      loads the model and fixes the one indexing convention the rest relies on
  extract.py    runs the prompt pairs, caches every layer's activations, computes r̂
  projection.py the demo's fixed 3D frame (r̂ plus two leftover axes), from those cached activations
  scan.py       alignment with r̂ at every block and token of a prompt, in one forward pass
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
the causal check over its bar.
