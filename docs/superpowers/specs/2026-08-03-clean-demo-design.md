# Presenter demo — design

A cleaner, presenter-driven version of the refusal-direction demo, to be shown
live to a room of engineers. The existing `index.html` and `live.html` stay as
the detailed version; this is a new, additional front end that reuses the same
backend. Nothing is removed.

## Why

The current pages carry a lot of detail — model card, per-layer stats, chart
legends, explanatory prose. That is useful for someone digging in on their own,
but it is clutter when you are narrating over a shared screen. The demo also
never shows the latent space of the *real* model: Layer 1 is a toy 3D cloud,
Layer 2 is text plus a projection line chart, and there is no spatial view that
ties the two together.

## What we're building

One page, `frontend/demo.html`, structured as three acts you step through with
the arrow keys or space. Big central stage, minimal chrome, a single slider, and
a dot indicator for which act you are on.

### Act 1 — the idea (toy)

The existing Layer 1 cloud, decluttered: two clouds of synthetic points, `r̂`
drawn through them, one ablate/inject slider. Reuses the canvas projection code
from `index.html`. Keeps the "this is a toy" label.

### Act 2 — it's real (the bridge)

A new 2D scatter of the *real* model's activations, so the room can see the same
structure on Qwen2.5-3B that they just saw on toy points. This is the new piece.

The projection is fixed and chosen for honesty, not discovered:

- **x-axis** = projection onto `r̂` (the refusal direction itself).
- **y-axis** = the leading direction of the variance left once `r̂` is removed
  (call it `u`, with `u ⟂ r̂`).

Because the axes are chosen this way, the harmful and harmless activations
separate left-to-right by construction of what `r̂` is, and `y` shows they are
otherwise mixed. That is the same story as Act 1's "collapse onto the axis". The
current prompt's activation appears as a marked point; under ablation it slides
to `x ≈ 0`, under injection it slides right. The act states plainly that this is
a 2D shadow of a 2048-dimensional space, chosen to show `r̂` — not a structure
the model handed us.

This deliberately revisits something CLAUDE.md had cut ("PCA/UMAP/t-SNE
switcher… representation geometry platform"). What we are building is narrower: a
single fixed projection built to illustrate one direction, not a switcher and not
a general tool. That keeps it inside the honesty commitments.

### Act 3 — the model obeys (live)

The live generation from Layer 2, decluttered: prompt, one slider, the streamed
output, and the projection trace. The headline is the flip — the same prompt
refused at baseline and complied-with under ablation, shown together. Reuses the
SSE streaming from `live.html`. Keeps the "substring matcher, read effect sizes"
caveat.

## Architecture

Front end: `frontend/demo.html`, a slide shell over the three acts. Keyboard
navigation (`←` `→` / space), a dot indicator, big type. It reuses the canvas
cloud code (Act 1) and the SSE client (Act 3), so the only genuinely new front-
end code is the shell and the Act 2 scatter. No build step, no CDN, in keeping
with the current stack.

Back end, small additions to what exists:

- `extract.py` computes and stores a second axis `u` (leading PC of the
  activations after `r̂` is projected out; unit length, orthogonal to `r̂`) and
  the precomputed 2D coordinates for the harmful and harmless sets, into the
  existing artifact. No extra forward passes — it reuses the cached activations.
- One read-only endpoint, `/api/projection`, serves those points.
- The generation stream gains a `y = h·u` field next to the `projection` (`x`) it
  already emits, so the live prompt's dot can sit and move in the Act 2 scatter.

## Visual style

Soft, paper-like, minimal. Muted accents on a cool greige stage, sized to read
from the back of a room.

| role | colour |
|---|---|
| paper / stage | `#EBEBE4` / `#F4F4EE` |
| ink / muted | `#2A2C2C` / `#6E706A` |
| harmless | `#6E93AE` (dusty blue) |
| harmful | `#C67E60` (terracotta) |
| r̂ | `#7C9E77` (sage) |

## Testing

Mirror the existing pattern:

- A self-test for Act 2's projection: the harmful and harmless clouds must
  separate along `x` (the `r̂` axis), and `u ⟂ r̂` to tolerance. This lives with
  the direction checks.
- The existing `check_causal`, `check_direction`, `check_stream` and
  `check_local_only` continue to cover the shared backend.
- Manual: step through all three acts, confirm the keyboard navigation, and
  confirm the live point in Act 2 moves as the slider changes.

## Out of scope

- No switcher between projection methods; one fixed projection only.
- No changes to `index.html` or `live.html`; they remain the detailed version.
- Nothing that reframes this as a general representation-geometry tool.

## Definition of done

- `frontend/demo.html` steps through the three acts with the keyboard.
- Act 2 shows the real projection with the live prompt's point moving under the
  slider.
- The projection self-test passes; the existing suites stay green.
- The presenter demo reads clearly on a shared screen without narration crutches.
