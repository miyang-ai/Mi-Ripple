---
name: mirage
description: Diagnose and restore grid-like, granular, scale-like, tiled, or ripple artifacts in AI-generated images. Use when a user asks to remove digital ripple, decoder grids, repeating scales, honeycomb texture, granular AI texture, or degradation caused by iterative image-to-image editing.
license: MIT
compatibility: Requires Python 3.11+, local file access, and permission to run shell commands. Network access and a MIYANG API key are optional and only needed for explicitly authorized regeneration.
metadata:
  author: MIYANG
  version: "0.1.0"
  repository: "https://github.com/miyang-ai/Mi-Ripple"
---

# Restore digital-ripple artifacts

Use the MIYANG `mirage` package to diagnose first, choose only a compatible
treatment, inspect the visual evidence, and return the actual output files to the
user.

Do not treat this as a generic denoiser. The workflow distinguishes:

1. isolated periodic lattice artifacts, which can be selectively notched;
2. granular artifacts in unstructured areas, which can receive masked reduction;
3. artifacts entangled with hair, foliage, fabric, stone, or other content,
   which require human review or optional cleaned-reference regeneration.

## Non-negotiable rules

- Work on the original image at native resolution. Do not resize, recompress, or
  screenshot it before diagnosis.
- Run the deterministic local pipeline before proposing regeneration.
- Never run paid or content-changing regeneration without the user's explicit
  approval in the current conversation.
- Never ask the user to paste an API key into chat. Ask them to set
  `MIYANG_API_KEY` in their environment.
- Files containing `_refclean` are model references, not deliverable images.
- Do not claim an artifact is removed only because a numeric score decreased.
  Inspect the heat map and comparison boards.
- Preserve generated JSON sidecars and the XML trace with the output.
- A pipeline `passed` result means measured filtering damage stayed within the
  configured limits. It does not mean a person has accepted the image.

## Locate or install the tool

First check whether the command is available:

```bash
mirage --help
```

If this skill is being used from a checkout of the repository, install that
checkout into an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Otherwise install from the canonical repository into an isolated environment:

```bash
python3 -m venv .mirage-venv
.mirage-venv/bin/python -m pip install \
  "git+https://github.com/miyang-ai/Mi-Ripple.git"
```

Use the corresponding environment's `mirage` executable for subsequent steps.
Do not modify the user's global Python environment.

## Inputs

Obtain:

- the exact local path of one source image;
- an output directory that will not overwrite the source;
- optional user priorities such as preserving hair, paper texture, foliage, or
  color.

If the image path is missing or ambiguous, ask for it. Do not choose a recently
used image on the user's behalf.

Supported image decoding is provided by Pillow. Prefer PNG for intermediate and
final files.

## Step 1: run the local pipeline

```bash
mirage "/absolute/path/input.png" "/absolute/path/output"
```

This performs diagnosis, deterministic routing, safe local treatment when
available, aligned verification, and provenance recording.

Read:

- `<stem>_restored.json` for the outcome and action sequence;
- `<stem>_input_diag.json` for measurements and selected windows;
- `<stem>_input_diag_board.png` for native-pixel diagnostic crops;
- `<stem>_input_scaleheat.png` for whole-frame flagged tiles;
- `<stem>_verify_board.png` when filtering was applied;
- `<stem>_restored.png` when a final local result was delivered.

Open the generated boards with the agent's image-reading capability. Do not
interpret JSON without checking the images.

## Step 2: handle the outcome

### `delivered`

Inspect the source, final image, heat map, and verification board.

Check specifically:

- hair remains continuous rather than becoming smooth blocks;
- foliage, flowers, gravel, fabric, stone, and paper texture were not erased;
- flat gradients did not gain cloudy stains or ringing;
- no new repeated scales, honeycomb cells, woven meshes, or wave bands appeared;
- framing, dimensions, and color remain unchanged on local filtering routes.

If the visual check passes, give the user:

1. the final image path or attachment;
2. the comparison/verification board;
3. a short statement of the route taken;
4. any remaining suspected artifact.

### `needs_human_decision`

This normally means the measured artifact overlaps image content and filtering
cannot safely separate it.

Show the diagnosis board and heat map. Explain which regions triggered review.
Then ask whether the user authorizes one regeneration attempt, explicitly
stating that it:

- can incur API cost;
- can change semantic details and identity;
- can change dimensions, framing, and color;
- still requires visual acceptance afterward.

Do not proceed until the user explicitly agrees.

### `failed`

Read the final JSON and the last step's error. Report the concrete failure and
retain all artifacts already written. Do not silently switch providers or repeat
a possibly billable request.

### `cancelled` or `step_limit`

Report that no deliverable was produced. Preserve the trace for diagnosis.

## Step 3: optional regeneration

Only after explicit approval, verify that `MIYANG_API_KEY` is set without
printing its value:

```bash
test -n "$MIYANG_API_KEY"
```

Run one bounded attempt by default:

```bash
mirage "/absolute/path/input.png" "/absolute/path/output-regenerated" \
  --allow-regen --max-regen 1
```

Never increase `--max-regen` without separate user approval. A timeout or
interrupted response can have unknown billing state; do not automatically retry
it.

After regeneration, inspect both the full image and the relevant native-pixel
windows. Compare identity, composition, geometry, hair, textured materials, and
color against the source. Report semantic or color changes separately from
artifact reduction.

## Interpretation guide

- `lattice.detected`: isolated periodic spectral components were detected. The
  flag may remain true after safe attenuation near the sampling limit.
- `granule.level=flat`: multiple unstructured windows support masked local
  reduction.
- `granule.level=pervasive`: broad evidence; regeneration or human review is
  normally safer than stronger filtering.
- `scale_index.level=structured`: similarly sized components cover at least 6%
  of grading tiles on the canonical canvas.
- `suspected`: report for inspection; it is not sufficient evidence for
  aggressive automatic treatment.

Legitimate repeated objects can trigger these measurements. Conversely,
directional wide scales, woven hair, and long wave bands can look severe while
receiving a low scale index. Visual evidence has final authority.

## User-facing completion format

Keep the handoff short:

```text
处理完成。
路径：诊断 → [实际步骤] → 验收
结果：<final image>
对比：<verification or diagnosis board>
备注：<remaining uncertainty or “未发现明显结构损伤”>
```

Embed or attach the final image and board when the interface supports it; do not
only print filesystem paths if the user cannot access those paths.
