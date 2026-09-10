# MIRAGE

## AI Skill — start here

**[Open `skills/mirage/SKILL.md` →](https://github.com/miyang-ai/Mi-Ripple/blob/main/skills/mirage/SKILL.md)**

AI coding agents should read this skill first. It contains the complete workflow
for installing MIRAGE, processing an image, inspecting the generated evidence,
requesting permission before paid regeneration, and returning the actual result.
The skill never authorizes paid regeneration on the user's behalf.

Install it from the canonical repository:

```text
https://github.com/miyang-ai/Mi-Ripple
```

Direct skill URL:

```text
https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/skills/mirage/SKILL.md
```

Reference implementation of **MIRAGE**, MIYANG's diagnosis-guided workflow for
restoring grid-like and scale-like artifacts introduced by iterative,
reference-conditioned AI image editing.

[Try MIRAGE on the MIYANG Lab website →](https://lab.miyang.cn/ripple/)

[Experience MIRAGE with Alice →](https://alice.miyang.cn/)

[Visit the MIYANG official website →](https://miyang.cn/)

MIRAGE does not apply one aggressive filter to every image. It first separates:

- **Periodic lattice artifacts**: isolated spectral peaks that can be selectively
  notched with low measured distortion.
- **Granular artifacts in unstructured regions**: 3–8 px texture that can be
  reduced behind a structure-protection mask.
- **Content-entangled artifacts**: repeated texture overlapping hair, foliage,
  fabric, or other legitimate detail. These require human review or optional
  regeneration from a cleaned reference.

> [!IMPORTANT]
> This is a research implementation, not a universal artifact detector. Automatic
> scores are triage signals. Review the generated heat maps and comparison boards
> before accepting an output.

## Before / after

These are the five comparisons currently used by the
[MIYANG Lab tool](https://lab.miyang.cn/ripple/). Open an image to inspect it at
native resolution.

### Night hair · Image 2.5

| Before | After |
| --- | --- |
| [![Night hair before restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/night-hair-before.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/night-hair-before.jpg) | [![Night hair after restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/night-hair-after.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/night-hair-after.jpg) |

The images share the same source composition but come from separate experimental
branches: direct regeneration versus cleaned-reference regeneration followed by
selective lattice notching. Regeneration is not pixel-aligned restoration and can
change fine semantic details.

### Moss gorge · Image 2.5

| Before | After |
| --- | --- |
| [![Moss gorge Image 2.5 before restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/moss-gorge-image25-before.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/moss-gorge-image25-before.jpg) | [![Moss gorge Image 2.5 after restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/moss-gorge-image25-after.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/moss-gorge-image25-after.jpg) |

### Moss gorge · Image 2.0

| Before | After |
| --- | --- |
| [![Moss gorge Image 2.0 before restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/moss-gorge-image2-before.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/moss-gorge-image2-before.jpg) | [![Moss gorge Image 2.0 after restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/moss-gorge-image2-after.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/moss-gorge-image2-after.jpg) |

### Rainforest path · Image 2.0

| Before | After |
| --- | --- |
| [![Rainforest path before restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/rainforest-path-before.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/rainforest-path-before.jpg) | [![Rainforest path after restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/rainforest-path-after.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/rainforest-path-after.jpg) |

### Ice cave · Image 2.0

| Before | After |
| --- | --- |
| [![Ice cave before restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/ice-cave-before.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/ice-cave-before.jpg) | [![Ice cave after restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/assets/comparison/ice-cave-after.jpg)](https://github.com/miyang-ai/Mi-Ripple/blob/main/assets/comparison/ice-cave-after.jpg) |

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/miyang-ai/Mi-Ripple.git
cd mirage
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

OpenCV-based face protection is optional:

```bash
pip install -e ".[face]"
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

Local deterministic processing is the default:

```bash
mirage input.png output/
```

The output directory contains:

- a diagnosis JSON and visual inspection boards;
- intermediate masks and processed candidates;
- a machine-readable XML decision trace;
- a final image when the deterministic route can deliver one;
- a sidecar containing hashes, parameters, measurements, and the final outcome.

If artifacts overlap image content, the default route stops with
`needs_human_decision`. Optional regeneration requires explicit permission and a
MIYANG API key:

```bash
export MIYANG_API_KEY=...
mirage input.png output/ --allow-regen --max-regen 1
```

Regeneration may change image content, dimensions, and color. Its output remains
a candidate until a person accepts it.

## Python API

```python
from pathlib import Path

from mirage.pipeline import run

result = run(Path("input.png"), Path("output"))
print(result["outcome"], result["final"])
```

Individual measurement and processing modules are also public:

- `mirage.diagnosis`
- `mirage.scale_index`
- `mirage.notch`
- `mirage.spatial`
- `mirage.reference`
- `mirage.verify`

## Method and evidence

The accompanying paper is:

> Yicheng Xu, Jiayin Chen, and Muting Wang. **MIRAGE: Restoring Images
> Degraded by Iterative AI Editing.** MIYANG Technology (Shanghai) Co., Ltd.,
> 2026.

The [LaTeX manuscript](paper/manuscript.tex), [references](paper/references.bib),
and publication figures are included in [`paper/`](paper/).

### Paper figures

[![Restoration results across moss gorge, wisteria tunnel, and ice cave](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/paper/figures/teaser_restoration_en.png)](https://github.com/miyang-ai/Mi-Ripple/blob/main/paper/figures/teaser_restoration_en.pdf)

| Artifact forms | Targeted restoration |
| --- | --- |
| [![Periodic lattice and granular artifact forms](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/paper/figures/ripple_forms_en.png)](https://github.com/miyang-ai/Mi-Ripple/blob/main/paper/figures/ripple_forms_en.pdf) | [![Before and after facial restoration](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/paper/figures/restoration_en.png)](https://github.com/miyang-ai/Mi-Ripple/blob/main/paper/figures/restoration_en.pdf) |

#### Selective notch versus broad spectral suppression

[![Input, selective notch, and soft-clipping comparison](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/paper/figures/notch_comparison_en.png)](https://github.com/miyang-ai/Mi-Ripple/blob/main/paper/figures/notch_comparison_en.pdf)

[![Residual comparison for selective notch and soft clipping](https://raw.githubusercontent.com/miyang-ai/Mi-Ripple/main/paper/figures/notch_residuals_en.png)](https://github.com/miyang-ai/Mi-Ripple/blob/main/paper/figures/notch_residuals_en.pdf)

See [docs/ALGORITHM.md](docs/ALGORITHM.md) for the decision flow and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) before interpreting reported scores.
Provider integration and billing boundaries are documented in
[docs/REGENERATION.md](docs/REGENERATION.md).
The publication-ready manuscript will be linked here when its permanent public
record is available.

The test suite uses synthetic deterministic inputs. The documented comparison and
paper figures are curated publication assets; private user images and internal
production archives are not included.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Brand and license

The source code is released under the [MIT License](LICENSE).

MIYANG, its logo, and associated product branding are trademarks or brand assets
of MIYANG Technology (Shanghai) Co., Ltd. The MIT License does not grant rights
to use those marks. See [TRADEMARKS.md](TRADEMARKS.md).

Copyright © 2026 MIYANG Technology (Shanghai) Co., Ltd.
