# Mi-Ripple algorithm

Mi-Ripple is a diagnosis-guided restoration workflow for structured artifacts that
can accumulate during iterative, reference-conditioned AI image editing. The
implementation measures in CIELAB lightness unless a module states otherwise.

## Artifact classes

### Periodic lattice

A periodic lattice appears as small isolated peaks in the two-dimensional
spectrum. Mi-Ripple compares log FFT amplitude with a local two-dimensional median
baseline, keeps only small connected components above that baseline, and
attenuates those components with a feathered mask while preserving phase.

This route is pixel-aligned and is verified against the input.

### Granular texture in unstructured regions

Scale-like granules occupy a broad 3–8 px band rather than isolated spectral
points. A structure mask combines edge magnitude, orientation coherence, and
local high-frequency density. Band reduction is permitted only where these
signals indicate weak structure.

### Content-entangled texture

In hair, foliage, fabric, stone, and similar regions, visible artifacts can
overlap legitimate content in frequency, direction, and scale. No filter can
separate them without potentially removing real detail. Mi-Ripple therefore
requests human review or, when explicitly authorized, creates a
`reference_only` cleaned image for one bounded regeneration attempt.

## Decision flow

```text
input
  └─ diagnose
      ├─ clean ────────────────────────────────────> return original
      ├─ isolated lattice ─────────────────────────> notch → verify
      ├─ flat-region granules ─────────────────────> masked reduction → notch → verify
      └─ content-entangled / structured granules
           ├─ regeneration not authorized ─────────> request human decision
           └─ regeneration authorized
                └─ clean reference → regenerate → diagnose → filter if safe → verify
```

When masked reduction exceeds a structural damage limit, the pipeline rolls back
to a notch-only candidate.

## Default calibration

The default thresholds reproduce the measurements used in the Mi-Ripple paper.
They were calibrated on limited samples and are not universal classifier
thresholds.

- Granular band: Gaussian difference at sigma 1.0 and 3.0.
- Flat-window size: 96 px.
- Whole-frame scale-index tile: 128 px with a 64 px stride.
- Scale-index grading: below 2% `none`, 2–6% `suspected`, at least 6%
  `structured`.
- Structure verification: residual standard deviation at most 0.6 CIELAB L and
  high-frequency retention at least 0.90 in selected structure windows.

Whole-frame scale grading is measured on a long edge of at most 1280 px while
preserving aspect ratio. Periodic lattice diagnosis and all processing run at
native resolution.

## Records

Every pipeline run records:

- source and output SHA-256 hashes;
- dimensions and algorithm parameters;
- selected windows and scale-index tiles;
- allowed actions, selected actions, and reasons;
- verification results and rollback decisions;
- regeneration metadata without credentials.

These records support reproducibility; they do not replace visual inspection.
