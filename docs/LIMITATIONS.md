# Limitations and responsible use

Read this document before interpreting MIRAGE scores or accepting an output.

## The detector is not universal

The default thresholds were calibrated on a limited collection of generated
images. Dense legitimate leaves, flowers, gravel, paper texture, brushwork, and
fabric can overlap the measured band. Heat maps and selected windows must be
checked against image semantics.

The scale index measures the coverage of similarly sized components. It does not
measure their visibility, semantic correctness, or visual quality. Directional
wide scales, woven patterns, and long wave bands may be visually obvious while
receiving a low scale index.

## Filtering cannot repair every artifact

Periodic isolated peaks are separable. Content-entangled texture is not.
Increasing filter strength in hair, foliage, fabric, or stone can remove valid
detail and produce cloudy or washed-out results. The reference-clean output is
therefore never a deliverable image.

The historical radial-whitening filter is intentionally excluded from this
repository because it damaged directional content.

## Regeneration is not restoration in the pixel-aligned sense

Optional regeneration can change:

- identity, pose, expression, or object geometry;
- fine texture and brushwork;
- image dimensions and framing;
- color and saturation.

The pipeline bounds regeneration attempts and requires explicit authorization,
but those controls cannot guarantee semantic fidelity. A person must compare and
accept regenerated candidates.

## A residual lattice may remain

Lattice notching stops before attenuation would threaten legitimate detail near
the sampling limit. A detection flag can remain true after treatment even when
the strongest component has been reduced.

## Numerical versions matter

FFT, median filtering, connected-component measurements, and color conversion
can change across NumPy, SciPy, scikit-image, and Pillow versions. Release
validation includes regression tests, but users requiring exact paper values
should reproduce the dependency versions recorded with the corresponding
release.

## Provenance

MIRAGE is intended for quality control, not for hiding that an image was
AI-generated. Processing records should remain attached to distributed outputs
when provenance matters.
