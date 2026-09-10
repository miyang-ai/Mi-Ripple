"""Structure-aware spatial suppression for scale-like granules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .common import (
    bandpass,
    lab_to_rgb8,
    load_rgb,
    now_iso,
    rgb_to_lab,
    sha256_of,
    smoothstep,
    structure_signals,
    write_json,
)

PROFILES = {
    "loose": {
        "edge": (1.5, 4.0),
        "coh": (0.35, 0.70),
        "dense": (5.0, 10.0),
    },
    "strict": {
        "edge": (0.8, 2.5),
        "coh": (0.15, 0.45),
        "dense": (1.8, 4.0),
    },
}


def structure_mask(
    lightness: np.ndarray,
    profile: str = "strict",
    signals: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Return treatment permission where 1 is editable and 0 is protected."""
    parameters = PROFILES[profile]
    signals = signals or structure_signals(lightness)
    protect = 1 - (
        1 - smoothstep(signals["gmag"], *parameters["edge"])
    ) * (
        1 - smoothstep(signals["coh"], *parameters["coh"])
    ) * (
        1 - smoothstep(signals["hp_energy"], *parameters["dense"])
    )
    weight = np.clip(ndimage.gaussian_filter(1 - protect, 3.0), 0, 1)
    stats = {
        "profile": profile,
        "area_pct_w_gt_0p5": round(float((weight > 0.5).mean() * 100), 2),
        "area_pct_w_lt_0p1": round(float((weight < 0.1).mean() * 100), 2),
        "mean_w": round(float(weight.mean()), 4),
    }
    return weight, stats


def iso_clean(
    rgb: np.ndarray,
    strength: float = 1.0,
    profile: str = "strict",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reduce the 3–8 px band only where the structure mask permits it."""
    lab = rgb_to_lab(rgb)
    weight, mask_stats = structure_mask(lab[..., 0], profile)
    removed = {}
    for index, name in enumerate(("L", "a", "b")):
        difference = strength * weight * bandpass(lab[..., index])
        lab[..., index] -= difference
        removed[name] = round(float(difference.std()), 4)
    return lab_to_rgb8(lab), weight, {
        "strength": strength,
        "mask": mask_stats,
        "removed_std_by_channel": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce scale-like texture behind a structure mask."
    )
    parser.add_argument("image")
    parser.add_argument("out_dir")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="strict")
    parser.add_argument("--strength", type=float, default=1.0)
    arguments = parser.parse_args()
    source = Path(arguments.image)
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output, weight, stats = iso_clean(
        load_rgb(source),
        arguments.strength,
        arguments.profile,
    )
    output_path = output_dir / f"{source.stem}_iso_{arguments.profile}.png"
    Image.fromarray(output).save(output_path)
    Image.fromarray((weight * 255).astype(np.uint8)).save(
        output_dir / f"{source.stem}_iso_{arguments.profile}_mask.png"
    )
    write_json(
        output_path.with_suffix(".json"),
        {
            "stage": "iso_clean",
            "grade": (
                "deliverable"
                if arguments.profile == "strict"
                else "reference_only"
            ),
            "source": str(source),
            "source_sha256": sha256_of(source),
            "output": str(output_path),
            **stats,
            "time": now_iso(),
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
