"""Aligned distortion verification for deliverable-grade filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (
    build_board,
    crop_tile,
    load_rgb,
    now_iso,
    residual_tile,
    rgb_to_L,
    sha256_of,
    window_damage,
    write_json,
)

LIMITS = {
    "structure_max_diff_std": 0.6,
    "structure_min_hf_ratio": 0.90,
    "flat_max_band_increase": 0.02,
    "whole_max_diff_std": 1.0,
}


def verify(
    before: np.ndarray,
    after: np.ndarray,
    flat_boxes: list,
    structure_boxes: list,
    limits: dict = LIMITS,
) -> dict:
    if before.shape != after.shape:
        raise ValueError(
            "Verification requires pixel-aligned images: "
            f"{before.shape} vs {after.shape}"
        )
    before_lightness = rgb_to_L(before)
    after_lightness = rgb_to_L(after)
    height, width = before_lightness.shape
    reasons: list[str] = []
    flat_rows = []
    structure_rows = []
    for box in flat_boxes:
        damage = window_damage(
            before_lightness,
            after_lightness,
            tuple(box),
        )
        damage["box"] = list(box)
        damage["ok"] = (
            damage["band_std_after"]
            <= damage["band_std_before"] + limits["flat_max_band_increase"]
        )
        if not damage["ok"]:
            reasons.append(
                f"Flat window {box}: band SD increased "
                f"{damage['band_std_before']} -> {damage['band_std_after']}"
            )
        flat_rows.append(damage)
    for box in structure_boxes:
        damage = window_damage(
            before_lightness,
            after_lightness,
            tuple(box),
        )
        damage["box"] = list(box)
        damage["ok"] = (
            damage["diff_std"] <= limits["structure_max_diff_std"]
            and damage["hf_std_ratio"]
            >= limits["structure_min_hf_ratio"]
        )
        if not damage["ok"]:
            reasons.append(
                f"Structure window {box}: residual SD "
                f"{damage['diff_std']}, HF retention "
                f"{damage['hf_std_ratio']}"
            )
        structure_rows.append(damage)
    whole = window_damage(
        before_lightness,
        after_lightness,
        (0, 0, width, height),
    )
    if whole["diff_std"] > limits["whole_max_diff_std"]:
        reasons.append(
            f"Whole-image residual SD {whole['diff_std']} > "
            f"{limits['whole_max_diff_std']}"
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "limits": limits,
        "whole_image": whole,
        "flat_windows": flat_rows,
        "structure_windows": structure_rows,
    }


def verify_board(
    before: np.ndarray,
    after: np.ndarray,
    result: dict,
    out: Path,
    title: str,
) -> Path:
    difference = rgb_to_L(after) - rgb_to_L(before)
    rows = []
    groups = (
        ("flat", result["flat_windows"]),
        ("structure", result["structure_windows"]),
    )
    for kind, source_rows in groups:
        for row in source_rows:
            box = tuple(row["box"])
            label = (
                f"{kind} {box} | band SD "
                f"{row['band_std_before']} -> {row['band_std_after']} | "
                f"residual SD {row['diff_std']} | "
                f"HF retention {row['hf_std_ratio']}"
            )
            if not row["ok"]:
                label += " | LIMIT EXCEEDED"
            rows.append(
                {
                    "label": label,
                    "color": "#2E7D5B" if row["ok"] else "#A44A2A",
                    "tiles": [
                        crop_tile(before, box),
                        crop_tile(after, box),
                        residual_tile(difference, box),
                    ],
                }
            )
    whole = result["whole_image"]
    subtitle = (
        f"{'passed' if result['passed'] else 'failed'}; whole residual SD "
        f"{whole['diff_std']}, max {whole['diff_abs_max']}. "
        "Columns: before; after; residual at 6x. "
        "A pass is not human acceptance."
    )
    build_board(title, subtitle, rows, out=out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify aligned filtering distortion."
    )
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("out_dir")
    parser.add_argument("--diag")
    arguments = parser.parse_args()
    before = load_rgb(arguments.before)
    after = load_rgb(arguments.after)
    flat_boxes = []
    structure_boxes = []
    if arguments.diag:
        diagnosis = json.loads(
            Path(arguments.diag).read_text(encoding="utf-8")
        )
        flat_boxes = [
            row["box"] for row in diagnosis["granule"]["flat_windows"]
        ]
        structure_boxes = [
            row["box"]
            for row in diagnosis["granule"]["oriented_windows"]
        ]
    result = verify(before, after, flat_boxes, structure_boxes)
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(arguments.after).stem
    result.update(
        {
            "stage": "verify",
            "before": arguments.before,
            "after": arguments.after,
            "after_sha256": sha256_of(arguments.after),
            "time": now_iso(),
        }
    )
    result["board"] = str(
        verify_board(
            before,
            after,
            result,
            output_dir / f"{stem}_verify_board.png",
            f"Verification: {Path(arguments.after).name}",
        )
    )
    write_json(output_dir / f"{stem}_verify.json", result)
    print(json.dumps({"passed": result["passed"], "reasons": result["reasons"]}))


if __name__ == "__main__":
    main()
