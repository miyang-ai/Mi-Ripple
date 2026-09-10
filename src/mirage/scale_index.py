"""Whole-frame scale-like texture index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .common import (
    bandpass,
    font,
    load_rgb,
    now_iso,
    rgb_to_L,
    sha256_of,
    write_json,
)

TILE, STRIDE = 128, 64
CANON_CANVAS = (1280, 720)
SCALE_THRESHOLDS = {
    "min_cov_pct": 19.0,
    "max_area_cv": 0.65,
    "min_circ": 0.55,
    "max_aniso": 0.35,
    "min_band_std": 0.30,
    "blob_area_min": 12,
    "blob_area_max": 300,
    "suspected_min_pct": 2.0,
    "structured_min_pct": 6.0,
}


def _band_anisotropy(band: np.ndarray) -> float:
    gy, gx = np.gradient(band)
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    trace = jxx + jyy
    if trace <= 1e-12:
        return 0.0
    discriminant = float(
        np.sqrt(max((jxx - jyy) ** 2 + 4 * jxy * jxy, 0.0))
    )
    return discriminant / trace


def tile_features(
    lightness: np.ndarray,
    thresholds: dict = SCALE_THRESHOLDS,
) -> dict:
    band = bandpass(lightness)
    standard_deviation = float(band.std())
    areas: list[float] = []
    circularities: list[float] = []
    for sign in (1, -1):
        mask = sign * band > standard_deviation
        labels, count = ndimage.label(mask)
        if not count:
            continue
        indices = np.arange(1, count + 1)
        component_areas = ndimage.sum(mask, labels, index=indices)
        eroded_boundary = mask & ~ndimage.binary_erosion(mask)
        perimeters = ndimage.sum(eroded_boundary, labels, index=indices)
        keep = (component_areas >= thresholds["blob_area_min"]) & (
            component_areas <= thresholds["blob_area_max"]
        )
        areas.extend(component_areas[keep].tolist())
        circularities.extend(
            (
                4
                * np.pi
                * component_areas[keep]
                / np.maximum(perimeters[keep], 1) ** 2
            )
            .clip(0, 1)
            .tolist()
        )
    area_array = np.array(areas) if areas else np.array([0.0])
    coverage = float(area_array.sum() / lightness.size * 100)
    area_cv = (
        float(area_array.std() / max(area_array.mean(), 1e-9))
        if len(areas) >= 5
        else 9.9
    )
    circularity = float(np.mean(circularities)) if circularities else 0.0
    return {
        "band_std": standard_deviation,
        "cov_pct": coverage,
        "area_cv": area_cv,
        "circ": circularity,
        "aniso": _band_anisotropy(band),
        "n_blobs": len(areas),
    }


def is_scale_tile(
    features: dict,
    thresholds: dict = SCALE_THRESHOLDS,
) -> bool:
    return (
        features["band_std"] >= thresholds["min_band_std"]
        and features["cov_pct"] >= thresholds["min_cov_pct"]
        and features["area_cv"] <= thresholds["max_area_cv"]
        and features["circ"] >= thresholds["min_circ"]
        and features["aniso"] <= thresholds["max_aniso"]
    )


def scale_level(
    percentage: float,
    thresholds: dict = SCALE_THRESHOLDS,
) -> str:
    if percentage >= thresholds["structured_min_pct"]:
        return "structured"
    if percentage >= thresholds["suspected_min_pct"]:
        return "suspected"
    return "none"


def to_canon_canvas(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """Downsample grading input to a 1280 px long edge; never upscale."""
    height, width = rgb.shape[:2]
    if width <= CANON_CANVAS[0]:
        return rgb, False
    target = (
        CANON_CANVAS[0],
        max(1, int(round(height * CANON_CANVAS[0] / width))),
    )
    image = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
    resized = image.resize(target, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64) / 255.0, True


def scale_index(
    rgb: np.ndarray,
    thresholds: dict = SCALE_THRESHOLDS,
) -> dict:
    lightness = rgb_to_L(rgb)
    height, width = lightness.shape
    ys = list(range(0, height - TILE + 1, STRIDE))
    xs = list(range(0, width - TILE + 1, STRIDE))
    flags = np.zeros((len(ys), len(xs)), dtype=bool)
    coverages = np.zeros_like(flags, dtype=float)
    area_cvs = np.full_like(coverages, 9.9)
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            features = tile_features(
                lightness[y : y + TILE, x : x + TILE],
                thresholds,
            )
            flags[row, column] = is_scale_tile(features, thresholds)
            coverages[row, column] = features["cov_pct"]
            area_cvs[row, column] = features["area_cv"]
    percentage = float(flags.mean() * 100) if flags.size else 0.0
    finite_area_cvs = area_cvs[area_cvs < 9]
    return {
        "scale_tile_pct": round(percentage, 1),
        "level": scale_level(percentage, thresholds),
        "tiles": int(flags.size),
        "flagged_tiles": int(flags.sum()),
        "median_cov_pct": (
            round(float(np.median(coverages)), 1) if coverages.size else 0.0
        ),
        "median_area_cv": (
            round(float(np.median(finite_area_cvs)), 3)
            if finite_area_cvs.size
            else 9.9
        ),
        "tile": TILE,
        "stride": STRIDE,
        "thresholds": thresholds,
        "flag_grid": flags,
        "grid_origin": {"ys": ys, "xs": xs},
    }


def scale_heatmap(
    rgb: np.ndarray,
    result: dict,
    out: Path,
    title: str = "",
) -> Path:
    image = Image.fromarray(
        np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    ).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)
    ys = result["grid_origin"]["ys"]
    xs = result["grid_origin"]["xs"]
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            if result["flag_grid"][row, column]:
                drawing.rectangle(
                    [x, y, x + TILE, y + TILE],
                    fill=(217, 48, 37, 70),
                )
    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    drawing = ImageDraw.Draw(rendered)
    drawing.rectangle([0, 0, rendered.size[0], 34], fill=(255, 255, 255))
    drawing.text(
        (10, 6),
        (
            f"{title}  scale-like tiles {result['scale_tile_pct']:.1f}% "
            f"({result['level']}; red = flagged)"
        ),
        fill="#1C1F23",
        font=font(18),
    )
    rendered.save(out)
    return out


def serializable(result: dict) -> dict:
    ys = result["grid_origin"]["ys"]
    xs = result["grid_origin"]["xs"]
    boxes = [
        [xs[column], ys[row], xs[column] + TILE, ys[row] + TILE]
        for row in range(len(ys))
        for column in range(len(xs))
        if result["flag_grid"][row, column]
    ]
    return {
        key: value
        for key, value in result.items()
        if key not in ("flag_grid", "grid_origin")
    } | {"flagged_tile_boxes": boxes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whole-frame scale-like texture."
    )
    parser.add_argument("image")
    parser.add_argument("out_dir")
    arguments = parser.parse_args()
    source = Path(arguments.image)
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(source)
    result = scale_index(rgb)
    heatmap = scale_heatmap(
        rgb,
        result,
        output_dir / f"{source.stem}_scaleheat.png",
        source.stem,
    )
    record = {
        "stage": "scale_index",
        "source": str(source),
        "source_sha256": sha256_of(source),
        "heatmap": str(heatmap),
        "time": now_iso(),
        **serializable(result),
    }
    write_json(output_dir / f"{source.stem}_scale_index.json", record)
    print(json.dumps(serializable(result), indent=2))


if __name__ == "__main__":
    main()
