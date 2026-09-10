"""Diagnosis of periodic lattice and scale-like texture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .common import (
    bandpass,
    build_board,
    crop_tile,
    load_rgb,
    now_iso,
    rgb_to_L,
    sha256_of,
    structure_signals,
    to_img,
    write_json,
)
from .notch import DEFAULT_PARAMS as NOTCH_PARAMS
from .notch import find_isolated_peaks
from .scale_index import (
    CANON_CANVAS,
    SCALE_THRESHOLDS,
    scale_heatmap,
    scale_index,
    to_canon_canvas,
)
from .scale_index import (
    serializable as scale_serializable,
)
from .spatial import structure_mask

WINDOW = 96
THRESHOLDS = {
    "lattice_min_peaks": 2,
    "lattice_min_excess": 2.5,
    "granule_min_band_std": 0.35,
    "granule_max_kurtosis": 8.0,
    "granule_min_mid_blob_cov_pct": 19.0,
    "granule_max_anisotropy": 0.35,
    "flat_min_flagged": 2,
    "pervasive_min_flagged": 3,
    "pervasive_min_ratio": 0.6,
}


def band_anisotropy(band: np.ndarray) -> float:
    gy, gx = np.gradient(band)
    jxx = (gx * gx).mean()
    jyy = (gy * gy).mean()
    jxy = (gx * gy).mean()
    trace = jxx + jyy + 1e-12
    discriminant = np.sqrt(
        max(trace * trace / 4 - (jxx * jyy - jxy * jxy), 0.0)
    )
    return float(2 * discriminant / trace)


def window_features(lightness: np.ndarray) -> dict:
    band = bandpass(lightness)
    standard_deviation = float(band.std())
    normalized = (band - band.mean()) / max(standard_deviation, 1e-9)
    kurtosis = float((normalized**4).mean() - 3)
    anisotropy = band_anisotropy(band)
    areas: list[float] = []
    for sign in (1, -1):
        mask = sign * band > standard_deviation
        labels, count = ndimage.label(mask)
        if count:
            areas.extend(
                ndimage.sum(
                    mask,
                    labels,
                    index=np.arange(1, count + 1),
                ).tolist()
            )
    area_array = np.array(areas) if areas else np.array([0.0])
    middle = area_array[(area_array >= 12) & (area_array <= 300)]
    return {
        "band_std": round(standard_deviation, 4),
        "kurtosis": round(kurtosis, 2),
        "anisotropy": round(anisotropy, 3),
        "mid_blob_cov_pct": round(
            float(middle.sum() / lightness.size * 100),
            2,
        ),
        "mid_blob_median_area": round(
            float(np.median(middle)) if len(middle) else 0.0,
            1,
        ),
    }


def flag_granule(
    features: dict,
    thresholds: dict = THRESHOLDS,
) -> bool:
    return (
        features["band_std"] >= thresholds["granule_min_band_std"]
        and features["kurtosis"] <= thresholds["granule_max_kurtosis"]
        and features["mid_blob_cov_pct"]
        >= thresholds["granule_min_mid_blob_cov_pct"]
        and features["anisotropy"] <= thresholds["granule_max_anisotropy"]
    )


def _pick_windows(
    score: np.ndarray,
    count: int,
    window: int,
    minimum_score: float,
    windowed: bool = False,
) -> list[tuple[int, int, int, int]]:
    height, width = score.shape
    if height < window or width < window:
        return []
    mean = (
        score
        if windowed
        else ndimage.uniform_filter(score, window, mode="constant")
    )
    half = window // 2
    valid = np.full_like(mean, -np.inf)
    valid[half : height - half, half : width - half] = mean[
        half : height - half,
        half : width - half,
    ]
    taken = np.zeros_like(score, dtype=bool)
    boxes = []
    for _ in range(count):
        candidates = np.where(taken, -np.inf, valid)
        index = int(np.argmax(candidates))
        center_y, center_x = divmod(index, width)
        if (
            not np.isfinite(candidates[center_y, center_x])
            or candidates[center_y, center_x] < minimum_score
        ):
            break
        x1, y1 = center_x - half, center_y - half
        boxes.append((x1, y1, x1 + window, y1 + window))
        taken[
            max(0, y1 - half) : y1 + window + half,
            max(0, x1 - half) : x1 + window + half,
        ] = True
    return boxes


def select_windows(
    lightness: np.ndarray,
    flat_count: int = 6,
    oriented_count: int = 3,
) -> tuple[list, list, dict]:
    signals = structure_signals(lightness)
    weight, _ = structure_mask(lightness, "strict", signals)
    weight_mean = ndimage.uniform_filter(weight, WINDOW, mode="constant")
    weight_minimum = ndimage.minimum_filter(
        weight,
        WINDOW,
        mode="constant",
        cval=0.0,
    )
    band = bandpass(lightness)
    local_band = np.sqrt(
        ndimage.uniform_filter(band * band, WINDOW, mode="constant")
    )
    flat_score = np.where(
        (weight_mean >= 0.75) & (weight_minimum >= 0.25),
        local_band,
        -np.inf,
    )
    flat = _pick_windows(
        flat_score,
        flat_count,
        WINDOW,
        0.0,
        windowed=True,
    )
    oriented_score = signals["coh"] * (
        1 - np.clip(signals["gmag"] / 6.0, 0, 1)
    )
    oriented = _pick_windows(
        oriented_score,
        oriented_count,
        WINDOW,
        0.25,
    )
    return flat, oriented, signals


def diagnose(
    rgb: np.ndarray,
    thresholds: dict = THRESHOLDS,
    scale_thresholds: dict = SCALE_THRESHOLDS,
    scale_heat_out: Path | str | None = None,
) -> dict:
    lightness = rgb_to_L(rgb)
    canonical_rgb, normalized = to_canon_canvas(rgb)
    scale_result = scale_index(canonical_rgb, scale_thresholds)
    native_scale_result = (
        scale_index(rgb, scale_thresholds) if normalized else scale_result
    )
    peak_parameters = {
        key: value
        for key, value in NOTCH_PARAMS.items()
        if key != "feather"
    }
    lattice_result = find_isolated_peaks(lightness, **peak_parameters)["stats"]
    lattice = (
        lattice_result["isolated_components_kept"]
        >= thresholds["lattice_min_peaks"]
        and lattice_result["max_peak_excess"]
        >= thresholds["lattice_min_excess"]
    )
    flat_boxes, oriented_boxes, _ = select_windows(lightness)
    flat_rows = []
    for box in flat_boxes:
        x1, y1, x2, y2 = box
        features = window_features(lightness[y1:y2, x1:x2])
        features["flagged"] = flag_granule(features, thresholds)
        flat_rows.append({"box": list(box), **features})
    oriented_rows = [
        {
            "box": list(box),
            **window_features(
                lightness[box[1] : box[3], box[0] : box[2]]
            ),
        }
        for box in oriented_boxes
    ]
    flagged_count = sum(row["flagged"] for row in flat_rows)
    flagged_ratio = flagged_count / len(flat_rows) if flat_rows else 0.0
    if flagged_count == 0:
        granule_level = "none"
    elif (
        flagged_count >= thresholds["pervasive_min_flagged"]
        and flagged_ratio >= thresholds["pervasive_min_ratio"]
    ):
        granule_level = "pervasive"
    elif flagged_count >= thresholds["flat_min_flagged"]:
        granule_level = "flat"
    else:
        granule_level = "suspected"
    output = {
        "pixels": [int(lightness.shape[1]), int(lightness.shape[0])],
        "lattice": {"detected": bool(lattice), **lattice_result},
        "granule": {
            "level": granule_level,
            "flagged_windows": int(flagged_count),
            "flat_windows": flat_rows,
            "oriented_windows": oriented_rows,
        },
        "scale_index": scale_serializable(scale_result)
        | {
            "measured_on": (
                f"{canonical_rgb.shape[1]}x{canonical_rgb.shape[0]}"
            ),
            "canvas_normalized_for_grading": normalized,
            "canon_long_edge": CANON_CANVAS[0],
        },
        "scale_index_native": {
            "scale_tile_pct": native_scale_result["scale_tile_pct"],
            "level": native_scale_result["level"],
            "pixels": [int(lightness.shape[1]), int(lightness.shape[0])],
        },
        "thresholds": thresholds,
        "flags": {
            "lattice": bool(lattice),
            "granule_flat": granule_level in ("flat", "pervasive"),
            "granule_pervasive": granule_level == "pervasive",
            "granule_suspected": granule_level == "suspected",
            "scale_structured": scale_result["level"] == "structured",
            "scale_suspected": scale_result["level"] == "suspected",
        },
    }
    if scale_heat_out is not None:
        output["scale_index"]["heatmap"] = str(
            scale_heatmap(
                canonical_rgb,
                scale_result,
                Path(scale_heat_out),
                Path(scale_heat_out).stem,
            )
        )
    return output


def diag_board(
    rgb: np.ndarray,
    diagnosis: dict,
    out: Path,
    title: str,
) -> Path:
    lightness = rgb_to_L(rgb)
    band = bandpass(lightness)
    rows = []
    groups = (
        ("flat", diagnosis["granule"]["flat_windows"]),
        ("oriented", diagnosis["granule"]["oriented_windows"]),
    )
    for kind, source_rows in groups:
        for row in source_rows:
            box = tuple(row["box"])
            flagged = row.get("flagged", False)
            label = (
                f"{kind} {box} | band SD {row['band_std']} | "
                f"kurtosis {row['kurtosis']} | "
                f"mid-blob coverage {row['mid_blob_cov_pct']}%"
            )
            if flagged:
                label += " | FLAGGED"
            rows.append(
                {
                    "label": label,
                    "color": "#A44A2A" if flagged else "#20252B",
                    "tiles": [
                        crop_tile(rgb, box),
                        to_img(
                            128
                            + band[box[1] : box[3], box[0] : box[2]]
                            * 6
                            * 2.55
                        ),
                    ],
                }
            )
    lattice = diagnosis["lattice"]
    subtitle = (
        f"lattice={'detected' if lattice['detected'] else 'not detected'}; "
        f"granule={diagnosis['granule']['level']}; "
        f"scale tiles={diagnosis['scale_index']['scale_tile_pct']}% "
        f"({diagnosis['scale_index']['level']}). "
        "Columns: source pixels; 3–8 px residual at 6x."
    )
    build_board(title, subtitle, rows, out=out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose digital-ripple artifact classes."
    )
    parser.add_argument("image")
    parser.add_argument("out_dir")
    arguments = parser.parse_args()
    source = Path(arguments.image)
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(source)
    result = diagnose(
        rgb,
        scale_heat_out=output_dir / f"{source.stem}_scaleheat.png",
    )
    result.update(
        {
            "stage": "diagnose",
            "source": str(source),
            "source_sha256": sha256_of(source),
            "time": now_iso(),
        }
    )
    result["board"] = str(
        diag_board(
            rgb,
            result,
            output_dir / f"{source.stem}_diag_board.png",
            f"Diagnosis: {source.name}",
        )
    )
    write_json(output_dir / f"{source.stem}_diag.json", result)
    print(
        json.dumps(
            {
                "lattice": result["flags"]["lattice"],
                "granule_level": result["granule"]["level"],
                "scale_tile_pct": result["scale_index"]["scale_tile_pct"],
                "scale_level": result["scale_index"]["level"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
