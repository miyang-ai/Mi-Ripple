"""Selective spectral notching for isolated periodic lattice components."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .common import (
    lab_to_rgb8,
    load_rgb,
    now_iso,
    rgb_to_lab,
    sha256_of,
    write_json,
)

DEFAULT_PARAMS = {
    "local_win": 21,
    "peak_tau": 1.2,
    "max_support": 80,
    "low_cut": 24,
    "feather": 1.5,
}


def default_workers() -> int:
    value = os.environ.get("MIRAGE_WORKERS", "").strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return max(1, min(8, os.cpu_count() or 1))


def median_baseline(
    array: np.ndarray,
    size: int,
    workers: int | None = None,
) -> np.ndarray:
    """Compute a deterministic striped median filter with full-window halos."""
    worker_count = default_workers() if workers is None else max(1, int(workers))
    height = array.shape[0]
    if worker_count == 1 or height < 2:
        return ndimage.median_filter(array, size=size, mode="nearest")
    halo = size // 2
    bounds = np.linspace(0, height, worker_count + 1).astype(int)
    output = np.empty_like(array)

    def work(index: int) -> None:
        low = int(bounds[index])
        high = int(bounds[index + 1])
        if high <= low:
            return
        start = max(0, low - halo)
        end = min(height, high + halo)
        piece = ndimage.median_filter(
            array[start:end],
            size=size,
            mode="nearest",
        )
        output[low:high] = piece[low - start : low - start + (high - low)]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(work, range(worker_count)))
    return output


def find_isolated_peaks(
    channel: np.ndarray,
    local_win: int = 21,
    peak_tau: float = 1.2,
    max_support: int = 80,
    low_cut: int = 24,
    pad: int = 64,
    workers: int | None = None,
) -> dict:
    """Find small connected spectral components above a local median baseline."""
    padded = np.pad(channel, pad, mode="reflect")
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    del padded
    amplitude = np.abs(spectrum)
    log_amplitude = np.log1p(amplitude)
    local_baseline = median_baseline(log_amplitude, local_win, workers)
    excess = log_amplitude - local_baseline
    del log_amplitude, local_baseline
    height, width = excess.shape
    center_y, center_x = height // 2, width // 2
    yy, xx = np.ogrid[0:height, 0:width]
    radius = np.hypot(yy - center_y, xx - center_x)
    candidates = (excess > peak_tau) & (radius >= low_cut)
    del radius
    labels, count = ndimage.label(candidates)
    sizes = (
        ndimage.sum(candidates, labels, index=np.arange(1, count + 1))
        if count
        else np.array([])
    )
    kept_ids = [
        index + 1 for index, size in enumerate(sizes) if size <= max_support
    ]
    lookup = np.zeros(count + 1, dtype=bool)
    if kept_ids:
        lookup[kept_ids] = True
    keep = lookup[labels]
    peaks = []
    if kept_ids:
        positions = ndimage.maximum_position(excess, labels, index=kept_ids)
        if len(kept_ids) == 1:
            positions = [positions]
        for component_id, (peak_y, peak_x) in zip(
            kept_ids,
            positions,
            strict=True,
        ):
            peak_y, peak_x = int(peak_y), int(peak_x)
            frequency_y = (peak_y - center_y) / height
            frequency_x = (peak_x - center_x) / width
            frequency_radius = float(np.hypot(frequency_y, frequency_x))
            peaks.append(
                {
                    "excess": round(float(excess[peak_y, peak_x]), 3),
                    "support": int(sizes[component_id - 1]),
                    "period_px": (
                        round(1.0 / frequency_radius, 2)
                        if frequency_radius > 0
                        else None
                    ),
                    "angle_deg": round(
                        float(np.degrees(np.arctan2(frequency_y, frequency_x))),
                        1,
                    ),
                }
            )
    peaks.sort(key=lambda peak: -peak["excess"])
    return {
        "f": spectrum,
        "amp": amplitude,
        "excess": excess,
        "keep": keep,
        "cand": candidates,
        "pad": pad,
        "shape": channel.shape,
        "stats": {
            "components_above_tau": int(count),
            "isolated_components_kept": len(kept_ids),
            "rejected_large_components": int(count - len(kept_ids)),
            "notched_bins": int(keep.sum()),
            "notched_bins_pct": round(float(keep.mean() * 100), 4),
            "max_peak_excess": (
                round(float(excess[keep].max()), 3) if keep.any() else 0.0
            ),
            "peaks_top": peaks[:12],
        },
    }


def isolated_peak_notch(
    channel: np.ndarray,
    local_win: int = 21,
    peak_tau: float = 1.2,
    max_support: int = 80,
    low_cut: int = 24,
    feather: float = 1.5,
    workers: int | None = None,
) -> tuple[np.ndarray, dict]:
    diagnosis = find_isolated_peaks(
        channel,
        local_win,
        peak_tau,
        max_support,
        low_cut,
        workers=workers,
    )
    keep = diagnosis["keep"]
    excess = diagnosis["excess"]
    spectrum = diagnosis["f"]
    amplitude = diagnosis["amp"]
    pad = diagnosis["pad"]
    height, width = diagnosis["shape"]
    target = np.where(keep, np.maximum(excess, 0.0), 0.0)
    scale = np.exp(-ndimage.gaussian_filter(target, feather))
    output = np.real(
        np.fft.ifft2(np.fft.ifftshift(spectrum * scale))
    )[pad : pad + height, pad : pad + width]
    stats = dict(diagnosis["stats"])
    stats["energy_removed_pct"] = round(
        float(
            1.0
            - ((amplitude * scale) ** 2).sum()
            / max((amplitude**2).sum(), 1e-9)
        )
        * 100.0,
        5,
    )
    return output, stats


def notch_image(
    rgb: np.ndarray,
    params: dict | None = None,
    channels: tuple[str, ...] = ("L",),
    workers: int | None = None,
) -> tuple[np.ndarray, dict]:
    parameters = dict(DEFAULT_PARAMS, **(params or {}))
    lab = rgb_to_lab(rgb)
    stats = {}
    for name, index in (("L", 0), ("a", 1), ("b", 2)):
        if name in channels:
            lab[..., index], stats[name] = isolated_peak_notch(
                lab[..., index],
                workers=workers,
                **parameters,
            )
    return lab_to_rgb8(lab), {
        "params": parameters,
        "channels": list(channels),
        "per_channel": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attenuate isolated periodic lattice peaks."
    )
    parser.add_argument("image")
    parser.add_argument("out_dir")
    parser.add_argument("--tau", type=float, default=DEFAULT_PARAMS["peak_tau"])
    parser.add_argument("--chroma", action="store_true")
    parser.add_argument("--suffix", default="notch")
    arguments = parser.parse_args()
    source = Path(arguments.image)
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = ("L", "a", "b") if arguments.chroma else ("L",)
    output, stats = notch_image(
        load_rgb(source),
        {"peak_tau": arguments.tau},
        channels,
    )
    output_path = output_dir / f"{source.stem}_{arguments.suffix}.png"
    Image.fromarray(output).save(output_path)
    write_json(
        output_path.with_suffix(".json"),
        {
            "stage": "notch",
            "grade": "deliverable",
            "source": str(source),
            "source_sha256": sha256_of(source),
            "output": str(output_path),
            "pixels": list(Image.fromarray(output).size),
            **stats,
            "time": now_iso(),
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
