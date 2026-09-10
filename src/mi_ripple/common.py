"""Shared numerical, image-I/O, provenance, and inspection-board helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import color as skcolor

BAND_SIGMAS = (1.0, 3.0)
RESIDUAL_GAIN = 6.0


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an opaque RGB image as float64 values in [0, 1]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return skcolor.rgb2lab(rgb)


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    lab = lab.copy()
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return np.clip(np.rint(skcolor.lab2rgb(lab) * 255), 0, 255).astype(np.uint8)


def rgb_to_L(rgb: np.ndarray) -> np.ndarray:
    return skcolor.rgb2lab(rgb)[..., 0]


def bandpass(
    channel: np.ndarray,
    sigmas: tuple[float, float] = BAND_SIGMAS,
) -> np.ndarray:
    """Difference-of-Gaussians band used for approximately 3–8 px texture."""
    return ndimage.gaussian_filter(channel, sigmas[0]) - ndimage.gaussian_filter(
        channel, sigmas[1]
    )


def smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def structure_signals(
    lightness: np.ndarray,
    sig_grad: float = 1.5,
    sig_int: float = 5.0,
) -> dict[str, np.ndarray]:
    """Return edge magnitude, coherence, tangent direction, and texture density."""
    smoothed = ndimage.gaussian_filter(lightness, sig_grad)
    gy, gx = np.gradient(smoothed)
    gmag = ndimage.gaussian_filter(np.hypot(gx, gy), 2.0)
    jxx = ndimage.gaussian_filter(gx * gx, sig_int)
    jyy = ndimage.gaussian_filter(gy * gy, sig_int)
    jxy = ndimage.gaussian_filter(gx * gy, sig_int)
    trace = jxx + jyy + 1e-6
    determinant = jxx * jyy - jxy * jxy
    discriminant = np.sqrt(np.maximum(trace * trace / 4 - determinant, 0))
    lambda1 = trace / 2 + discriminant
    lambda2 = trace / 2 - discriminant
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)) ** 2
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    highpass = lightness - ndimage.gaussian_filter(lightness, 2.0)
    highpass_energy = np.sqrt(ndimage.gaussian_filter(highpass * highpass, 6.0))
    return {
        "gmag": gmag,
        "coh": coherence,
        "theta": theta,
        "hp_energy": highpass_energy,
    }


def along_orientation_mean(
    band: np.ndarray,
    theta: np.ndarray,
    half_len: int = 10,
    sigma: float = 5.0,
) -> np.ndarray:
    """Average along the estimated local tangent direction."""
    height, width = band.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = np.cos(theta), np.sin(theta)
    accumulator = np.zeros_like(band)
    weight_sum = 0.0
    for offset in range(-half_len, half_len + 1):
        weight = np.exp(-0.5 * (offset / sigma) ** 2)
        accumulator += weight * ndimage.map_coordinates(
            band,
            [yy + offset * dy, xx + offset * dx],
            order=1,
            mode="reflect",
        )
        weight_sum += weight
    return accumulator / weight_sum


def window_damage(
    before: np.ndarray,
    after: np.ndarray,
    box: tuple[int, int, int, int],
) -> dict:
    """Measure aligned changes inside one x1, y1, x2, y2 window."""
    x1, y1, x2, y2 = box
    before_window = before[y1:y2, x1:x2]
    after_window = after[y1:y2, x1:x2]

    def highpass(array: np.ndarray) -> np.ndarray:
        return array - ndimage.gaussian_filter(array, 1.5)

    difference = after_window - before_window
    before_hf_std = float(highpass(before_window).std())
    after_hf_std = float(highpass(after_window).std())
    return {
        "hf_std_before": round(before_hf_std, 4),
        "hf_std_after": round(after_hf_std, 4),
        "hf_std_ratio": round(after_hf_std / max(before_hf_std, 1e-9), 3),
        "band_std_before": round(float(bandpass(before_window).std()), 4),
        "band_std_after": round(float(bandpass(after_window).std()), 4),
        "diff_std": round(float(difference.std()), 4),
        "diff_abs_max": round(float(np.abs(difference).max()), 4),
    }


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def write_json(path: str | Path, value: dict) -> Path:
    output = Path(path)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        (
            "/System/Library/Fonts/STHeiti Medium.ttc"
            if bold
            else "/System/Library/Fonts/STHeiti Light.ttc"
        ),
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size, index=1 if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def to_img(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def residual_tile(
    difference: np.ndarray,
    box: tuple[int, int, int, int],
    gain: float = RESIDUAL_GAIN,
) -> Image.Image:
    x1, y1, x2, y2 = box
    return to_img(128 + difference[y1:y2, x1:x2] * gain * 2.55)


def crop_tile(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> Image.Image:
    x1, y1, x2, y2 = box
    return to_img(rgb[y1:y2, x1:x2] * 255)


def build_board(
    title: str,
    subtitle: str,
    rows: list[dict],
    tile: int = 480,
    out: str | Path | None = None,
) -> Image.Image:
    """Build a nearest-neighbor inspection board without hiding small artifacts."""
    gap, margin, header, label_height = 16, 40, 120, 64
    columns = max(len(row["tiles"]) for row in rows) if rows else 1
    width = margin * 2 + columns * tile + (columns - 1) * gap
    height = header + len(rows) * (tile + label_height + gap) + margin
    board = Image.new("RGB", (width, height), "#F3F4F6")
    drawing = ImageDraw.Draw(board)
    drawing.text((margin, 30), title, font=font(36, True), fill="#171A1F")
    drawing.text((margin, 80), subtitle, font=font(20), fill="#555E68")
    y = header
    for row in rows:
        for index, image in enumerate(row["tiles"]):
            x = margin + index * (tile + gap)
            drawing.rounded_rectangle(
                (x - 4, y - 4, x + tile + 4, y + tile + 4),
                radius=10,
                fill="#FFFFFF",
                outline="#D7DAE0",
                width=2,
            )
            board.paste(
                image.resize((tile, tile), Image.Resampling.NEAREST),
                (x, y),
            )
        drawing.text(
            (margin, y + tile + 10),
            row["label"],
            font=font(22, True),
            fill=row.get("color", "#20252B"),
        )
        y += tile + label_height + gap
    if out is not None:
        board.save(out, optimize=True)
    return board
