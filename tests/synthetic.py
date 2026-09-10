"""Deterministic synthetic inputs used by the public test suite."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color as skcolor

HEIGHT, WIDTH = 384, 512


def _lab_to_rgb8(
    lightness: np.ndarray,
    a: float = 12.0,
    b: float = 18.0,
) -> np.ndarray:
    lab = np.stack(
        [
            np.clip(lightness, 0, 100),
            np.full_like(lightness, a),
            np.full_like(lightness, b),
        ],
        axis=-1,
    )
    return np.clip(
        np.rint(skcolor.lab2rgb(lab) * 255),
        0,
        255,
    ).astype(np.uint8)


def base_lightness(seed: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    lightness = 62 + 12 * (xx / WIDTH) - 8 * (yy / HEIGHT)
    stripes = 6 * np.sin((xx + 0.6 * yy) * 2 * np.pi / 14)
    mask = np.zeros_like(lightness)
    mask[HEIGHT // 2 :, WIDTH // 2 :] = 1.0
    mask = ndimage.gaussian_filter(mask, 6)
    rng = np.random.default_rng(seed)
    return (
        lightness
        + stripes * mask
        + ndimage.gaussian_filter(
            rng.normal(0, 1.2, lightness.shape),
            0.8,
        )
    )


def make_clean() -> np.ndarray:
    return _lab_to_rgb8(base_lightness())


def make_lattice(amplitude: float = 0.35) -> np.ndarray:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    lattice = amplitude * (((xx + yy) % 2) * 2 - 1)
    lattice += 0.8 * amplitude * np.cos(2 * np.pi * xx / 4)
    return _lab_to_rgb8(base_lightness() + lattice)


def make_granule(
    standard_deviation: float = 0.9,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    granules = np.zeros((HEIGHT, WIDTH))
    count = int(HEIGHT * WIDTH / 40)
    centers_y = rng.integers(0, HEIGHT, count)
    centers_x = rng.integers(0, WIDTH, count)
    radii = rng.uniform(2.0, 4.0, count)
    amplitudes = rng.choice([-1.0, 1.0], count)
    for y0, x0, radius, amplitude in zip(
        centers_y,
        centers_x,
        radii,
        amplitudes,
        strict=True,
    ):
        y1 = max(0, int(y0 - radius - 1))
        y2 = min(HEIGHT, int(y0 + radius + 2))
        x1 = max(0, int(x0 - radius - 1))
        x2 = min(WIDTH, int(x0 + radius + 2))
        granules[y1:y2, x1:x2] += amplitude * (
            np.hypot(
                yy[y1:y2, x1:x2] - y0,
                xx[y1:y2, x1:x2] - x0,
            )
            <= radius
        )
    granules = ndimage.gaussian_filter(granules, 0.7)
    granules = granules / granules.std() * standard_deviation
    return _lab_to_rgb8(base_lightness() + granules)


def make_textured(
    uniform: bool,
    seed: int = 5,
    amplitude: float = 2.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    base = 55 + ndimage.gaussian_filter(
        rng.normal(0, 1, (HEIGHT, WIDTH)),
        5,
    ) * 60
    components = np.zeros((HEIGHT, WIDTH))
    count = int(HEIGHT * WIDTH / 45)
    centers_y = rng.integers(0, HEIGHT, count)
    centers_x = rng.integers(0, WIDTH, count)
    radii = (
        np.full(count, 3.0)
        if uniform
        else np.exp(rng.uniform(np.log(1.0), np.log(9.0), count))
    )
    aspects = (
        np.ones(count) if uniform else rng.uniform(1.0, 3.5, count)
    )
    angles = rng.uniform(0, np.pi, count)
    signs = rng.choice([-1.0, 1.0], count)
    for y0, x0, radius, aspect, angle, sign in zip(
        centers_y,
        centers_x,
        radii,
        aspects,
        angles,
        signs,
        strict=True,
    ):
        extent = radius * aspect
        y1 = max(0, int(y0 - extent - 1))
        y2 = min(HEIGHT, int(y0 + extent + 2))
        x1 = max(0, int(x0 - extent - 1))
        x2 = min(WIDTH, int(x0 + extent + 2))
        dy = yy[y1:y2, x1:x2] - y0
        dx = xx[y1:y2, x1:x2] - x0
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        components[y1:y2, x1:x2] += sign * (
            (u / (radius * aspect)) ** 2 + (v / radius) ** 2 <= 1
        )
    components = ndimage.gaussian_filter(components, 0.7)
    components = components / components.std() * amplitude
    return _lab_to_rgb8(base + components)
