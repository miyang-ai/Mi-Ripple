"""Reference-only cleaning before optional image regeneration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .common import (
    along_orientation_mean,
    lab_to_rgb8,
    load_rgb,
    now_iso,
    rgb_to_lab,
    sha256_of,
    smoothstep,
    structure_signals,
    write_json,
)
from .notch import DEFAULT_PARAMS as NOTCH_PARAMS
from .notch import isolated_peak_notch
from .spatial import PROFILES

PARAMS = {
    "notch": dict(NOTCH_PARAMS),
    "iso": {**PROFILES["loose"], "strength": 1.0},
    "dir": {
        "coh": (0.15, 0.45),
        "edge_guard": (3.0, 6.0),
        "half_len": 10,
        "line_sigma": 5.0,
        "strength": 0.85,
    },
    "face_guard": {"feather": 10.0, "expand": 0.25},
}


def detect_faces(
    rgb: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], str]:
    """Detect optional face-protection boxes using OpenCV Haar cascades."""
    try:
        import cv2
    except ImportError:
        return [], "cv2_unavailable"
    gray = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    gray = cv2.equalizeHist(gray)
    height, width = gray.shape
    minimum_side = max(24, int(min(height, width) * 0.06))
    boxes: list[tuple[int, int, int, int]] = []
    cascade_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    )
    for name in cascade_names:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        flips = (False, True) if "profile" in name else (False,)
        for flip in flips:
            input_gray = cv2.flip(gray, 1) if flip else gray
            detected = cascade.detectMultiScale(
                input_gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(minimum_side, minimum_side),
            )
            for x, y, box_width, box_height in detected:
                if flip:
                    x = width - x - box_width
                boxes.append(
                    (
                        int(x),
                        int(y),
                        int(x + box_width),
                        int(y + box_height),
                    )
                )
    merged: list[tuple[int, int, int, int]] = []
    boxes.sort(
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        reverse=True,
    )
    for box in boxes:
        for index, current in enumerate(merged):
            intersection = (
                max(box[0], current[0]),
                max(box[1], current[1]),
                min(box[2], current[2]),
                min(box[3], current[3]),
            )
            if (
                intersection[0] < intersection[2]
                and intersection[1] < intersection[3]
            ):
                merged[index] = (
                    min(box[0], current[0]),
                    min(box[1], current[1]),
                    max(box[2], current[2]),
                    max(box[3], current[3]),
                )
                break
        else:
            merged.append(box)
    return merged, "haar_cascade" if merged else "haar_cascade_no_face"


def face_guard_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    feather: float,
    expand: float,
) -> np.ndarray:
    guard = np.zeros(shape)
    height, width = shape
    for x1, y1, x2, y2 in boxes:
        box_width, box_height = x2 - x1, y2 - y1
        guard_x1 = max(0, int(x1 - box_width * expand))
        guard_y1 = max(0, int(y1 - box_height * expand))
        guard_x2 = min(width, int(x2 + box_width * expand))
        guard_y2 = min(
            height,
            int(y2 + box_height * expand * 1.4),
        )
        guard[guard_y1:guard_y2, guard_x1:guard_x2] = 1.0
    return ndimage.gaussian_filter(guard, feather) if boxes else guard


def reference_grade_clean(
    rgb: np.ndarray,
    params: dict | None = None,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Create an aggressively cleaned reference that is not a deliverable."""
    parameters = params or PARAMS
    lab = rgb_to_lab(rgb)
    notched_lightness, notch_stats = isolated_peak_notch(
        lab[..., 0],
        **parameters["notch"],
    )
    lab[..., 0] = np.clip(notched_lightness, 0, 100)
    lightness = lab[..., 0]
    signals = structure_signals(lightness, sig_int=4.0)
    iso_parameters = parameters["iso"]
    directional_parameters = parameters["dir"]
    iso_weight = ndimage.gaussian_filter(
        (
            1
            - smoothstep(signals["coh"], *iso_parameters["coh"])
        )
        * (
            1
            - smoothstep(signals["gmag"], *iso_parameters["edge"])
        )
        * (
            1
            - smoothstep(
                signals["hp_energy"],
                *iso_parameters["dense"],
            )
        ),
        3.0,
    )
    directional_weight = ndimage.gaussian_filter(
        smoothstep(signals["coh"], *directional_parameters["coh"])
        * (
            1
            - smoothstep(
                signals["gmag"],
                *directional_parameters["edge_guard"],
            )
        ),
        2.0,
    ) * directional_parameters["strength"]
    if face_boxes:
        guard = face_guard_mask(
            lightness.shape,
            face_boxes,
            parameters["face_guard"]["feather"],
            parameters["face_guard"]["expand"],
        )
        iso_weight *= 1 - guard
        directional_weight *= 1 - guard
    output = lab.copy()
    for index in range(3):
        channel = lab[..., index]
        band = ndimage.gaussian_filter(
            channel,
            1.0,
        ) - ndimage.gaussian_filter(channel, 3.0)
        reduced = channel - iso_parameters["strength"] * iso_weight * band
        along = along_orientation_mean(
            reduced,
            signals["theta"],
            directional_parameters["half_len"],
            directional_parameters["line_sigma"],
        )
        output[..., index] = (
            reduced * (1 - directional_weight)
            + along * directional_weight
        )
    stats = {
        "notch": notch_stats,
        "area_pct_iso_gt_0p5": round(
            float((iso_weight > 0.5).mean() * 100),
            2,
        ),
        "area_pct_dir_gt_0p5": round(
            float((directional_weight > 0.5).mean() * 100),
            2,
        ),
        "face_boxes": [list(box) for box in (face_boxes or [])],
    }
    return lab_to_rgb8(output), {
        "w_iso": iso_weight,
        "w_dir": directional_weight,
    }, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reference-only cleaned image for regeneration."
    )
    parser.add_argument("image")
    parser.add_argument("out_dir")
    parser.add_argument("--no-face-guard", action="store_true")
    arguments = parser.parse_args()
    source = Path(arguments.image)
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(source)
    if arguments.no_face_guard:
        boxes, detection_method = [], "disabled"
    else:
        boxes, detection_method = detect_faces(rgb)
    output, masks, stats = reference_grade_clean(rgb, PARAMS, boxes)
    output_path = output_dir / f"{source.stem}_refclean.png"
    Image.fromarray(output).save(output_path)
    Image.fromarray(
        (np.clip(masks["w_iso"], 0, 1) * 255).astype(np.uint8)
    ).save(output_dir / f"{source.stem}_refclean_w_iso.png")
    write_json(
        output_path.with_suffix(".json"),
        {
            "stage": "reference_clean",
            "grade": "reference_only",
            "not_a_deliverable": True,
            "source": str(source),
            "source_sha256": sha256_of(source),
            "output": str(output_path),
            "params": {
                key: value if key != "notch" else dict(value)
                for key, value in PARAMS.items()
            },
            "face_detection": detection_method,
            **stats,
            "time": now_iso(),
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
