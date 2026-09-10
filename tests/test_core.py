from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from mirage import pipeline
from mirage.common import rgb_to_L, window_damage
from mirage.diagnosis import diagnose, flag_granule, window_features
from mirage.notch import find_isolated_peaks, median_baseline, notch_image
from mirage.regeneration import build_prompt, regenerate, request_size
from mirage.scale_index import scale_index
from mirage.scale_index import serializable as scale_serializable
from mirage.spatial import iso_clean
from mirage.verify import verify
from tests.synthetic import (
    HEIGHT,
    WIDTH,
    make_clean,
    make_granule,
    make_lattice,
    make_textured,
)


def save_image(directory: Path, name: str, array: np.ndarray) -> Path:
    path = directory / name
    Image.fromarray(array).save(path)
    return path


class FakeClient:
    def __init__(
        self,
        size: tuple[int, int] | None = None,
        with_lattice: bool = True,
    ) -> None:
        self.calls: list[dict] = []
        self.size = size
        self.with_lattice = with_lattice
        self.last_meta = {"fake": True}

    def edit(
        self,
        image_bytes: bytes,
        filename: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
    ) -> bytes:
        self.calls.append(
            {
                "filename": filename,
                "prompt": prompt,
                "model": model,
                "size": size,
                "quality": quality,
            }
        )
        width, height = self.size or tuple(
            int(value) for value in size.split("x")
        )
        source = make_lattice() if self.with_lattice else make_clean()
        image = Image.fromarray(source).resize((width, height))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def test_lattice_is_detected_and_reduced() -> None:
    lightness = rgb_to_L(make_lattice() / 255.0)
    before = find_isolated_peaks(lightness)["stats"]
    output, stats = notch_image(make_lattice() / 255.0)
    after = find_isolated_peaks(
        rgb_to_L(output / 255.0)
    )["stats"]
    assert before["max_peak_excess"] >= 2.5
    assert after["max_peak_excess"] < before["max_peak_excess"] - 2.0
    assert stats["per_channel"]["L"]["energy_removed_pct"] < 1.0


def test_clean_image_is_not_lattice_positive() -> None:
    result = find_isolated_peaks(rgb_to_L(make_clean() / 255.0))
    assert result["stats"]["max_peak_excess"] < 2.5


def test_granule_window_is_flagged_but_clean_window_is_not() -> None:
    granule = rgb_to_L(make_granule() / 255.0)
    clean = rgb_to_L(make_clean() / 255.0)
    assert flag_granule(window_features(granule[40:136, 40:136]))
    assert not flag_granule(window_features(clean[40:136, 40:136]))


def test_diagnosis_separates_clean_and_granular_inputs() -> None:
    clean = diagnose(make_clean() / 255.0)
    granule = diagnose(make_granule() / 255.0)
    assert clean["granule"]["level"] == "none"
    assert not clean["flags"]["lattice"]
    assert granule["granule"]["level"] in ("flat", "pervasive")


def test_spatial_reduction_protects_directional_texture() -> None:
    rgb = make_granule() / 255.0
    output, weight, _ = iso_clean(rgb, 1.0, "strict")
    before = rgb_to_L(rgb)
    after = rgb_to_L(output / 255.0)
    flat = window_damage(before, after, (40, 40, 136, 136))
    directional = window_damage(
        before,
        after,
        (WIDTH - 136, HEIGHT - 136, WIDTH - 40, HEIGHT - 40),
    )
    assert flat["band_std_after"] < 0.7 * flat["band_std_before"]
    assert directional["hf_std_ratio"] > 0.85
    assert weight[HEIGHT - 100, WIDTH - 100] < 0.5


def test_verification_detects_structural_damage() -> None:
    rgb = make_clean() / 255.0
    unchanged = verify(
        rgb,
        rgb,
        [[40, 40, 136, 136]],
        [[WIDTH - 136, HEIGHT - 136, WIDTH - 40, HEIGHT - 40]],
    )
    assert unchanged["passed"]
    damaged = rgb.copy()
    damaged[HEIGHT - 136 : HEIGHT - 40, WIDTH - 136 : WIDTH - 40] = (
        ndimage.gaussian_filter(
            damaged[
                HEIGHT - 136 : HEIGHT - 40,
                WIDTH - 136 : WIDTH - 40,
            ],
            3.0,
        )
    )
    result = verify(
        rgb,
        damaged,
        [[40, 40, 136, 136]],
        [[WIDTH - 136, HEIGHT - 136, WIDTH - 40, HEIGHT - 40]],
    )
    assert not result["passed"]
    with pytest.raises(ValueError):
        verify(rgb, rgb[:-1], [], [])


def test_scale_index_separates_uniform_and_varied_components() -> None:
    uniform = scale_index(make_textured(True) / 255.0)
    varied = scale_index(make_textured(False) / 255.0)
    assert uniform["level"] == "structured"
    assert uniform["scale_tile_pct"] > 50
    assert varied["level"] == "none"
    assert varied["scale_tile_pct"] < 2
    json.dumps(scale_serializable(uniform))


def test_median_worker_count_does_not_change_result() -> None:
    rng = np.random.default_rng(11)
    array = rng.normal(0, 1, (64, 37))
    expected = median_baseline(array, 21, workers=1)
    for workers in (2, 3, 5):
        assert np.array_equal(
            median_baseline(array, 21, workers=workers),
            expected,
        )


def test_request_size_and_prompts() -> None:
    assert request_size(1536, 1024) == "1536x1024"
    assert request_size(1011, 638) == "1008x624"
    prompts = build_prompt({"zh": "保留自然纸张纹理。"})
    assert set(prompts) == {"ko", "en", "zh"}
    assert prompts["zh"].endswith("保留自然纸张纹理。")


def test_regeneration_writes_prompt_and_sidecar(tmp_path: Path) -> None:
    reference = save_image(tmp_path, "reference.png", make_granule())
    client = FakeClient()
    output = tmp_path / "regenerated.png"
    sidecar = regenerate(
        reference,
        output,
        client,
        hint={"en": "Keep the balustrade smooth."},
    )
    prompt = json.loads(
        output.with_suffix(".prompt.json").read_text(encoding="utf-8")
    )
    assert client.calls[0]["prompt"] == prompt["prompt_en"]
    assert sidecar["saved_size"] == [WIDTH, HEIGHT]
    assert sidecar["size_verified"]
    assert sidecar["grade"].startswith("candidate")


def test_aspect_mismatch_is_cropped_without_upscaling(
    tmp_path: Path,
) -> None:
    reference = save_image(tmp_path, "reference.png", make_clean())
    sidecar = regenerate(
        reference,
        tmp_path / "regenerated.png",
        FakeClient(size=(768, 512)),
    )
    assert sidecar["center_cropped_to_requested_ratio"]
    width, height = sidecar["saved_size"]
    assert abs(width / height - WIDTH / HEIGHT) < 0.01
    assert height == 512


def test_clean_pipeline_passes_through(tmp_path: Path) -> None:
    source = save_image(tmp_path, "clean.png", make_clean())
    result = pipeline.run(source, tmp_path / "output")
    assert result["outcome"] == "delivered"
    assert result["final_is_source_copy"]
    assert [step["action"] for step in result["steps"]] == [
        "diagnose",
        "finish",
    ]


def test_lattice_pipeline_notches_and_verifies(tmp_path: Path) -> None:
    source = save_image(tmp_path, "lattice.png", make_lattice())
    events: list[dict] = []
    result = pipeline.run(
        source,
        tmp_path / "output",
        on_event=events.append,
    )
    assert [step["action"] for step in result["steps"]] == [
        "diagnose",
        "notch",
        "verify",
        "finish",
    ]
    assert result["verify"]["passed"]
    assert Path(result["final"]).is_file()
    assert events[-1]["type"] == "outcome"
    assert all(
        "result" not in event["step"]
        for event in events
        if event["type"] == "step_start"
    )


def test_structured_pipeline_requires_permission(tmp_path: Path) -> None:
    source = save_image(
        tmp_path,
        "structured.png",
        make_textured(True),
    )
    result = pipeline.run(source, tmp_path / "output")
    assert result["outcome"] == "needs_human_decision"
    assert result["regen_rounds"] == 0


def test_cancel_stops_between_steps(tmp_path: Path) -> None:
    source = save_image(tmp_path, "lattice.png", make_lattice())
    calls = 0

    def should_cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    result = pipeline.run(
        source,
        tmp_path / "output",
        should_cancel=should_cancel,
    )
    assert result["outcome"] == "cancelled"
    assert result["final"] is None
    assert [step["action"] for step in result["steps"]] == ["diagnose"]
    assert (
        tmp_path / "output" / "lattice_restored.json"
    ).is_file()
