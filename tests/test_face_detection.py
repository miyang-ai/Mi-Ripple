from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mi_ripple import pipeline
from mi_ripple.reference import detect_faces
from tests.synthetic import make_clean, make_lattice


class FakeCv2Error(Exception):
    pass


class FakeCascade:
    def __init__(self, path: str, detections=(), empty=False, fail=False) -> None:
        self.path = path
        self.detections = detections
        self.is_empty = empty
        self.fail = fail

    def empty(self) -> bool:
        return self.is_empty

    def detectMultiScale(self, image, **kwargs):  # noqa: N802
        if self.fail:
            raise FakeCv2Error("detectMultiScale failed")
        if "frontalface_default" in self.path:
            return list(self.detections)
        return []


def fake_cv2(cascade=None, data=True) -> types.ModuleType:
    """A minimal importable cv2; leave out pieces to model broken builds."""
    module = types.ModuleType("cv2")
    module.error = FakeCv2Error
    module.COLOR_RGB2GRAY = 7
    module.cvtColor = lambda image, code: image[..., 0]
    module.equalizeHist = lambda gray: gray
    module.flip = lambda image, code: image[:, ::-1]
    if data:
        module.data = types.SimpleNamespace(haarcascades="cascades/")
    if cascade is not None:
        module.CascadeClassifier = cascade
    return module


def test_missing_opencv_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert detect_faces(make_clean() / 255.0) == ([], "cv2_unavailable")


@pytest.mark.parametrize(
    ("module", "status"),
    [
        (fake_cv2(), "cv2_error:AttributeError"),
        (fake_cv2(FakeCascade, data=False), "cv2_error:AttributeError"),
        (
            fake_cv2(lambda path: FakeCascade(path, empty=True)),
            "cv2_error:FileNotFoundError",
        ),
        (
            fake_cv2(lambda path: FakeCascade(path, fail=True)),
            "cv2_error:FakeCv2Error",
        ),
    ],
    ids=["no-cascade-api", "no-cascade-data", "cascades-empty", "runtime-error"],
)
def test_unusable_opencv_degrades_to_cv2_error(
    monkeypatch: pytest.MonkeyPatch,
    module: types.ModuleType,
    status: str,
) -> None:
    monkeypatch.setitem(sys.modules, "cv2", module)
    assert detect_faces(make_clean() / 255.0) == ([], status)


def test_failing_opencv_import_is_cv2_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "cv2"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise ImportError("DLL load failed")\n',
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert detect_faces(make_clean() / 255.0) == ([], "cv2_error:ImportError")


def test_working_opencv_still_returns_merged_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detections = [(10, 20, 30, 40), (20, 30, 30, 40), (200, 200, 30, 30)]
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        fake_cv2(lambda path: FakeCascade(path, detections=detections)),
    )
    boxes, status = detect_faces(make_clean() / 255.0)
    assert status == "haar_cascade"
    assert boxes == [(10, 20, 50, 70), (200, 200, 230, 230)]
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2(FakeCascade))
    assert detect_faces(make_clean() / 255.0) == ([], "haar_cascade_no_face")


def test_incompatible_opencv_does_not_stop_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "lattice.png"
    Image.fromarray(make_lattice()).save(source)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2())
    result = pipeline.run(source, tmp_path / "output")
    assert result["outcome"] == "delivered"
    assert [step["action"] for step in result["steps"]] == [
        "diagnose",
        "notch",
        "verify",
        "finish",
    ]
    assert result["verify"]["passed"]
    assert np.asarray(Image.open(result["final"])).shape == make_lattice().shape
