import asyncio
from datetime import UTC, datetime
from pathlib import Path

from tote_vision.adapters.dinov2 import CellExtractionError, Dinov2CellClassifier
from tote_vision.adapters.image_resolver import LocalImageResolver
from tote_vision.core.models import (
    CellClassification,
    CellGeometry,
    InspectionRequest,
)


class FakeDinov2Runtime:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities

    def predict(self, image_path: Path, polygon):
        return self._probabilities


def test_non_empty_threshold_routes_cell_to_non_empty(tmp_path: Path) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    classifier = Dinov2CellClassifier(
        model_path=Path("dinov2-vits14"),
        classifier_path=Path("empty-cell-head.safetensors"),
        image_resolver=LocalImageResolver(tmp_path),
        device="cpu",
        empty_threshold=0.9,
        non_empty_threshold=0.7,
        runtime=FakeDinov2Runtime([0.2, 0.8]),
    )
    request = InspectionRequest(
        inspection_id="INS-1",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    result = asyncio.run(
        classifier.classify(
            request,
            CellGeometry(cell_id="A", polygon=((0, 0), (10, 0), (10, 10), (0, 10))),
        )
    )

    assert result.classification is CellClassification.NON_EMPTY
    assert result.non_empty_probability == 0.8


def test_low_confidence_prediction_routes_to_uncertain(tmp_path: Path) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    classifier = Dinov2CellClassifier(
        model_path=Path("dinov2-vits14"),
        classifier_path=Path("empty-cell-head.safetensors"),
        image_resolver=LocalImageResolver(tmp_path),
        device="cpu",
        empty_threshold=0.9,
        non_empty_threshold=0.7,
        runtime=FakeDinov2Runtime([0.55, 0.45]),
    )
    request = InspectionRequest(
        inspection_id="INS-2",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    result = asyncio.run(
        classifier.classify(
            request,
            CellGeometry(cell_id="A", polygon=((0, 0), (10, 0), (10, 10), (0, 10))),
        )
    )

    assert result.classification is CellClassification.UNCERTAIN


def test_failed_cell_extraction_routes_to_bad_capture(tmp_path: Path) -> None:
    class FailedRuntime:
        def predict(self, image_path: Path, polygon):
            raise CellExtractionError("empty crop")

    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    classifier = Dinov2CellClassifier(
        model_path=Path("dinov2-vits14"),
        classifier_path=Path("empty-cell-head.safetensors"),
        image_resolver=LocalImageResolver(tmp_path),
        device="cpu",
        empty_threshold=0.9,
        non_empty_threshold=0.7,
        runtime=FailedRuntime(),
    )
    request = InspectionRequest(
        inspection_id="INS-3",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    result = asyncio.run(
        classifier.classify(
            request,
            CellGeometry(cell_id="A", polygon=((0, 0), (10, 0), (10, 10), (0, 10))),
        )
    )

    assert result.classification is CellClassification.BAD_CAPTURE
