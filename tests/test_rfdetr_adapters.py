import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from tote_vision.adapters.image_resolver import LocalImageResolver
from tote_vision.adapters.rfdetr import (
    RfdetrInferenceSession,
    RfdetrLayoutDetector,
    RfdetrToteDetector,
)
from tote_vision.core.models import InspectionRequest, ToteLayout


class FakeRfdetrModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image_path: str, threshold: float):
        self.calls += 1
        return SimpleNamespace(
            xyxy=[
                [5, 5, 95, 95],
                [4, 5, 50, 50],  # minor spill is accepted, then clipped to the tote
                [50, 5, 95, 50],
                [5, 50, 50, 95],
                [50, 50, 95, 95],
                [5, 5, 50, 50],
                [0, 0, 10, 10],
            ],
            confidence=[0.1596, 0.94, 0.93, 0.92, 0.91, 0.70, 0.95],
            class_id=[0, 1, 1, 1, 1, 1, 1],
            mask=None,
        )


class FakeToteOnlyModel:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def predict(self, image_path: str, threshold: float):
        return SimpleNamespace(
            xyxy=[[5, 5, 95, 95]],
            confidence=[self.confidence],
            class_id=[0],
            mask=None,
        )


def test_cells_are_constrained_deduplicated_and_inferred_without_expected_layout(
    tmp_path: Path,
) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    model = FakeRfdetrModel()
    session = RfdetrInferenceSession(
        checkpoint_path=Path("rf-detr-seg-small-totes.pth"),
        image_resolver=LocalImageResolver(tmp_path),
        confidence_threshold=0.5,
        model=model,
    )
    request = InspectionRequest(
        inspection_id="INS-1",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    tote = asyncio.run(RfdetrToteDetector(session, tote_class_id=0).detect(request))
    layout = asyncio.run(RfdetrLayoutDetector(session, cell_class_id=1).detect(request, tote))

    assert model.calls == 1
    assert tote.detected
    assert tote.confidence == 0.1596
    assert layout.layout is ToteLayout.FOUR_CELL
    assert [cell.cell_id for cell in layout.cells] == ["A", "B", "C", "D"]
    assert all(
        5 <= x <= 95 and 5 <= y <= 95
        for cell in layout.cells
        for x, y in cell.polygon
    )


def test_open_layout_uses_the_tote_polygon_for_the_cell(tmp_path: Path) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    model = FakeRfdetrModel()
    raw = model.predict("unused", 0.5)
    raw.xyxy = raw.xyxy[:2]
    raw.confidence = raw.confidence[:2]
    raw.class_id = raw.class_id[:2]
    model.predict = lambda image_path, threshold: raw
    session = RfdetrInferenceSession(
        checkpoint_path=Path("rf-detr-seg-small-totes.pth"),
        image_resolver=LocalImageResolver(tmp_path),
        confidence_threshold=0.05,
        model=model,
    )
    request = InspectionRequest(
        inspection_id="INS-OPEN",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    tote = asyncio.run(RfdetrToteDetector(session, tote_class_id=0).detect(request))
    layout = asyncio.run(RfdetrLayoutDetector(session, cell_class_id=1).detect(request, tote))

    assert layout.layout is ToteLayout.OPEN
    assert layout.cells[0].polygon == tote.polygon


def test_open_tote_fallback_uses_high_confidence_tote_when_no_cells_detected(
    tmp_path: Path,
) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    session = RfdetrInferenceSession(
        checkpoint_path=Path("rf-detr-seg-small-totes.pth"),
        image_resolver=LocalImageResolver(tmp_path),
        confidence_threshold=0.05,
        model=FakeToteOnlyModel(confidence=0.82),
    )
    request = InspectionRequest(
        inspection_id="INS-FALLBACK",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    tote = asyncio.run(RfdetrToteDetector(session, tote_class_id=0).detect(request))
    layout = asyncio.run(RfdetrLayoutDetector(session, cell_class_id=1).detect(request, tote))

    assert tote.detected
    assert layout.layout is ToteLayout.OPEN
    assert layout.confidence == tote.confidence
    assert layout.cells[0].cell_id == "A"
    assert layout.cells[0].polygon == tote.polygon


def test_empty_instance_mask_falls_back_to_box_polygon(tmp_path: Path) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    session = RfdetrInferenceSession(
        checkpoint_path=Path("rf-detr-seg-small-totes.pth"),
        image_resolver=LocalImageResolver(tmp_path),
        confidence_threshold=0.05,
        model=SimpleNamespace(
            predict=lambda image_path, threshold: SimpleNamespace(
                xyxy=[[5, 5, 95, 95]],
                confidence=[0.82],
                class_id=[0],
                mask=[[[0, 0], [0, 0]]],
            )
        ),
    )
    request = InspectionRequest(
        inspection_id="INS-EMPTY-MASK",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    tote = asyncio.run(RfdetrToteDetector(session, tote_class_id=0).detect(request))

    assert tote.detected
    assert tote.polygon == ((5.0, 5.0), (95.0, 5.0), (95.0, 95.0), (5.0, 95.0))


def test_open_tote_fallback_rejects_weak_tote_when_no_cells_detected(
    tmp_path: Path,
) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")
    session = RfdetrInferenceSession(
        checkpoint_path=Path("rf-detr-seg-small-totes.pth"),
        image_resolver=LocalImageResolver(tmp_path),
        confidence_threshold=0.05,
        model=FakeToteOnlyModel(confidence=0.62),
    )
    request = InspectionRequest(
        inspection_id="INS-NO-FALLBACK",
        tote_id="TOTE-1",
        image_uri="/artifacts/tote.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    tote = asyncio.run(
        RfdetrToteDetector(session, tote_class_id=0, confidence_threshold=0.1).detect(
            request
        )
    )
    layout = asyncio.run(RfdetrLayoutDetector(session, cell_class_id=1).detect(request, tote))

    assert tote.detected
    assert layout.layout is ToteLayout.UNKNOWN
    assert layout.cells == ()
