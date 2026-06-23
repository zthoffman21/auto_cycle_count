import asyncio
from datetime import UTC, datetime

from tote_vision.application.inspect_empty_tote import InspectEmptyTote
from tote_vision.core.decision import DecisionEngine, DecisionPolicy
from tote_vision.core.geometry import GeometryValidator
from tote_vision.core.models import (
    CellClassification,
    CellGeometry,
    CellResult,
    CoordinateSpace,
    InspectionDecision,
    InspectionRequest,
    LayoutDetection,
    ReasonCode,
    ToteDetection,
    ToteLayout,
)


class _StubToteDetector:
    async def detect(self, request: InspectionRequest) -> ToteDetection:
        return ToteDetection(
            detected=True,
            confidence=1.0,
            polygon=((0, 0), (100, 0), (100, 100), (0, 100)),
            model_name="stub",
            model_version="stub-v1",
            coordinate_space=CoordinateSpace.NORMALIZED_100,
        )


class _StubLayoutDetector:
    async def detect(self, request: InspectionRequest, tote: ToteDetection) -> LayoutDetection:
        return LayoutDetection(
            layout=ToteLayout.OPEN,
            confidence=1.0,
            cells=(CellGeometry(cell_id="A", polygon=((5, 5), (95, 5), (95, 95), (5, 95))),),
            model_name="stub",
            model_version="stub-v1",
        )


class _StubCellClassifier:
    name = "stub"

    async def classify(self, request: InspectionRequest, cell: CellGeometry) -> CellResult:
        return CellResult(
            cell_id=cell.cell_id,
            classification=CellClassification.EMPTY,
            empty_probability=1.0,
            non_empty_probability=0.0,
            uncertain_probability=0.0,
            model_name="stub",
            model_version="stub-v1",
        )


def test_pipeline_returns_auditable_result() -> None:
    inspector = InspectEmptyTote(
        tote_detector=_StubToteDetector(),
        layout_detector=_StubLayoutDetector(),
        cell_classifiers={"stub": _StubCellClassifier()},
        geometry_validator=GeometryValidator(),
        decision_engine=DecisionEngine(DecisionPolicy()),
        decision_version="test-v1",
    )
    request = InspectionRequest(
        inspection_id="INS-1",
        tote_id="TOTE-1",
        image_uri="file:///tmp/test.png",
        station_id="STATION-1",
        camera_id="CAM-1",
        captured_at=datetime.now(UTC),
    )

    result = asyncio.run(inspector.execute(request))

    assert result.decision is InspectionDecision.PASS
    assert result.reason_code is ReasonCode.ALL_CELLS_EMPTY
    assert len(result.cells) == 1
