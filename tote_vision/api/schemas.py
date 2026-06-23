from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tote_vision.core.models import (
    CoordinateSpace,
    InspectionRequest,
    InspectionResult,
    ToteLayout,
)


class EmptyToteInspectionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    inspection_id: str = Field(
        default_factory=lambda: f"VISION-{uuid4().hex[:12].upper()}",
        alias="inspectionId",
        min_length=1,
        max_length=128,
    )
    tote_id: str = Field(default="UNASSIGNED", alias="toteId", min_length=1, max_length=128)
    image_uri: str = Field(alias="imageUri", min_length=1, max_length=2048)
    station_id: str = Field(default="UNASSIGNED", alias="stationId", min_length=1, max_length=128)
    camera_id: str = Field(default="UNASSIGNED", alias="cameraId", min_length=1, max_length=128)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC), alias="capturedAt")
    classifier_method: str | None = Field(default=None, alias="classifierMethod")

    def to_domain(self) -> InspectionRequest:
        return InspectionRequest(**self.model_dump(by_alias=False))


class GeometryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    valid: bool
    confidence: float
    issues: list[str]


class CellResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cell_id: str = Field(alias="cellId")
    result: str
    empty_probability: float = Field(alias="emptyProbability")
    non_empty_probability: float = Field(alias="nonEmptyProbability")
    uncertain_probability: float = Field(alias="uncertainProbability")
    crop_uri: str | None = Field(alias="cropUri")
    mask_uri: str | None = Field(alias="maskUri")
    overlay_uri: str | None = Field(alias="overlayUri")
    polygon: list[list[float]] | None


class EmptyToteInspectionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    inspection_id: str = Field(alias="inspectionId")
    tote_id: str = Field(alias="toteId")
    result: str
    reason_code: str = Field(alias="reasonCode")
    observed_layout: ToteLayout | None = Field(alias="observedLayout")
    layout_confidence: float | None = Field(alias="layoutConfidence")
    geometry: GeometryResponse | None
    cells: list[CellResponse]
    method_results: dict[str, list[CellResponse]] | None = Field(
        default=None, alias="methodResults"
    )
    detected_cells: list[dict[str, object]] = Field(alias="detectedCells")
    model_versions: dict[str, str] = Field(alias="modelVersions")
    decision_version: str = Field(alias="decisionVersion")
    completed_at: datetime = Field(alias="completedAt")
    image_uri: str | None = Field(alias="imageUri")
    tote_polygon: list[list[float]] | None = Field(alias="totePolygon")
    coordinate_space: CoordinateSpace = Field(alias="coordinateSpace")
    metadata: dict[str, object]

    @classmethod
    def from_domain(cls, result: InspectionResult) -> EmptyToteInspectionResponse:
        geometry = None
        if result.geometry:
            geometry = GeometryResponse(
                valid=result.geometry.valid,
                confidence=result.geometry.confidence,
                issues=list(result.geometry.issues),
            )
        method_results = None
        if result.method_results is not None:
            method_results = {
                method: _build_cell_responses(cells)
                for method, cells in result.method_results.items()
            }
        return cls(
            inspectionId=result.inspection_id,
            toteId=result.tote_id,
            result=result.decision.value,
            reasonCode=result.reason_code.value,
            observedLayout=result.observed_layout,
            layoutConfidence=result.layout_confidence,
            geometry=geometry,
            cells=_build_cell_responses(result.cells),
            methodResults=method_results,
            detectedCells=[
                {
                    "cellId": cell.cell_id,
                    "polygon": [list(point) for point in cell.polygon],
                }
                for cell in result.detected_cells
            ],
            modelVersions=result.model_versions,
            decisionVersion=result.decision_version,
            completedAt=result.completed_at,
            imageUri=result.image_uri,
            totePolygon=(
                [list(point) for point in result.tote_polygon]
                if result.tote_polygon
                else None
            ),
            coordinateSpace=result.coordinate_space,
            metadata=result.metadata,
        )


def _build_cell_responses(cells: tuple) -> list[CellResponse]:
    return [
        CellResponse(
            cellId=cell.cell_id,
            result=cell.classification.value,
            emptyProbability=cell.empty_probability,
            nonEmptyProbability=cell.non_empty_probability,
            uncertainProbability=cell.uncertain_probability,
            cropUri=cell.crop_uri,
            maskUri=cell.mask_uri,
            overlayUri=cell.overlay_uri,
            polygon=[list(point) for point in cell.polygon] if cell.polygon else None,
        )
        for cell in cells
    ]
