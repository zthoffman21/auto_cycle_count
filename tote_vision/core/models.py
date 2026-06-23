from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ToteLayout(StrEnum):
    OPEN = "OPEN"
    TWO_CELL = "TWO_CELL"
    FOUR_CELL = "FOUR_CELL"
    UNKNOWN = "UNKNOWN"


class CellClassification(StrEnum):
    EMPTY = "EMPTY"
    NON_EMPTY = "NON_EMPTY"
    UNCERTAIN = "UNCERTAIN"
    BAD_CAPTURE = "BAD_CAPTURE"


class InspectionDecision(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    ERROR = "ERROR"


class ReasonCode(StrEnum):
    ALL_CELLS_EMPTY = "ALL_CELLS_EMPTY"
    NON_EMPTY_CELL_DETECTED = "NON_EMPTY_CELL_DETECTED"
    UNCERTAIN_CELL = "UNCERTAIN_CELL"
    BAD_CAPTURE = "BAD_CAPTURE"
    TOTE_NOT_DETECTED = "TOTE_NOT_DETECTED"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    INFERENCE_ERROR = "INFERENCE_ERROR"


class CoordinateSpace(StrEnum):
    IMAGE_PIXELS = "IMAGE_PIXELS"
    NORMALIZED_100 = "NORMALIZED_100"


Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class InspectionRequest:
    inspection_id: str
    tote_id: str
    image_uri: str
    station_id: str
    camera_id: str
    captured_at: datetime


@dataclass(frozen=True, slots=True)
class ToteDetection:
    detected: bool
    confidence: float
    polygon: Polygon | None
    model_name: str
    model_version: str
    coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_PIXELS


@dataclass(frozen=True, slots=True)
class CellGeometry:
    cell_id: str
    polygon: Polygon
    crop_uri: str | None = None
    mask_uri: str | None = None


@dataclass(frozen=True, slots=True)
class LayoutDetection:
    layout: ToteLayout
    confidence: float
    cells: tuple[CellGeometry, ...]
    model_name: str
    model_version: str


@dataclass(frozen=True, slots=True)
class GeometryValidation:
    valid: bool
    confidence: float
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CellResult:
    cell_id: str
    classification: CellClassification
    empty_probability: float
    non_empty_probability: float
    uncertain_probability: float
    model_name: str
    model_version: str
    polygon: Polygon | None = None
    crop_uri: str | None = None
    mask_uri: str | None = None
    overlay_uri: str | None = None


@dataclass(frozen=True, slots=True)
class InspectionResult:
    inspection_id: str
    tote_id: str
    decision: InspectionDecision
    reason_code: ReasonCode
    observed_layout: ToteLayout | None
    layout_confidence: float | None
    geometry: GeometryValidation | None
    cells: tuple[CellResult, ...]
    detected_cells: tuple[CellGeometry, ...]
    model_versions: dict[str, str]
    decision_version: str
    completed_at: datetime
    image_uri: str | None = None
    tote_polygon: Polygon | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_PIXELS
    metadata: dict[str, Any] = field(default_factory=dict)
    method_results: dict[str, tuple[CellResult, ...]] | None = None
