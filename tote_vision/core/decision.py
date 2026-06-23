from __future__ import annotations

from dataclasses import dataclass

from tote_vision.core.models import (
    CellClassification,
    CellResult,
    GeometryValidation,
    InspectionDecision,
    ReasonCode,
)


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    non_empty_decision: InspectionDecision = InspectionDecision.REVIEW


class DecisionEngine:
    def __init__(self, policy: DecisionPolicy) -> None:
        self._policy = policy

    def decide(
        self,
        *,
        tote_detected: bool,
        geometry: GeometryValidation | None,
        cells: tuple[CellResult, ...],
    ) -> tuple[InspectionDecision, ReasonCode]:
        if not tote_detected:
            return InspectionDecision.ERROR, ReasonCode.TOTE_NOT_DETECTED
        if geometry is None or not geometry.valid:
            return InspectionDecision.ERROR, ReasonCode.INVALID_GEOMETRY
        if any(cell.classification is CellClassification.BAD_CAPTURE for cell in cells):
            return InspectionDecision.ERROR, ReasonCode.BAD_CAPTURE
        if any(cell.classification is CellClassification.NON_EMPTY for cell in cells):
            return self._policy.non_empty_decision, ReasonCode.NON_EMPTY_CELL_DETECTED
        if any(cell.classification is CellClassification.UNCERTAIN for cell in cells):
            return InspectionDecision.ERROR, ReasonCode.UNCERTAIN_CELL
        return InspectionDecision.PASS, ReasonCode.ALL_CELLS_EMPTY
