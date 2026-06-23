from tote_vision.core.decision import DecisionEngine, DecisionPolicy
from tote_vision.core.models import (
    CellClassification,
    CellResult,
    GeometryValidation,
    InspectionDecision,
    ReasonCode,
)


def _cell(classification: CellClassification) -> CellResult:
    return CellResult(
        cell_id="A",
        classification=classification,
        empty_probability=1.0 if classification is CellClassification.EMPTY else 0.0,
        non_empty_probability=1.0 if classification is CellClassification.NON_EMPTY else 0.0,
        uncertain_probability=1.0 if classification is CellClassification.UNCERTAIN else 0.0,
        model_name="test",
        model_version="test",
    )


VALID_GEOMETRY = GeometryValidation(valid=True, confidence=1.0)


def test_all_empty_cells_pass() -> None:
    decision = DecisionEngine(DecisionPolicy()).decide(
        tote_detected=True,
        geometry=VALID_GEOMETRY,
        cells=(_cell(CellClassification.EMPTY),),
    )

    assert decision == (InspectionDecision.PASS, ReasonCode.ALL_CELLS_EMPTY)


def test_non_empty_cell_routes_to_review_by_default() -> None:
    decision = DecisionEngine(DecisionPolicy()).decide(
        tote_detected=True,
        geometry=VALID_GEOMETRY,
        cells=(_cell(CellClassification.NON_EMPTY),),
    )

    assert decision == (
        InspectionDecision.REVIEW,
        ReasonCode.NON_EMPTY_CELL_DETECTED,
    )


