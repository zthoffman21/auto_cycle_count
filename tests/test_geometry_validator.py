from tote_vision.core.geometry import GeometryValidator
from tote_vision.core.models import (
    CellGeometry,
    LayoutDetection,
    ToteDetection,
    ToteLayout,
)


def _tote() -> ToteDetection:
    return ToteDetection(
        detected=True,
        confidence=0.95,
        polygon=((0, 0), (100, 0), (100, 100), (0, 100)),
        model_name="test",
        model_version="test",
    )


def test_rejects_wrong_cell_count() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.FOUR_CELL,
        confidence=0.9,
        cells=(CellGeometry("A", ((1, 1), (10, 1), (10, 10), (1, 10))),),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert not result.valid
    assert result.issues == ("FOUR_CELL requires 4 cells; detected 1",)


def test_rejects_cell_inside_bounds_but_outside_tote_polygon() -> None:
    tote = ToteDetection(
        detected=True,
        confidence=0.95,
        polygon=((50, 0), (100, 50), (50, 100), (0, 50)),
        model_name="test",
        model_version="test",
    )
    layout = LayoutDetection(
        layout=ToteLayout.OPEN,
        confidence=0.9,
        cells=(CellGeometry("A", ((0, 0), (10, 0), (10, 10), (0, 10))),),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(tote, layout)

    assert not result.valid
    assert result.issues == ("cell A extends outside tote bounds",)


def test_accepts_minor_contour_simplification_drift() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.OPEN,
        confidence=0.9,
        cells=(CellGeometry("A", ((-0.5, 0), (100.5, 0), (100.5, 100), (-0.5, 100))),),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert result.valid
