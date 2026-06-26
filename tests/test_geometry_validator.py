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


def test_accepts_cell_with_high_containment_and_bounded_protrusion() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.OPEN,
        confidence=0.9,
        cells=(CellGeometry("A", ((0, 0), (102, 0), (102, 100), (0, 100))),),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert result.valid


def test_rejects_cell_with_large_boundary_protrusion() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.OPEN,
        confidence=0.9,
        cells=(CellGeometry("A", ((0, 0), (110, 0), (110, 100), (0, 100))),),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert not result.valid
    assert result.issues == ("cell A extends outside tote bounds",)


def test_rejects_excessive_cell_overlap() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.TWO_CELL,
        confidence=0.9,
        cells=(
            CellGeometry("A", ((0, 0), (75, 0), (75, 100), (0, 100))),
            CellGeometry("B", ((25, 0), (100, 0), (100, 100), (25, 100))),
        ),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert not result.valid
    assert result.issues == ("cells A and B overlap too much",)


def test_rejects_low_total_cell_coverage() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.FOUR_CELL,
        confidence=0.9,
        cells=(
            CellGeometry("A", ((0, 0), (10, 0), (10, 10), (0, 10))),
            CellGeometry("B", ((90, 0), (100, 0), (100, 10), (90, 10))),
            CellGeometry("C", ((0, 90), (10, 90), (10, 100), (0, 100))),
            CellGeometry("D", ((90, 90), (100, 90), (100, 100), (90, 100))),
        ),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert not result.valid
    assert result.issues == ("cell coverage is too low (4%)",)


def test_accepts_uneven_four_cell_layout_with_crooked_dividers() -> None:
    layout = LayoutDetection(
        layout=ToteLayout.FOUR_CELL,
        confidence=0.9,
        cells=(
            CellGeometry("A", ((0, 0), (55, 0), (45, 55), (0, 52))),
            CellGeometry("B", ((55, 0), (100, 0), (100, 50), (45, 55))),
            CellGeometry("C", ((0, 52), (45, 55), (52, 100), (0, 100))),
            CellGeometry("D", ((45, 55), (100, 50), (100, 100), (52, 100))),
        ),
        model_name="test",
        model_version="test",
    )

    result = GeometryValidator().validate(_tote(), layout)

    assert result.valid
