from __future__ import annotations

from tote_vision.core.models import (
    GeometryValidation,
    LayoutDetection,
    Polygon,
    ToteDetection,
    ToteLayout,
)

EXPECTED_CELL_COUNTS = {
    ToteLayout.OPEN: 1,
    ToteLayout.TWO_CELL: 2,
    ToteLayout.FOUR_CELL: 4,
}


class GeometryValidator:
    def validate(
        self,
        tote: ToteDetection,
        layout: LayoutDetection,
    ) -> GeometryValidation:
        issues: list[str] = []
        expected_count = EXPECTED_CELL_COUNTS.get(layout.layout)

        if expected_count is None:
            issues.append("observed layout is unknown")
        elif len(layout.cells) != expected_count:
            issues.append(
                f"{layout.layout.value} requires {expected_count} cells; "
                f"detected {len(layout.cells)}"
            )

        if tote.polygon is None:
            issues.append("tote polygon is missing")
        else:
            for cell in layout.cells:
                if len(cell.polygon) < 3:
                    issues.append(f"cell {cell.cell_id} polygon has fewer than three points")
                elif not _polygon_inside_bounds(cell.polygon, tote.polygon):
                    issues.append(f"cell {cell.cell_id} extends outside tote bounds")

        confidence = min(tote.confidence, layout.confidence) if not issues else 0.0
        return GeometryValidation(
            valid=not issues,
            confidence=confidence,
            issues=tuple(issues),
        )


def _polygon_inside_bounds(inner: Polygon, outer: Polygon) -> bool:
    outer_x = [point[0] for point in outer]
    outer_y = [point[1] for point in outer]
    tolerance = max(
        1.0,
        min(max(outer_x) - min(outer_x), max(outer_y) - min(outer_y)) * 0.01,
    )
    return all(
        _point_inside_or_on_polygon(point, outer)
        or _distance_to_polygon(point, outer) <= tolerance
        for point in inner
    )


def _point_inside_or_on_polygon(point: tuple[float, float], polygon: Polygon) -> bool:
    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-6
            and min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6
            and min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6
        ):
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
    return inside


def _distance_to_polygon(point: tuple[float, float], polygon: Polygon) -> float:
    return min(
        _distance_to_segment(point, start, polygon[(index + 1) % len(polygon)])
        for index, start in enumerate(polygon)
    )


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    import math

    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x**2 + delta_y**2
    if length_squared == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest_x = start[0] + projection * delta_x
    closest_y = start[1] + projection * delta_y
    return math.hypot(point[0] - closest_x, point[1] - closest_y)
