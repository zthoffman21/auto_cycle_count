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
MIN_CELL_CONTAINMENT_RATIO = 0.95
MAX_CELL_PROTRUSION_RATIO = 0.03
MAX_CELL_OVERLAP_IOU = 0.20
MIN_TOTAL_COVERAGE_RATIOS = {
    ToteLayout.OPEN: 0.50,
    ToteLayout.TWO_CELL: 0.55,
    ToteLayout.FOUR_CELL: 0.60,
}


class GeometryValidator:
    def validate(
        self,
        tote: ToteDetection,
        layout: LayoutDetection,
    ) -> GeometryValidation:
        issues: list[str] = []
        expected_count = EXPECTED_CELL_COUNTS.get(layout.layout)
        cell_count_matches = False

        if expected_count is None:
            issues.append("observed layout is unknown")
        elif len(layout.cells) != expected_count:
            issues.append(
                f"{layout.layout.value} requires {expected_count} cells; "
                f"detected {len(layout.cells)}"
            )
        else:
            cell_count_matches = True

        if tote.polygon is None:
            issues.append("tote polygon is missing")
        else:
            basic_issue_count = len(issues)
            for cell in layout.cells:
                if len(cell.polygon) < 3:
                    issues.append(f"cell {cell.cell_id} polygon has fewer than three points")
                elif not _polygon_inside_bounds(cell.polygon, tote.polygon):
                    issues.append(f"cell {cell.cell_id} extends outside tote bounds")
            if cell_count_matches and len(issues) == basic_issue_count:
                issues.extend(_validate_cell_overlap(layout))
                issues.extend(_validate_total_coverage(tote, layout))

        confidence = min(tote.confidence, layout.confidence) if not issues else 0.0
        return GeometryValidation(
            valid=not issues,
            confidence=confidence,
            issues=tuple(issues),
        )


def _validate_cell_overlap(layout: LayoutDetection) -> list[str]:
    issues: list[str] = []
    for index, first in enumerate(layout.cells):
        for second in layout.cells[index + 1 :]:
            iou = _polygon_iou(first.polygon, second.polygon)
            if iou > MAX_CELL_OVERLAP_IOU:
                issues.append(
                    f"cells {first.cell_id} and {second.cell_id} overlap too much"
                )
    return issues


def _validate_total_coverage(
    tote: ToteDetection,
    layout: LayoutDetection,
) -> list[str]:
    minimum = MIN_TOTAL_COVERAGE_RATIOS.get(layout.layout)
    if minimum is None or tote.polygon is None:
        return []

    tote_area = _polygon_area(tote.polygon)
    if tote_area <= 0:
        return ["tote polygon has zero area"]

    clipped_cell_area = sum(
        _polygon_area(_clip_polygon_to_convex_polygon(cell.polygon, tote.polygon))
        for cell in layout.cells
    )
    coverage = clipped_cell_area / tote_area
    if coverage < minimum:
        return [f"cell coverage is too low ({coverage:.0%})"]
    return []


def _polygon_iou(first: Polygon, second: Polygon) -> float:
    first_area = _polygon_area(first)
    second_area = _polygon_area(second)
    if first_area <= 0 or second_area <= 0:
        return 0.0

    if _is_convex(first) and _is_convex(second):
        intersection_area = _polygon_area(_clip_polygon_to_convex_polygon(first, second))
    else:
        intersection_area = _bounds_intersection_area(first, second)
    union = first_area + second_area - intersection_area
    return intersection_area / union if union > 0 else 0.0


def _is_convex(polygon: Polygon) -> bool:
    if len(polygon) < 4:
        return True

    orientation = 0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        next_next_point = polygon[(index + 2) % len(polygon)]
        cross = (next_point[0] - point[0]) * (next_next_point[1] - next_point[1]) - (
            next_point[1] - point[1]
        ) * (next_next_point[0] - next_point[0])
        if abs(cross) <= 1e-6:
            continue
        current_orientation = 1 if cross > 0 else -1
        if orientation == 0:
            orientation = current_orientation
        elif orientation != current_orientation:
            return False
    return True


def _bounds_intersection_area(first: Polygon, second: Polygon) -> float:
    first_min_x, first_min_y, first_max_x, first_max_y = _bounds(first)
    second_min_x, second_min_y, second_max_x, second_max_y = _bounds(second)
    width = max(0.0, min(first_max_x, second_max_x) - max(first_min_x, second_min_x))
    height = max(0.0, min(first_max_y, second_max_y) - max(first_min_y, second_min_y))
    return width * height


def _bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_inside_bounds(inner: Polygon, outer: Polygon) -> bool:
    tolerance = _bounds_tolerance(outer, MAX_CELL_PROTRUSION_RATIO)
    inner_area = _polygon_area(inner)
    if inner_area > 0:
        intersection = _clip_polygon_to_convex_polygon(inner, outer)
        containment = _polygon_area(intersection) / inner_area if intersection else 0.0
        max_protrusion = max(
            (
                0.0
                if _point_inside_or_on_polygon(point, outer)
                else _distance_to_polygon(point, outer)
            )
            for point in inner
        )
        if containment >= MIN_CELL_CONTAINMENT_RATIO and max_protrusion <= tolerance:
            return True

    tolerance = _bounds_tolerance(outer, 0.01)
    return all(
        _point_inside_or_on_polygon(point, outer)
        or _distance_to_polygon(point, outer) <= tolerance
        for point in inner
    )


def _bounds_tolerance(polygon: Polygon, ratio: float) -> float:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return max(1.0, min(max(xs) - min(xs), max(ys) - min(ys)) * ratio)


def _polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        )
        / 2
    )


def _clip_polygon_to_convex_polygon(polygon: Polygon, clip: Polygon) -> Polygon:
    if len(polygon) < 3 or len(clip) < 3:
        return ()

    result = polygon
    clip_orientation = _signed_area(clip)
    for start, end in zip(clip, clip[1:] + clip[:1], strict=True):
        if not result:
            return ()
        next_result: list[tuple[float, float]] = []
        for index, current in enumerate(result):
            previous = result[(index + len(result) - 1) % len(result)]
            current_inside = _inside_clip_edge(current, start, end, clip_orientation)
            previous_inside = _inside_clip_edge(previous, start, end, clip_orientation)
            if current_inside != previous_inside:
                intersection = _line_intersection(previous, current, start, end)
                if intersection is not None:
                    next_result.append(intersection)
            if current_inside:
                next_result.append(current)
        result = tuple(next_result)
    return result


def _signed_area(polygon: Polygon) -> float:
    return (
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        )
        / 2
    )


def _inside_clip_edge(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    clip_orientation: float,
) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (
        end[1] - start[1]
    ) * (point[0] - start[0])
    return cross >= -1e-6 if clip_orientation >= 0 else cross <= 1e-6


def _line_intersection(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[float, float] | None:
    first_dx = first_end[0] - first_start[0]
    first_dy = first_end[1] - first_start[1]
    second_dx = second_end[0] - second_start[0]
    second_dy = second_end[1] - second_start[1]
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) <= 1e-12:
        return None
    ratio = (
        (second_start[0] - first_start[0]) * second_dy
        - (second_start[1] - first_start[1]) * second_dx
    ) / denominator
    return (first_start[0] + ratio * first_dx, first_start[1] + ratio * first_dy)


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
