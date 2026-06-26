from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from tote_vision.core.models import CellGeometry, InspectionRequest, Polygon, ToteLayout
from tote_vision.core.ports import LayoutDetector, ToteDetector
from tote_vision.core.training_models import RegionClass, TrainingImage, TrainingRegion


class TrainingGeometryPredictionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingGeometryPrediction:
    layout: ToteLayout
    confidence: float
    regions: tuple[TrainingRegion, ...]


class TrainingGeometryPredictor:
    def __init__(self, tote_detector: ToteDetector, layout_detector: LayoutDetector) -> None:
        self._tote_detector = tote_detector
        self._layout_detector = layout_detector

    async def predict(
        self, image: TrainingImage, image_path: Path
    ) -> TrainingGeometryPrediction:
        request = InspectionRequest(
            inspection_id=f"TRAIN-{image.image_id}-{uuid4().hex[:8]}",
            tote_id=image.image_id,
            image_uri=str(image_path),
            station_id="TRAINING_DATA_STUDIO",
            camera_id="TRAINING_UPLOAD",
            captured_at=datetime.now(UTC),
        )
        tote = await self._tote_detector.detect(request)
        if not tote.detected or tote.polygon is None:
            raise TrainingGeometryPredictionError("tote was not detected")
        layout = await self._layout_detector.detect(request, tote)
        tote_polygon = _four_point_polygon(tote.polygon)
        regions = _prediction_regions(tote_polygon, layout.layout, layout.cells)
        return TrainingGeometryPrediction(
            layout=layout.layout,
            confidence=layout.confidence,
            regions=regions,
        )


def _prediction_regions(
    tote_polygon: Polygon,
    layout: ToteLayout,
    cells: tuple[CellGeometry, ...],
) -> tuple[TrainingRegion, ...]:
    if layout is ToteLayout.OPEN:
        return (
            _region(RegionClass.TOTE, tote_polygon),
            _region(RegionClass.CELL, tote_polygon, cell_id="A"),
        )

    if layout is ToteLayout.TWO_CELL:
        if len(cells) != 2:
            raise TrainingGeometryPredictionError("two-cell layout requires two cells")
        divider = _infer_two_cell_divider(tote_polygon, cells)
        cell_regions = _generated_cell_regions(tote_polygon, (divider,))
        return (
            _region(RegionClass.TOTE, tote_polygon),
            _region(RegionClass.DIVIDER, divider),
            *cell_regions,
        )

    if layout is ToteLayout.FOUR_CELL:
        if len(cells) != 4:
            raise TrainingGeometryPredictionError("four-cell layout requires four cells")
        dividers = _infer_four_cell_dividers(tote_polygon, cells)
        cell_regions = _generated_cell_regions(tote_polygon, dividers)
        return (
            _region(RegionClass.TOTE, tote_polygon),
            *(_region(RegionClass.DIVIDER, divider) for divider in dividers),
            *cell_regions,
        )

    raise TrainingGeometryPredictionError("layout could not be inferred")


def _region(
    region_class: RegionClass,
    polygon: Polygon,
    *,
    cell_id: str | None = None,
) -> TrainingRegion:
    return TrainingRegion(
        region_id=f"{region_class.value}-{uuid4().hex[:12]}",
        region_class=region_class,
        polygon=polygon,
        cell_id=cell_id,
        cell_state=None,
    )


def _four_point_polygon(polygon: Polygon) -> Polygon:
    if len(polygon) == 4:
        return _order_corners(polygon)
    try:
        import cv2
        import numpy as np

        points = np.asarray(polygon, dtype=np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        return _order_corners(tuple((float(x), float(y)) for x, y in box))
    except Exception:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return (
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        )


def _order_corners(points: Polygon) -> Polygon:
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    ordered = sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    start = min(range(len(ordered)), key=lambda index: ordered[index][0] + ordered[index][1])
    rotated = ordered[start:] + ordered[:start]
    if _signed_area(rotated) < 0:
        rotated = (rotated[0], *reversed(rotated[1:]))
    return tuple(rotated)


def _signed_area(polygon: Polygon) -> float:
    return (
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        )
        / 2
    )


def _infer_two_cell_divider(tote_polygon: Polygon, cells: tuple[CellGeometry, ...]) -> Polygon:
    first, second = sorted(cells, key=lambda cell: _center(cell.polygon))
    first_center = _center(first.polygon)
    second_center = _center(second.polygon)
    if abs(first_center[0] - second_center[0]) >= abs(first_center[1] - second_center[1]):
        x = (_bounds(first.polygon)[2] + _bounds(second.polygon)[0]) / 2
        return _axis_divider(tote_polygon, "x", x)
    y = (_bounds(first.polygon)[3] + _bounds(second.polygon)[1]) / 2
    return _axis_divider(tote_polygon, "y", y)


def _infer_four_cell_dividers(
    tote_polygon: Polygon, cells: tuple[CellGeometry, ...]
) -> tuple[Polygon, Polygon]:
    ordered = sorted(cells, key=lambda cell: _center(cell.polygon)[1])
    top = sorted(ordered[:2], key=lambda cell: _center(cell.polygon)[0])
    bottom = sorted(ordered[2:], key=lambda cell: _center(cell.polygon)[0])
    left_column = [top[0], bottom[0]]
    right_column = [top[1], bottom[1]]
    top_row = top
    bottom_row = bottom
    x = (
        sum(
            (_bounds(left.polygon)[2] + _bounds(right.polygon)[0]) / 2
            for left, right in zip(left_column, right_column, strict=True)
        )
        / 2
    )
    y = (
        sum(
            (_bounds(upper.polygon)[3] + _bounds(lower.polygon)[1]) / 2
            for upper, lower in zip(top_row, bottom_row, strict=True)
        )
        / 2
    )
    return (_axis_divider(tote_polygon, "x", x), _axis_divider(tote_polygon, "y", y))


def _axis_divider(tote_polygon: Polygon, axis: str, value: float) -> Polygon:
    intersections: list[tuple[float, float]] = []
    for start, end in zip(tote_polygon, tote_polygon[1:] + tote_polygon[:1], strict=True):
        if axis == "x":
            if (start[0] - value) * (end[0] - value) > 0 or start[0] == end[0]:
                continue
            ratio = (value - start[0]) / (end[0] - start[0])
            if 0 <= ratio <= 1:
                intersections.append((value, start[1] + ratio * (end[1] - start[1])))
        else:
            if (start[1] - value) * (end[1] - value) > 0 or start[1] == end[1]:
                continue
            ratio = (value - start[1]) / (end[1] - start[1])
            if 0 <= ratio <= 1:
                intersections.append((start[0] + ratio * (end[0] - start[0]), value))
    unique = _unique_points(intersections)
    if len(unique) < 2:
        raise TrainingGeometryPredictionError("divider could not be clipped to tote")
    return tuple(sorted(unique, key=lambda point: (point[1], point[0]))[:2])


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in points:
        if all(
            math.hypot(point[0] - existing[0], point[1] - existing[1]) > 1e-6
            for existing in unique
        ):
            unique.append(point)
    return unique


def _generated_cell_regions(
    tote_polygon: Polygon,
    dividers: tuple[Polygon, ...],
) -> tuple[TrainingRegion, ...]:
    polygons = [tote_polygon]
    for divider in dividers:
        next_polygons: list[Polygon] = []
        for polygon in polygons:
            first = _clip_polygon_to_line(polygon, divider[0], divider[1], True)
            second = _clip_polygon_to_line(polygon, divider[0], divider[1], False)
            if len(first) < 3 or len(second) < 3:
                raise TrainingGeometryPredictionError("divider does not split tote into cells")
            next_polygons.extend((first, second))
        polygons = next_polygons
    ordered = _order_cell_polygons(polygons)
    return tuple(
        _region(RegionClass.CELL, polygon, cell_id=chr(ord("A") + index))
        for index, polygon in enumerate(ordered)
    )


def _clip_polygon_to_line(
    polygon: Polygon,
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    keep_positive: bool,
) -> Polygon:
    result: list[tuple[float, float]] = []
    for index, current in enumerate(polygon):
        previous = polygon[(index + len(polygon) - 1) % len(polygon)]
        current_distance = _signed_distance(current, line_start, line_end)
        previous_distance = _signed_distance(previous, line_start, line_end)
        current_inside = current_distance >= 0 if keep_positive else current_distance <= 0
        previous_inside = previous_distance >= 0 if keep_positive else previous_distance <= 0
        if current_inside != previous_inside:
            ratio = previous_distance / (previous_distance - current_distance)
            result.append(
                (
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                )
            )
        if current_inside:
            result.append(current)
    return tuple(result)


def _signed_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    return (
        (line_end[0] - line_start[0]) * (point[1] - line_start[1])
        - (line_end[1] - line_start[1]) * (point[0] - line_start[0])
    )


def _order_cell_polygons(polygons: list[Polygon]) -> list[Polygon]:
    entries = [(polygon, _center(polygon)) for polygon in polygons]
    if len(entries) == 2:
        x_spread = abs(entries[0][1][0] - entries[1][1][0])
        y_spread = abs(entries[0][1][1] - entries[1][1][1])
        entries.sort(key=lambda entry: entry[1][0 if x_spread >= y_spread else 1])
        return [entry[0] for entry in entries]
    entries.sort(key=lambda entry: entry[1][1])
    top = sorted(entries[:2], key=lambda entry: entry[1][0])
    bottom = sorted(entries[2:], key=lambda entry: entry[1][0])
    return [entry[0] for entry in (*top, *bottom)]


def _bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _center(polygon: Polygon) -> tuple[float, float]:
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )
