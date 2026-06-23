from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tote_vision.adapters.image_resolver import LocalImageResolver
from tote_vision.core.models import (
    CellGeometry,
    CoordinateSpace,
    InspectionRequest,
    LayoutDetection,
    Polygon,
    ToteDetection,
    ToteLayout,
)


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    polygon: Polygon
    mask: Any | None = None


class RfdetrInferenceSession:
    def __init__(
        self,
        checkpoint_path: Path,
        image_resolver: LocalImageResolver,
        confidence_threshold: float,
        cache_size: int = 32,
        model: Any | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._image_resolver = image_resolver
        self._confidence_threshold = confidence_threshold
        self._cache_size = cache_size
        self._model = model or _load_model(checkpoint_path)
        self._cache: OrderedDict[str, tuple[Detection, ...]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self._checkpoint_path.name

    async def predict(self, request: InspectionRequest) -> tuple[Detection, ...]:
        return await asyncio.to_thread(self._predict_sync, request)

    def _predict_sync(self, request: InspectionRequest) -> tuple[Detection, ...]:
        with self._lock:
            cached = self._cache.get(request.inspection_id)
            if cached is not None:
                self._cache.move_to_end(request.inspection_id)
                return cached

            image_path = self._image_resolver.resolve(request.image_uri)
            raw = self._model.predict(str(image_path), threshold=self._confidence_threshold)
            detections = _normalize_detections(raw)
            self._cache[request.inspection_id] = detections
            self._cache.move_to_end(request.inspection_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return detections


class RfdetrToteDetector:
    def __init__(
        self,
        session: RfdetrInferenceSession,
        tote_class_id: int,
        confidence_threshold: float = 0.1,
    ) -> None:
        self._session = session
        self._tote_class_id = tote_class_id
        self._confidence_threshold = confidence_threshold

    async def detect(self, request: InspectionRequest) -> ToteDetection:
        detections = await self._session.predict(request)
        candidates = [
            item
            for item in detections
            if item.class_id == self._tote_class_id
            and item.confidence >= self._confidence_threshold
        ]
        if not candidates:
            return ToteDetection(
                detected=False,
                confidence=0.0,
                polygon=None,
                model_name="rf-detr-segmentation",
                model_version=self._session.version,
            )
        tote = max(candidates, key=lambda item: item.confidence)
        return ToteDetection(
            detected=True,
            confidence=tote.confidence,
            polygon=tote.polygon,
            model_name="rf-detr-segmentation",
            model_version=self._session.version,
            coordinate_space=CoordinateSpace.IMAGE_PIXELS,
        )


class RfdetrLayoutDetector:
    def __init__(
        self,
        session: RfdetrInferenceSession,
        cell_class_id: int,
        tote_class_id: int = 0,
        confidence_threshold: float = 0.5,
        min_containment: float = 0.9,
        duplicate_iou_threshold: float = 0.7,
        max_cell_iou: float = 0.15,
    ) -> None:
        self._session = session
        self._cell_class_id = cell_class_id
        self._tote_class_id = tote_class_id
        self._confidence_threshold = confidence_threshold
        self._min_containment = min_containment
        self._duplicate_iou_threshold = duplicate_iou_threshold
        self._max_cell_iou = max_cell_iou

    async def detect(
        self, request: InspectionRequest, tote: ToteDetection
    ) -> LayoutDetection:
        detections = await self._session.predict(request)
        tote_candidates = [
            item for item in detections if item.class_id == self._tote_class_id
        ]
        tote_detection = max(tote_candidates, key=lambda item: item.confidence, default=None)
        cells = []
        if tote_detection is not None:
            for candidate in detections:
                if (
                    candidate.class_id != self._cell_class_id
                    or candidate.confidence < self._confidence_threshold
                ):
                    continue
                constrained = _constrain_to_tote(
                    candidate,
                    tote_detection,
                    self._min_containment,
                )
                if constrained is not None:
                    cells.append(constrained)
        cells = _suppress_duplicates(cells, self._duplicate_iou_threshold)
        ordered = _reading_order(cells)
        layout = {
            1: ToteLayout.OPEN,
            2: ToteLayout.TWO_CELL,
            4: ToteLayout.FOUR_CELL,
        }.get(len(ordered), ToteLayout.UNKNOWN)
        if _has_excessive_cell_overlap(ordered, self._max_cell_iou):
            layout = ToteLayout.UNKNOWN
        cell_polygons = (
            (tote.polygon,)
            if layout is ToteLayout.OPEN and tote.polygon is not None
            else tuple(item.polygon for item in ordered)
        )
        confidence = min((item.confidence for item in ordered), default=0.0)
        return LayoutDetection(
            layout=layout,
            confidence=confidence,
            cells=tuple(
                CellGeometry(cell_id=chr(ord("A") + index), polygon=polygon)
                for index, polygon in enumerate(cell_polygons)
            ),
            model_name="rf-detr-constrained-cell-segmentation",
            model_version=self._session.version,
        )


def _load_model(checkpoint_path: Path) -> Any:
    try:
        from rfdetr import RFDETR
    except ImportError as exc:
        raise RuntimeError("install the 'vision' dependency group to use RF-DETR") from exc
    return RFDETR.from_checkpoint(checkpoint_path)


def _normalize_detections(raw: Any) -> tuple[Detection, ...]:
    masks = getattr(raw, "mask", None)
    normalized: list[Detection] = []
    for index, (box, confidence, class_id) in enumerate(
        zip(raw.xyxy, raw.confidence, raw.class_id, strict=True)
    ):
        box_tuple = tuple(float(value) for value in box)
        mask = masks[index] if masks is not None else None
        polygon = _mask_polygon(mask) if mask is not None else _box_polygon(box_tuple)
        normalized.append(
            Detection(
                class_id=int(class_id),
                confidence=float(confidence),
                box=box_tuple,
                polygon=polygon,
                mask=mask,
            )
        )
    return tuple(normalized)


def _mask_polygon(mask: Any) -> Polygon:
    import cv2
    import numpy as np

    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("RF-DETR returned an empty instance mask")
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, closed=True)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    return tuple((float(point[0][0]), float(point[0][1])) for point in simplified)


def _box_polygon(box: tuple[float, float, float, float]) -> Polygon:
    x1, y1, x2, y2 = box
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _constrain_to_tote(
    cell: Detection,
    tote: Detection,
    min_containment: float,
) -> Detection | None:
    if cell.mask is not None and tote.mask is not None:
        import numpy as np

        cell_mask = np.asarray(cell.mask) > 0.5
        tote_mask = np.asarray(tote.mask) > 0.5
        if cell_mask.shape != tote_mask.shape or not cell_mask.any():
            return None
        clipped = cell_mask & tote_mask
        if clipped.sum() / cell_mask.sum() < min_containment:
            return None
        ys, xs = np.nonzero(clipped)
        return Detection(
            class_id=cell.class_id,
            confidence=cell.confidence,
            box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
            polygon=_mask_polygon(clipped),
            mask=clipped,
        )

    tote_x = [point[0] for point in tote.polygon]
    tote_y = [point[1] for point in tote.polygon]
    left = max(cell.box[0], min(tote_x))
    top = max(cell.box[1], min(tote_y))
    right = min(cell.box[2], max(tote_x))
    bottom = min(cell.box[3], max(tote_y))
    cell_area = max(0.0, cell.box[2] - cell.box[0]) * max(0.0, cell.box[3] - cell.box[1])
    overlap_area = max(0.0, right - left) * max(0.0, bottom - top)
    if cell_area == 0 or overlap_area / cell_area < min_containment:
        return None
    box = (left, top, right, bottom)
    return Detection(cell.class_id, cell.confidence, box, _box_polygon(box))


def _suppress_duplicates(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_detection_iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _detection_iou(first: Detection, second: Detection) -> float:
    if first.mask is not None and second.mask is not None:
        import numpy as np

        first_mask = np.asarray(first.mask) > 0.5
        second_mask = np.asarray(second.mask) > 0.5
        if first_mask.shape == second_mask.shape:
            union = (first_mask | second_mask).sum()
            return float((first_mask & second_mask).sum() / union) if union else 0.0
    left = max(first.box[0], second.box[0])
    top = max(first.box[1], second.box[1])
    right = min(first.box[2], second.box[2])
    bottom = min(first.box[3], second.box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first.box[2] - first.box[0]) * (first.box[3] - first.box[1])
    second_area = (second.box[2] - second.box[0]) * (second.box[3] - second.box[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _reading_order(detections: list[Detection]) -> list[Detection]:
    if len(detections) == 4:
        by_y = sorted(detections, key=lambda item: (item.box[1] + item.box[3]) / 2)
        top = sorted(by_y[:2], key=lambda item: (item.box[0] + item.box[2]) / 2)
        bottom = sorted(by_y[2:], key=lambda item: (item.box[0] + item.box[2]) / 2)
        return top + bottom
    return sorted(detections, key=lambda item: (item.box[0] + item.box[2]) / 2)


def _has_excessive_cell_overlap(detections: list[Detection], max_iou: float) -> bool:
    return any(
        _detection_iou(first, second) > max_iou
        for index, first in enumerate(detections)
        for second in detections[index + 1 :]
    )
