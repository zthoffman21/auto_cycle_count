from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from tote_vision.core.decision import DecisionEngine
from tote_vision.core.geometry import GeometryValidator
from tote_vision.core.models import (
    CellResult,
    GeometryValidation,
    InspectionDecision,
    InspectionRequest,
    InspectionResult,
    LayoutDetection,
    ReasonCode,
    ToteDetection,
)
from tote_vision.core.ports import CellClassifier, LayoutDetector, ToteDetector

logger = logging.getLogger(__name__)


class InspectEmptyTote:
    def __init__(
        self,
        tote_detector: ToteDetector,
        layout_detector: LayoutDetector,
        cell_classifiers: dict[str, CellClassifier],
        geometry_validator: GeometryValidator,
        decision_engine: DecisionEngine,
        decision_version: str,
    ) -> None:
        if not cell_classifiers:
            raise ValueError("at least one cell classifier is required")
        self._tote_detector = tote_detector
        self._layout_detector = layout_detector
        self._cell_classifiers = cell_classifiers
        self._primary_method = next(iter(cell_classifiers))
        self._geometry_validator = geometry_validator
        self._decision_engine = decision_engine
        self._decision_version = decision_version

    async def execute(
        self,
        request: InspectionRequest,
        classifier_method: str | None = None,
        comparison: bool = False,
    ) -> InspectionResult:
        rid = request.inspection_id
        t_total = time.perf_counter()
        try:
            t0 = time.perf_counter()
            tote = await self._tote_detector.detect(request)
            t_tote = time.perf_counter() - t0
            logger.info(
                "tote detection completed in %.1f ms (detected=%s)",
                t_tote * 1000,
                tote.detected,
                extra={"inspection_id": rid},
            )

            if not tote.detected:
                decision, reason = self._decision_engine.decide(
                    tote_detected=False, geometry=None, cells=()
                )
                logger.info(
                    "pipeline finished in %.1f ms | decision=%s reason=%s",
                    (time.perf_counter() - t_total) * 1000,
                    decision.value,
                    reason.value,
                    extra={"inspection_id": rid},
                )
                return self._result(request, decision, reason, tote=tote)

            t0 = time.perf_counter()
            layout = await self._layout_detector.detect(request, tote)
            t_layout = time.perf_counter() - t0
            logger.info(
                "layout detection completed in %.1f ms (layout=%s cells=%d)",
                t_layout * 1000,
                layout.layout.value,
                len(layout.cells),
                extra={"inspection_id": rid},
            )

            t0 = time.perf_counter()
            geometry = self._geometry_validator.validate(tote, layout)
            t_geometry = time.perf_counter() - t0
            logger.info(
                "geometry validation completed in %.1f ms (valid=%s)",
                t_geometry * 1000,
                geometry.valid,
                extra={"inspection_id": rid},
            )

            if not geometry.valid:
                decision, reason = self._decision_engine.decide(
                    tote_detected=True, geometry=geometry, cells=()
                )
                logger.info(
                    "pipeline finished in %.1f ms"
                    " | tote=%.1f ms layout=%.1f ms geometry=%.1f ms"
                    " | decision=%s reason=%s",
                    (time.perf_counter() - t_total) * 1000,
                    t_tote * 1000,
                    t_layout * 1000,
                    t_geometry * 1000,
                    decision.value,
                    reason.value,
                    extra={"inspection_id": rid},
                )
                return self._completed_result(
                    request=request,
                    decision=decision,
                    reason=reason,
                    tote=tote,
                    layout=layout,
                    geometry=geometry,
                    cells=(),
                )

            classifiers_to_run = self._resolve_classifiers(classifier_method, comparison)

            t0 = time.perf_counter()
            method_results: dict[str, tuple[CellResult, ...]] = {}
            for method_name, classifier in classifiers_to_run.items():
                results = tuple(
                    await asyncio.gather(
                        *(classifier.classify(request, cell) for cell in layout.cells)
                    )
                )
                method_results[method_name] = results

            t_classify = time.perf_counter() - t0
            logger.info(
                "cell classification completed in %.1f ms (%d cells, %d method(s))",
                t_classify * 1000,
                len(layout.cells),
                len(method_results),
                extra={"inspection_id": rid},
            )

            primary_method = next(iter(classifiers_to_run))
            primary_cells = method_results[primary_method]
            decision, reason = self._decision_engine.decide(
                tote_detected=True, geometry=geometry, cells=primary_cells
            )
            logger.info(
                "pipeline finished in %.1f ms"
                " | tote=%.1f ms layout=%.1f ms geometry=%.1f ms classify=%.1f ms"
                " | decision=%s reason=%s",
                (time.perf_counter() - t_total) * 1000,
                t_tote * 1000,
                t_layout * 1000,
                t_geometry * 1000,
                t_classify * 1000,
                decision.value,
                reason.value,
                extra={"inspection_id": rid},
            )
            return self._completed_result(
                request=request,
                decision=decision,
                reason=reason,
                tote=tote,
                layout=layout,
                geometry=geometry,
                cells=primary_cells,
                method_results=method_results if comparison or len(method_results) > 1 else None,
            )
        except Exception as exc:
            logger.exception(
                "inspection inference failed after %.1f ms",
                (time.perf_counter() - t_total) * 1000,
                extra={"inspection_id": request.inspection_id},
            )
            return InspectionResult(
                inspection_id=request.inspection_id,
                tote_id=request.tote_id,
                decision=InspectionDecision.ERROR,
                reason_code=ReasonCode.INFERENCE_ERROR,
                observed_layout=None,
                layout_confidence=None,
                geometry=None,
                cells=(),
                detected_cells=(),
                model_versions={},
                decision_version=self._decision_version,
                completed_at=datetime.now(UTC),
                image_uri=request.image_uri,
                metadata={"errorType": type(exc).__name__},
            )

    def _resolve_classifiers(
        self, classifier_method: str | None, comparison: bool
    ) -> dict[str, CellClassifier]:
        if comparison:
            return self._cell_classifiers
        if classifier_method is None:
            return {self._primary_method: self._cell_classifiers[self._primary_method]}
        if classifier_method not in self._cell_classifiers:
            available = ", ".join(self._cell_classifiers)
            raise ValueError(
                f"unknown classifier method '{classifier_method}'; available: {available}"
            )
        return {classifier_method: self._cell_classifiers[classifier_method]}

    def _result(
        self,
        request: InspectionRequest,
        decision: InspectionDecision,
        reason: ReasonCode,
        *,
        tote: ToteDetection,
    ) -> InspectionResult:
        return InspectionResult(
            inspection_id=request.inspection_id,
            tote_id=request.tote_id,
            decision=decision,
            reason_code=reason,
            observed_layout=None,
            layout_confidence=None,
            geometry=None,
            cells=(),
            detected_cells=(),
            model_versions={"toteDetector": tote.model_version},
            decision_version=self._decision_version,
            completed_at=datetime.now(UTC),
            image_uri=request.image_uri,
            tote_polygon=tote.polygon,
            coordinate_space=tote.coordinate_space,
        )

    def _completed_result(
        self,
        *,
        request: InspectionRequest,
        decision: InspectionDecision,
        reason: ReasonCode,
        tote: ToteDetection,
        layout: LayoutDetection,
        geometry: GeometryValidation,
        cells: tuple[CellResult, ...],
        method_results: dict[str, tuple[CellResult, ...]] | None = None,
    ) -> InspectionResult:
        model_versions: dict[str, str] = {
            "toteDetector": tote.model_version,
            "layoutDetector": layout.model_version,
        }
        if cells:
            model_versions["cellClassifier"] = cells[0].model_version
        return InspectionResult(
            inspection_id=request.inspection_id,
            tote_id=request.tote_id,
            decision=decision,
            reason_code=reason,
            observed_layout=layout.layout,
            layout_confidence=layout.confidence,
            geometry=geometry,
            cells=cells,
            detected_cells=layout.cells,
            model_versions=model_versions,
            decision_version=self._decision_version,
            completed_at=datetime.now(UTC),
            image_uri=request.image_uri,
            tote_polygon=tote.polygon,
            coordinate_space=tote.coordinate_space,
            method_results=method_results,
        )
