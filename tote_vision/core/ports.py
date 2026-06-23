from __future__ import annotations

from typing import Protocol

from tote_vision.core.models import (
    CellGeometry,
    CellResult,
    InspectionRequest,
    LayoutDetection,
    ToteDetection,
)


class ToteDetector(Protocol):
    async def detect(self, request: InspectionRequest) -> ToteDetection: ...


class LayoutDetector(Protocol):
    async def detect(
        self, request: InspectionRequest, tote: ToteDetection
    ) -> LayoutDetection: ...


class CellClassifier(Protocol):
    @property
    def name(self) -> str: ...

    async def classify(
        self, request: InspectionRequest, cell: CellGeometry
    ) -> CellResult: ...

