from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from tote_vision.adapters.image_resolver import LocalImageResolver
from tote_vision.core.models import (
    CellClassification,
    CellGeometry,
    CellResult,
    InspectionRequest,
)

# Prompt fed to GroundingDINO — generic enough to catch any warehouse item.
# GroundingDINO convention: each class ends with "." and classes are separated by ". ".
_TEXT_PROMPT = "object."


class GroundingDinoCellClassifier:
    """Zero-shot cell classifier using GroundingDINO text-prompted object detection.

    No domain-specific training required.  Any object detected above the confidence
    threshold means the cell is NON_EMPTY; zero detections means EMPTY.
    The box_threshold is the primary tuning knob — lower catches more items at the
    cost of more false positives.
    """

    def __init__(
        self,
        model_path: Path,
        image_resolver: LocalImageResolver,
        device: str,
        empty_threshold: float,
        non_empty_threshold: float,
        box_threshold: float = 0.25,
        runtime: Any | None = None,
    ) -> None:
        self._model_path = model_path
        self._image_resolver = image_resolver
        self._device = device
        self._empty_threshold = empty_threshold
        self._non_empty_threshold = non_empty_threshold
        self._box_threshold = box_threshold
        self._runtime = runtime or _GroundingDinoRuntime(model_path, device, box_threshold)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "grounding_dino"

    async def classify(self, request: InspectionRequest, cell: CellGeometry) -> CellResult:
        try:
            empty_prob, non_empty_prob = await asyncio.to_thread(
                self._classify_sync, request, cell
            )
        except _CellExtractionError:
            return CellResult(
                cell_id=cell.cell_id,
                classification=CellClassification.BAD_CAPTURE,
                empty_probability=0.0,
                non_empty_probability=0.0,
                uncertain_probability=0.0,
                model_name=f"grounding-dino-{self._model_path.name}",
                model_version="zero-shot",
                polygon=cell.polygon,
                crop_uri=cell.crop_uri,
                mask_uri=cell.mask_uri,
            )
        classification = _classification_for(
            empty_prob,
            non_empty_prob,
            empty_threshold=self._empty_threshold,
            non_empty_threshold=self._non_empty_threshold,
        )
        return CellResult(
            cell_id=cell.cell_id,
            classification=classification,
            empty_probability=empty_prob,
            non_empty_probability=non_empty_prob,
            uncertain_probability=1.0 - max(empty_prob, non_empty_prob),
            model_name=f"grounding-dino-{self._model_path.name}",
            model_version="zero-shot",
            polygon=cell.polygon,
            crop_uri=cell.crop_uri,
            mask_uri=cell.mask_uri,
        )

    def _classify_sync(
        self, request: InspectionRequest, cell: CellGeometry
    ) -> tuple[float, float]:
        image_path = self._image_resolver.resolve(request.image_uri)
        with self._lock:
            return self._runtime.predict(image_path, cell.polygon)


class _GroundingDinoRuntime:
    def __init__(self, model_path: Path, device: str, box_threshold: float) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "install the 'vision' dependency group to use grounding_dino"
            ) from exc

        self._torch = torch
        self._device = torch.device(device)
        self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path, local_files_only=True
            )
            .to(self._device)
            .eval()
        )
        self._box_threshold = box_threshold

    def predict(
        self, image_path: Path, polygon: tuple[tuple[float, float], ...]
    ) -> tuple[float, float]:
        from PIL import Image, ImageDraw

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                bounds = _polygon_bounds(polygon, image.width, image.height)
                crop = image.crop(bounds)
                shifted = [(x - bounds[0], y - bounds[1]) for x, y in polygon]
                mask = Image.new("L", crop.size, 0)
                ImageDraw.Draw(mask).polygon(shifted, fill=255)
                if mask.getbbox() is None:
                    raise _CellExtractionError("cell polygon produced an empty mask")
                background = Image.new("RGB", crop.size, (114, 114, 114))
                masked_crop = Image.composite(crop, background, mask)
        except OSError as exc:
            raise _CellExtractionError("cell crop could not be extracted") from exc

        inputs = self._processor(
            images=masked_crop,
            text=_TEXT_PROMPT,
            return_tensors="pt",
        )
        # Save input_ids before moving to device — processor needs them for label decoding
        input_ids = inputs["input_ids"]
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        # target_sizes expects (height, width) — PIL .size is (width, height)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=self._box_threshold,
            text_threshold=self._box_threshold * 0.8,
            target_sizes=[(masked_crop.height, masked_crop.width)],
        )
        scores = results[0]["scores"]
        if len(scores) == 0:
            return 0.95, 0.05  # no detections → confident EMPTY
        # Use the highest detection confidence directly as the non-empty probability.
        # This lets the system's non_empty_threshold separate confident detections
        # (real items) from weak ones (tote structure / texture artifacts).
        max_score = float(scores.max().item())
        return 1.0 - max_score, max_score


class _CellExtractionError(ValueError):
    pass


def _classification_for(
    empty: float,
    non_empty: float,
    *,
    empty_threshold: float,
    non_empty_threshold: float,
) -> CellClassification:
    if non_empty >= non_empty_threshold:
        return CellClassification.NON_EMPTY
    if empty >= empty_threshold:
        return CellClassification.EMPTY
    return CellClassification.UNCERTAIN


def _polygon_bounds(
    polygon: tuple[tuple[float, float], ...], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = max(0, int(min(xs)))
    top = max(0, int(min(ys)))
    right = min(image_width, int(max(xs)) + 1)
    bottom = min(image_height, int(max(ys)) + 1)
    if right <= left or bottom <= top:
        raise _CellExtractionError("cell polygon produces an empty image crop")
    return left, top, right, bottom
