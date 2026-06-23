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

_LABELS = (
    CellClassification.EMPTY,
    CellClassification.NON_EMPTY,
)


class CellExtractionError(ValueError):
    pass


class Dinov2CellClassifier:
    def __init__(
        self,
        model_path: Path,
        classifier_path: Path,
        image_resolver: LocalImageResolver,
        device: str,
        empty_threshold: float,
        non_empty_threshold: float,
        runtime: Any | None = None,
        classifier_name: str = "linear_probe",
    ) -> None:
        self._name = classifier_name
        self._model_path = model_path
        self._classifier_path = classifier_path
        self._image_resolver = image_resolver
        self._device = device
        self._empty_threshold = empty_threshold
        self._non_empty_threshold = non_empty_threshold
        self._runtime = runtime or _Dinov2Runtime(model_path, classifier_path, device)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def classify(
        self, request: InspectionRequest, cell: CellGeometry
    ) -> CellResult:
        try:
            probabilities = await asyncio.to_thread(self._classify_sync, request, cell)
        except CellExtractionError:
            return CellResult(
                cell_id=cell.cell_id,
                classification=CellClassification.BAD_CAPTURE,
                empty_probability=0.0,
                non_empty_probability=0.0,
                uncertain_probability=0.0,
                model_name=f"dinov2-{self._model_path.name}-linear-head",
                model_version=self._classifier_path.name,
                polygon=cell.polygon,
                crop_uri=cell.crop_uri,
                mask_uri=cell.mask_uri,
            )
        classification = _classification_for(
            probabilities,
            empty_threshold=self._empty_threshold,
            non_empty_threshold=self._non_empty_threshold,
        )
        return CellResult(
            cell_id=cell.cell_id,
            classification=classification,
            empty_probability=probabilities[0],
            non_empty_probability=probabilities[1],
            uncertain_probability=1.0 - max(probabilities),
            model_name=f"dinov2-{self._model_path.name}-linear-head",
            model_version=self._classifier_path.name,
            polygon=cell.polygon,
            crop_uri=cell.crop_uri,
            mask_uri=cell.mask_uri,
        )

    def _classify_sync(
        self, request: InspectionRequest, cell: CellGeometry
    ) -> tuple[float, float]:
        image_path = self._image_resolver.resolve(request.image_uri)
        with self._lock:
            values = self._runtime.predict(image_path, cell.polygon)
        if len(values) != len(_LABELS):
            raise ValueError("DINOv2 classifier head must return two class probabilities")
        return tuple(float(value) for value in values)


class _Dinov2Runtime:
    def __init__(self, model_path: Path, classifier_path: Path, device: str) -> None:
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("install the 'vision' dependency group to use DINOv2") from exc

        self._torch = torch
        self._device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self._device)
        self._model.eval()
        head = load_file(classifier_path, device=str(self._device))
        self._weight = head["weight"]
        self._bias = head["bias"]
        if self._weight.shape[0] != len(_LABELS) or self._bias.shape[0] != len(_LABELS):
            raise ValueError("DINOv2 classifier head must use EMPTY/NON_EMPTY order")

    def predict(self, image_path: Path, polygon: tuple[tuple[float, float], ...]) -> list[float]:
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
                    raise CellExtractionError("cell polygon produced an empty mask")
                background = Image.new("RGB", crop.size, (0, 0, 0))
                masked_crop = Image.composite(crop, background, mask)
        except OSError as exc:
            raise CellExtractionError("cell crop could not be extracted") from exc

        inputs = self._processor(images=masked_crop, return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            embedding = output.last_hidden_state[:, 0]
            logits = self._torch.nn.functional.linear(embedding, self._weight, self._bias)
            probabilities = self._torch.softmax(logits, dim=-1)[0]
        return probabilities.detach().cpu().tolist()


def _classification_for(
    probabilities: tuple[float, float],
    *,
    empty_threshold: float,
    non_empty_threshold: float,
) -> CellClassification:
    empty, non_empty = probabilities
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
        raise CellExtractionError("cell polygon produces an empty image crop")
    return left, top, right, bottom
