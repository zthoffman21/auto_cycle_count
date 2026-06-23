from __future__ import annotations

import asyncio
import math
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


class PatchAnomalyCellClassifier:
    """One-class patch-level anomaly detector.

    Learns only what an empty cell looks like. Any patch embedding that is far
    from the empty centroid raises the anomaly score — so novel objects are
    detected without requiring them in training data.
    """

    def __init__(
        self,
        model_path: Path,
        anomaly_model_path: Path,
        image_resolver: LocalImageResolver,
        device: str,
        empty_threshold: float,
        non_empty_threshold: float,
        runtime: Any | None = None,
    ) -> None:
        self._model_path = model_path
        self._anomaly_model_path = anomaly_model_path
        self._image_resolver = image_resolver
        self._device = device
        self._empty_threshold = empty_threshold
        self._non_empty_threshold = non_empty_threshold
        self._runtime = runtime or _PatchAnomalyRuntime(model_path, anomaly_model_path, device)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "patch_anomaly"

    async def classify(
        self, request: InspectionRequest, cell: CellGeometry
    ) -> CellResult:
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
                model_name=f"dinov2-{self._model_path.name}-patch-anomaly",
                model_version=self._anomaly_model_path.name,
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
            model_name=f"dinov2-{self._model_path.name}-patch-anomaly",
            model_version=self._anomaly_model_path.name,
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


class _PatchAnomalyRuntime:
    def __init__(self, model_path: Path, anomaly_model_path: Path, device: str) -> None:
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "install the 'vision' dependency group to use patch anomaly"
            ) from exc

        self._torch = torch
        self._device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self._device)
        self._model.eval()

        data = load_file(anomaly_model_path, device=str(self._device))
        self._centroids = data["centroids"]  # (N_patches, embed_dim)
        self._scale = float(data["scale"].item())
        # Normalized median empty score — used to calibrate the sigmoid so the
        # transition is anchored to the actual empty distribution, not a fixed 1.0.
        # Falls back to 0.85 for models trained before this field was added.
        empty_p50_norm = float(data["empty_p50_norm"].item()) if "empty_p50_norm" in data else 0.85
        gap = 1.0 - empty_p50_norm
        # k spans the empty-median-to-scale gap; center sits 65% of the way through it
        self._sigmoid_k = 1.5 / gap
        self._sigmoid_center = empty_p50_norm + gap * 0.65

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
                background = Image.new("RGB", crop.size, (0, 0, 0))
                masked_crop = Image.composite(crop, background, mask)
        except OSError as exc:
            raise _CellExtractionError("cell crop could not be extracted") from exc

        inputs = self._processor(images=masked_crop, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            # skip CLS token; keep per-position patch tokens
            patches = output.last_hidden_state[0, 1:, :]  # (N_patches, embed_dim)
            # compare each patch to its position-specific centroid
            dists = self._torch.norm(patches - self._centroids, dim=-1)  # (N_patches,)
            max_dist = float(dists.max().item())

        return _anomaly_score_to_probabilities(
            max_dist / self._scale, self._sigmoid_k, self._sigmoid_center
        )


class _CellExtractionError(ValueError):
    pass


def _anomaly_score_to_probabilities(score: float, k: float, center: float) -> tuple[float, float]:
    # Clamp to avoid math overflow for very large/small scores
    z = max(-50.0, min(50.0, k * (score - center)))
    non_empty = 1.0 / (1.0 + math.exp(-z))
    return 1.0 - non_empty, non_empty


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
