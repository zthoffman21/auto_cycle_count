from __future__ import annotations

import json
import math
import shutil
import threading
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, UnidentifiedImageError

from tote_vision.core.models import CellClassification, ToteLayout
from tote_vision.core.training_models import (
    DatasetSplit,
    RegionClass,
    TrainingImage,
    TrainingRegion,
)


class TrainingDataError(ValueError):
    pass


class TrainingDatasetStore:
    def __init__(self, root: Path, public_prefix: str = "/training-data/images") -> None:
        self._root = root.resolve()
        self._images = self._root / "images"
        self._annotations = self._root / "annotations"
        self._exports = self._root / "exports"
        self._public_prefix = public_prefix.rstrip("/")
        self._lock = threading.RLock()
        self._export_condition = threading.Condition()
        self._export_requested = False
        self._export_in_progress = False
        self._last_export_error: str | None = None
        self._images.mkdir(parents=True, exist_ok=True)
        self._annotations.mkdir(parents=True, exist_ok=True)
        self._export_thread = threading.Thread(
            target=self._export_worker,
            name="training-export-worker",
            daemon=True,
        )
        self._export_thread.start()

    @property
    def image_directory(self) -> Path:
        return self._images

    @property
    def export_directory(self) -> Path:
        return self._exports

    def export_state(self) -> tuple[bool, bool, str | None]:
        with self._export_condition:
            return (
                self._export_in_progress,
                self._export_requested,
                self._last_export_error,
            )

    def create_image(self, original_filename: str, content: bytes) -> TrainingImage:
        image_id = uuid4().hex
        try:
            with Image.open(BytesIO(content)) as source:
                source.verify()
            with Image.open(BytesIO(content)) as source:
                width, height = source.size
                suffix = _suffix_for_format(source.format)
        except (UnidentifiedImageError, OSError) as exc:
            raise TrainingDataError("upload must be a valid JPEG, PNG, or WebP image") from exc

        now = datetime.now(UTC)
        storage_filename = f"{image_id}{suffix}"
        record = TrainingImage(
            image_id=image_id,
            original_filename=Path(original_filename).name,
            storage_filename=storage_filename,
            image_uri=f"{self._public_prefix}/{storage_filename}",
            width=width,
            height=height,
            split=DatasetSplit.TRAIN,
            layout=ToteLayout.UNKNOWN,
            regions=(),
            ready=False,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            (self._images / storage_filename).write_bytes(content)
            self._write_record(record)
        return record

    def list_images(self) -> tuple[TrainingImage, ...]:
        with self._lock:
            records = [self._read_record(path) for path in self._annotations.glob("*.json")]
        return tuple(sorted(records, key=lambda item: item.created_at, reverse=True))

    def get_image(self, image_id: str) -> TrainingImage:
        path = self._record_path(image_id)
        if not path.is_file():
            raise KeyError(image_id)
        with self._lock:
            return self._read_record(path)

    def update_image(
        self,
        image_id: str,
        *,
        split: DatasetSplit,
        layout: ToteLayout,
        regions: tuple[TrainingRegion, ...],
        ready: bool,
    ) -> TrainingImage:
        with self._lock:
            current = self.get_image(image_id)
            _validate_regions(current, layout, regions, ready)
            updated = replace(
                current,
                split=split,
                layout=layout,
                regions=regions,
                ready=ready,
                updated_at=datetime.now(UTC),
            )
            self._write_record(updated)
            self._schedule_export_rebuild()
            return updated

    def delete_image(self, image_id: str) -> None:
        with self._lock:
            record = self.get_image(image_id)
            (self._images / record.storage_filename).unlink(missing_ok=True)
            self._record_path(image_id).unlink(missing_ok=True)
            self._schedule_export_rebuild()

    def delete_all_images(self) -> int:
        with self._lock:
            records = self.list_images()
            for record in records:
                (self._images / record.storage_filename).unlink(missing_ok=True)
                self._record_path(record.image_id).unlink(missing_ok=True)
            self._schedule_export_rebuild()
            return len(records)

    def _record_path(self, image_id: str) -> Path:
        if not image_id.isalnum():
            raise KeyError(image_id)
        return self._annotations / f"{image_id}.json"

    def _write_record(self, record: TrainingImage) -> None:
        path = self._record_path(record.image_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_record_to_dict(record), indent=2), encoding="utf-8")
        temporary.replace(path)

    def _read_record(self, path: Path) -> TrainingImage:
        return _record_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _schedule_export_rebuild(self) -> None:
        with self._export_condition:
            self._export_requested = True
            self._export_condition.notify()

    def _export_worker(self) -> None:
        while True:
            with self._export_condition:
                while not self._export_requested:
                    self._export_condition.wait()
                self._export_requested = False
                self._export_in_progress = True
                self._last_export_error = None
            try:
                self._rebuild_exports()
            except Exception as exc:  # pragma: no cover - background failure path
                with self._export_condition:
                    self._last_export_error = str(exc)
            finally:
                with self._export_condition:
                    self._export_in_progress = False

    def _rebuild_exports(self) -> None:
        temporary = self._root / ".exports-building"
        if temporary.exists():
            shutil.rmtree(temporary)
        rfdetr_root = temporary / "rfdetr"
        cells_root = temporary / "cells"
        for state in (CellClassification.EMPTY, CellClassification.NON_EMPTY):
            (cells_root / state.value).mkdir(parents=True, exist_ok=True)

        with self._lock:
            records = [record for record in self.list_images() if record.ready]
        for split in DatasetSplit:
            split_records = [record for record in records if record.split is split]
            _write_coco_split(split_records, split, rfdetr_root, self._images)
        for record in records:
            _write_cell_crops(record, self._images, cells_root)

        with self._lock:
            if self._exports.exists():
                shutil.rmtree(self._exports)
            temporary.replace(self._exports)


def _validate_regions(
    image: TrainingImage,
    layout: ToteLayout,
    regions: tuple[TrainingRegion, ...],
    ready: bool,
) -> None:
    for region in regions:
        minimum_points = 2 if region.region_class is RegionClass.DIVIDER else 3
        if len(region.polygon) < minimum_points:
            raise TrainingDataError(
                f"region {region.region_id} requires at least {minimum_points} points"
            )
        if any(
            x < 0 or y < 0 or x > image.width or y > image.height
            for x, y in region.polygon
        ):
            raise TrainingDataError(f"region {region.region_id} extends outside the image")
        if (
            region.region_class is RegionClass.CELL
            and region.cell_state is not None
            and region.cell_state
            not in (CellClassification.EMPTY, CellClassification.NON_EMPTY)
        ):
            raise TrainingDataError("training cells must be labeled EMPTY or NON_EMPTY")

    if not ready:
        return
    expected_cells = {
        ToteLayout.OPEN: 1,
        ToteLayout.TWO_CELL: 2,
        ToteLayout.FOUR_CELL: 4,
    }.get(layout)
    if expected_cells is None:
        raise TrainingDataError("ready images require a known tote layout")
    totes = [region for region in regions if region.region_class is RegionClass.TOTE]
    cells = [region for region in regions if region.region_class is RegionClass.CELL]
    dividers = [region for region in regions if region.region_class is RegionClass.DIVIDER]
    if len(totes) != 1:
        raise TrainingDataError("ready images require exactly one tote region")
    if len(cells) != expected_cells:
        raise TrainingDataError(f"{layout.value} requires {expected_cells} cell regions")
    if layout is ToteLayout.OPEN and totes[0].polygon != cells[0].polygon:
        raise TrainingDataError("OPEN tote and cell A must use the same polygon")
    expected_dividers = {
        ToteLayout.OPEN: 0,
        ToteLayout.TWO_CELL: 1,
        ToteLayout.FOUR_CELL: 2,
    }[layout]
    if len(dividers) != expected_dividers:
        raise TrainingDataError(f"{layout.value} requires {expected_dividers} divider regions")
    if any(cell.cell_id is None or cell.cell_state is None for cell in cells):
        raise TrainingDataError("every ready cell requires a cell ID and state")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise TrainingDataError("cell IDs must be unique within an image")


def _write_coco_split(
    records: list[TrainingImage],
    split: DatasetSplit,
    output_root: Path,
    image_root: Path,
) -> None:
    split_root = output_root / split.value
    split_root.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_index, record in enumerate(sorted(records, key=lambda item: item.image_id), start=1):
        shutil.copy2(image_root / record.storage_filename, split_root / record.storage_filename)
        images.append(
            {
                "id": image_index,
                "file_name": record.storage_filename,
                "width": record.width,
                "height": record.height,
            }
        )
        for region in record.regions:
            if region.region_class is RegionClass.DIVIDER:
                continue
            category_id = 1 if region.region_class is RegionClass.TOTE else 2
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index,
                    "category_id": category_id,
                    "segmentation": [[value for point in region.polygon for value in point]],
                    "bbox": _polygon_bbox(region.polygon),
                    "area": _polygon_area(region.polygon),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tote"}, {"id": 2, "name": "cell"}],
    }
    (split_root / "_annotations.coco.json").write_text(
        json.dumps(coco, indent=2), encoding="utf-8"
    )


def _write_cell_crops(record: TrainingImage, image_root: Path, output_root: Path) -> None:
    with Image.open(image_root / record.storage_filename) as source:
        image = source.convert("RGB")
        cells = [region for region in record.regions if region.region_class is RegionClass.CELL]
        for index, cell in enumerate(cells):
            if cell.cell_state is None:
                continue
            bounds = _integer_bounds(cell.polygon, image.width, image.height)
            crop = image.crop(bounds)
            shifted = [(x - bounds[0], y - bounds[1]) for x, y in cell.polygon]
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).polygon(shifted, fill=255)
            masked = Image.composite(crop, Image.new("RGB", crop.size), mask)
            filename = f"{record.image_id}_{cell.cell_id or index}.png"
            masked.save(output_root / cell.cell_state.value / filename)


def _suffix_for_format(image_format: str | None) -> str:
    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format or "")
    if suffix is None:
        raise TrainingDataError("upload must be a JPEG, PNG, or WebP image")
    return suffix


def _polygon_bbox(polygon) -> list[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _polygon_area(polygon) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                polygon, polygon[1:] + polygon[:1], strict=True
            )
        )
    ) / 2


def _integer_bounds(polygon, width: int, height: int) -> tuple[int, int, int, int]:
    bbox = _polygon_bbox(polygon)
    left = max(0, math.floor(bbox[0]))
    top = max(0, math.floor(bbox[1]))
    right = min(width, math.ceil(bbox[0] + bbox[2]))
    bottom = min(height, math.ceil(bbox[1] + bbox[3]))
    return left, top, right, bottom


def _record_to_dict(record: TrainingImage) -> dict:
    return {
        "imageId": record.image_id,
        "originalFilename": record.original_filename,
        "storageFilename": record.storage_filename,
        "imageUri": record.image_uri,
        "width": record.width,
        "height": record.height,
        "split": record.split.value,
        "layout": record.layout.value,
        "ready": record.ready,
        "createdAt": record.created_at.isoformat(),
        "updatedAt": record.updated_at.isoformat(),
        "regions": [
            {
                "regionId": region.region_id,
                "regionClass": region.region_class.value,
                "polygon": [list(point) for point in region.polygon],
                "cellId": region.cell_id,
                "cellState": region.cell_state.value if region.cell_state else None,
            }
            for region in record.regions
        ],
    }


def _record_from_dict(value: dict) -> TrainingImage:
    return TrainingImage(
        image_id=value["imageId"],
        original_filename=value["originalFilename"],
        storage_filename=value["storageFilename"],
        image_uri=value["imageUri"],
        width=value["width"],
        height=value["height"],
        split=DatasetSplit(value["split"]),
        layout=ToteLayout(value["layout"]),
        regions=tuple(
            TrainingRegion(
                region_id=region["regionId"],
                region_class=RegionClass(region["regionClass"]),
                polygon=tuple(tuple(point) for point in region["polygon"]),
                cell_id=region.get("cellId"),
                cell_state=(
                    CellClassification(region["cellState"])
                    if region.get("cellState")
                    else None
                ),
            )
            for region in value["regions"]
        ),
        ready=value["ready"],
        created_at=datetime.fromisoformat(value["createdAt"]),
        updated_at=datetime.fromisoformat(value["updatedAt"]),
    )
