from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from tote_vision.core.models import CellClassification, Polygon, ToteLayout


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class RegionClass(StrEnum):
    TOTE = "tote"
    CELL = "cell"
    DIVIDER = "divider"


@dataclass(frozen=True, slots=True)
class TrainingRegion:
    region_id: str
    region_class: RegionClass
    polygon: Polygon
    cell_id: str | None = None
    cell_state: CellClassification | None = None


@dataclass(frozen=True, slots=True)
class TrainingImage:
    image_id: str
    original_filename: str
    storage_filename: str
    image_uri: str
    width: int
    height: int
    split: DatasetSplit
    layout: ToteLayout
    regions: tuple[TrainingRegion, ...]
    ready: bool
    created_at: datetime
    updated_at: datetime
