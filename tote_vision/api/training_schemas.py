from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from tote_vision.core.models import CellClassification, ToteLayout
from tote_vision.core.training_models import (
    DatasetSplit,
    RegionClass,
    TrainingImage,
    TrainingRegion,
)


class TrainingRegionPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    region_id: str = Field(alias="regionId", min_length=1, max_length=128)
    region_class: RegionClass = Field(alias="regionClass")
    polygon: list[tuple[float, float]] = Field(min_length=2)
    cell_id: str | None = Field(default=None, alias="cellId", max_length=16)
    cell_state: CellClassification | None = Field(default=None, alias="cellState")

    def to_domain(self) -> TrainingRegion:
        return TrainingRegion(
            region_id=self.region_id,
            region_class=self.region_class,
            polygon=tuple(self.polygon),
            cell_id=self.cell_id,
            cell_state=self.cell_state,
        )


class TrainingImageUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    split: DatasetSplit
    layout: ToteLayout
    regions: list[TrainingRegionPayload]
    ready: bool = False


class TrainingImageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="imageId")
    original_filename: str = Field(alias="originalFilename")
    image_uri: str = Field(alias="imageUri")
    width: int
    height: int
    split: DatasetSplit
    layout: ToteLayout
    regions: list[TrainingRegionPayload]
    ready: bool
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, image: TrainingImage) -> TrainingImageResponse:
        return cls(
            imageId=image.image_id,
            originalFilename=image.original_filename,
            imageUri=image.image_uri,
            width=image.width,
            height=image.height,
            split=image.split,
            layout=image.layout,
            regions=[
                TrainingRegionPayload(
                    regionId=region.region_id,
                    regionClass=region.region_class,
                    polygon=list(region.polygon),
                    cellId=region.cell_id,
                    cellState=region.cell_state,
                )
                for region in image.regions
            ],
            ready=image.ready,
            createdAt=image.created_at,
            updatedAt=image.updated_at,
        )


class TrainingStatusResponse(BaseModel):
    total_images: int = Field(alias="totalImages")
    ready_images: int = Field(alias="readyImages")
    draft_images: int = Field(alias="draftImages")
    exported_cell_crops: int = Field(alias="exportedCellCrops")
    export_directory: str = Field(alias="exportDirectory")
    export_in_progress: bool = Field(alias="exportInProgress")
    export_pending: bool = Field(alias="exportPending")
    export_error: str | None = Field(default=None, alias="exportError")


class TrainingDeleteResponse(BaseModel):
    deleted_images: int = Field(alias="deletedImages")
