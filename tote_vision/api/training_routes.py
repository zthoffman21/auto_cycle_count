from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from tote_vision.adapters.training_dataset import (
    TrainingDataError,
    TrainingDatasetStore,
)
from tote_vision.api.training_schemas import (
    TrainingDeleteResponse,
    TrainingDraftPredictionItem,
    TrainingDraftPredictionRequest,
    TrainingDraftPredictionResponse,
    TrainingImageResponse,
    TrainingImageUpdate,
    TrainingStatusResponse,
)
from tote_vision.application.predict_training_geometry import (
    TrainingGeometryPredictionError,
    TrainingGeometryPredictor,
)

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/images", response_model=list[TrainingImageResponse], response_model_by_alias=True)
async def list_training_images(request: Request) -> list[TrainingImageResponse]:
    store: TrainingDatasetStore = request.app.state.training_store
    images = await asyncio.to_thread(store.list_images)
    return [TrainingImageResponse.from_domain(image) for image in images]


@router.post(
    "/images",
    response_model=list[TrainingImageResponse],
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def upload_training_images(
    request: Request,
    images: Annotated[list[UploadFile], File()],
) -> list[TrainingImageResponse]:
    if len(images) > 100:
        raise HTTPException(status_code=413, detail="upload is limited to 100 images per request")
    store: TrainingDatasetStore = request.app.state.training_store
    created = []
    for upload in images:
        content = await upload.read(request.app.state.max_upload_bytes + 1)
        await upload.close()
        if len(content) > request.app.state.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds upload limit")
        try:
            record = await asyncio.to_thread(
                store.create_image, upload.filename or "upload", content
            )
        except TrainingDataError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        created.append(TrainingImageResponse.from_domain(record))
    return created


@router.post(
    "/images/predict-drafts",
    response_model=TrainingDraftPredictionResponse,
    response_model_by_alias=True,
)
async def predict_draft_training_images(
    request: Request,
    payload: TrainingDraftPredictionRequest,
) -> TrainingDraftPredictionResponse:
    store: TrainingDatasetStore = request.app.state.training_store
    predictor: TrainingGeometryPredictor | None = getattr(
        request.app.state,
        "training_geometry_predictor",
        None,
    )
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="training geometry prediction is not available; configure RF-DETR",
        )
    images = await asyncio.to_thread(store.list_images)
    requested_ids = set(payload.image_ids) if payload.image_ids is not None else None
    selected = [
        image
        for image in images
        if requested_ids is None or image.image_id in requested_ids
    ]
    by_id = {image.image_id: image for image in selected}
    results: list[TrainingDraftPredictionItem] = []

    if requested_ids is not None:
        for missing_id in sorted(requested_ids - set(by_id)):
            results.append(
                TrainingDraftPredictionItem(
                    imageId=missing_id,
                    status="failed",
                    message="training image not found",
                )
            )

    for image in selected:
        if image.ready:
            results.append(
                TrainingDraftPredictionItem(
                    imageId=image.image_id,
                    status="skipped",
                    layout=image.layout,
                    regionCount=len(image.regions),
                    message="image is already ready",
                )
            )
            continue
        if image.regions and not payload.overwrite:
            results.append(
                TrainingDraftPredictionItem(
                    imageId=image.image_id,
                    status="skipped",
                    layout=image.layout,
                    regionCount=len(image.regions),
                    message="draft already has geometry",
                )
            )
            continue

        try:
            prediction = await predictor.predict(
                image, store.image_directory / image.storage_filename
            )
            updated = await asyncio.to_thread(
                store.update_image,
                image.image_id,
                split=image.split,
                layout=prediction.layout,
                regions=prediction.regions,
                ready=False,
            )
        except (TrainingGeometryPredictionError, TrainingDataError, FileNotFoundError) as exc:
            results.append(
                TrainingDraftPredictionItem(
                    imageId=image.image_id,
                    status="failed",
                    layout=image.layout,
                    regionCount=len(image.regions),
                    message=str(exc),
                )
            )
            continue

        results.append(
            TrainingDraftPredictionItem(
                imageId=image.image_id,
                status="predicted",
                layout=updated.layout,
                confidence=prediction.confidence,
                regionCount=len(updated.regions),
            )
        )

    return TrainingDraftPredictionResponse(
        predicted=sum(result.status == "predicted" for result in results),
        skipped=sum(result.status == "skipped" for result in results),
        failed=sum(result.status == "failed" for result in results),
        results=results,
    )


@router.delete("/images", response_model=TrainingDeleteResponse, response_model_by_alias=True)
async def delete_all_training_images(request: Request) -> TrainingDeleteResponse:
    store: TrainingDatasetStore = request.app.state.training_store
    deleted = await asyncio.to_thread(store.delete_all_images)
    return TrainingDeleteResponse(deletedImages=deleted)


@router.get(
    "/images/{image_id}",
    response_model=TrainingImageResponse,
    response_model_by_alias=True,
)
async def get_training_image(request: Request, image_id: str) -> TrainingImageResponse:
    store: TrainingDatasetStore = request.app.state.training_store
    try:
        image = await asyncio.to_thread(store.get_image, image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="training image not found") from exc
    return TrainingImageResponse.from_domain(image)


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_image(request: Request, image_id: str) -> None:
    store: TrainingDatasetStore = request.app.state.training_store
    try:
        await asyncio.to_thread(store.delete_image, image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="training image not found") from exc


@router.put(
    "/images/{image_id}",
    response_model=TrainingImageResponse,
    response_model_by_alias=True,
)
async def update_training_image(
    request: Request,
    image_id: str,
    payload: TrainingImageUpdate,
) -> TrainingImageResponse:
    store: TrainingDatasetStore = request.app.state.training_store
    try:
        image = await asyncio.to_thread(
            store.update_image,
            image_id,
            split=payload.split,
            layout=payload.layout,
            regions=tuple(region.to_domain() for region in payload.regions),
            ready=payload.ready,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="training image not found") from exc
    except TrainingDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrainingImageResponse.from_domain(image)


@router.get("/status", response_model=TrainingStatusResponse, response_model_by_alias=True)
async def training_status(request: Request) -> TrainingStatusResponse:
    store: TrainingDatasetStore = request.app.state.training_store
    images = await asyncio.to_thread(store.list_images)
    export_in_progress, export_pending, export_error = store.export_state()
    ready = sum(image.ready for image in images)
    cell_root = store.export_directory / "cells"
    cell_crops = (
        sum(1 for path in cell_root.glob("*/*") if path.is_file())
        if cell_root.exists()
        else 0
    )
    return TrainingStatusResponse(
        totalImages=len(images),
        readyImages=ready,
        draftImages=len(images) - ready,
        exportedCellCrops=cell_crops,
        exportDirectory=str(store.export_directory),
        exportInProgress=export_in_progress,
        exportPending=export_pending,
        exportError=export_error,
        predictionAvailable=getattr(request.app.state, "training_geometry_predictor", None)
        is not None,
    )
