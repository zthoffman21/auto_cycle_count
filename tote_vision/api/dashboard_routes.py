from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from tote_vision.adapters.local_artifacts import (
    LocalArtifactStore,
    UnsupportedImageError,
)
from tote_vision.api.schemas import EmptyToteInspectionResponse
from tote_vision.application.inspect_empty_tote import InspectEmptyTote
from tote_vision.core.models import InspectionRequest

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@router.post("/inspect", response_model=EmptyToteInspectionResponse, response_model_by_alias=True)
async def inspect_uploaded_image(
    request: Request,
    image: Annotated[UploadFile, File()],
    tote_id: Annotated[str, Form()] = "UNASSIGNED",
    inspection_id: Annotated[str | None, Form()] = None,
    station_id: Annotated[str, Form()] = "MANUAL_DASHBOARD",
    camera_id: Annotated[str, Form()] = "UPLOAD",
) -> EmptyToteInspectionResponse:
    resolved_inspection_id = inspection_id or f"DEV-{uuid4().hex[:12].upper()}"
    for field_name, value in (
        ("inspectionId", resolved_inspection_id),
        ("toteId", tote_id),
        ("stationId", station_id),
        ("cameraId", camera_id),
    ):
        if not _SAFE_ID.fullmatch(value):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{field_name} contains unsupported characters",
            )

    content = await _read_upload(image, request.app.state.max_upload_bytes)
    artifact_store: LocalArtifactStore = request.app.state.artifact_store
    try:
        artifact = artifact_store.save_image(
            resolved_inspection_id,
            image.content_type or "application/octet-stream",
            content,
        )
    except UnsupportedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    inspector: InspectEmptyTote = request.app.state.inspector
    result = await inspector.execute(
        InspectionRequest(
            inspection_id=resolved_inspection_id,
            tote_id=tote_id,
            image_uri=artifact.public_uri,
            station_id=station_id,
            camera_id=camera_id,
            captured_at=datetime.now(UTC),
        ),
        comparison=True,
    )
    return EmptyToteInspectionResponse.from_domain(result)


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    content = await upload.read(max_bytes + 1)
    await upload.close()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"image exceeds {max_bytes} byte upload limit",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image is empty",
        )
    return content
