from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from tote_vision.api.schemas import (
    EmptyToteInspectionRequest,
    EmptyToteInspectionResponse,
)
from tote_vision.application.inspect_empty_tote import InspectEmptyTote

router = APIRouter()


def get_inspector(request: Request) -> InspectEmptyTote:
    inspector = request.app.state.inspector
    if inspector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vision inference is not available; configure trained model paths",
        )
    return inspector


@router.get("/health/live", tags=["health"])
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/vision/inspect-empty-tote",
    response_model=EmptyToteInspectionResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    tags=["inspection"],
)
async def inspect_empty_tote(
    payload: EmptyToteInspectionRequest,
    inspector: Annotated[InspectEmptyTote, Depends(get_inspector)],
) -> EmptyToteInspectionResponse:
    result = await inspector.execute(
        payload.to_domain(), classifier_method=payload.classifier_method
    )
    return EmptyToteInspectionResponse.from_domain(result)
