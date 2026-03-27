from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import verify_api_key
from app.models.requests import ValidationCaseCreateRequest
from app.models.responses import ValidationCaseCreateResponse, ValidationCaseResponse
from app.services.validation_case_service import validation_case_service

router = APIRouter()


@router.post(
    "/validation-cases",
    response_model=ValidationCaseCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a validation case",
    tags=["validation-cases"],
)
async def create_validation_case(
    payload: ValidationCaseCreateRequest,
    _api_key: str = Depends(verify_api_key),
) -> ValidationCaseCreateResponse:
    try:
        return await validation_case_service.create_case(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/validation-cases/{case_id}",
    response_model=ValidationCaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Get validation case status",
    tags=["validation-cases"],
)
async def get_validation_case(
    case_id: str,
    _api_key: str = Depends(verify_api_key),
) -> ValidationCaseResponse:
    response = await validation_case_service.get_case(case_id)
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Validation case not found.",
        )
    return response
