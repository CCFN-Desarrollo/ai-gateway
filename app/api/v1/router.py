from fastapi import APIRouter

from app.api.v1.endpoints import health, identity, receipts, validation_cases

api_router = APIRouter()

api_router.include_router(health.router, prefix="", tags=["health"])
api_router.include_router(receipts.router, prefix="/validate", tags=["validation"])
api_router.include_router(identity.router, prefix="/validate", tags=["validation"])
api_router.include_router(validation_cases.router, prefix="", tags=["validation-cases"])
