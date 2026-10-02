from fastapi import APIRouter

from app.api.v2.endpoints import csf

api_router = APIRouter()

api_router.include_router(csf.router, prefix="/validate", tags=["validation"])
