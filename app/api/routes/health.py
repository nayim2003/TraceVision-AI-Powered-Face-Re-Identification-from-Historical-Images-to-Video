
# app/api/routes/health.py

from fastapi import APIRouter

from app.api.schemas import HealthResponse


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
)
def health_check():

    return HealthResponse(
        status="ok",
        service="TraceVision",
        version="0.1.0",
    )

