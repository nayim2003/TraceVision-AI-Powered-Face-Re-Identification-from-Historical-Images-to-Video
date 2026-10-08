
from fastapi import FastAPI

from app.api.routes import health
from app.api.routes import results
from app.api.routes import processing

from app.services.tracevision_service import TraceVisionService
from src.pipeline.factory import build_tracevision_pipeline


app = FastAPI(
    title="TraceVision API",
    description=(
        "AI-powered cross-temporal face "
        "re-identification and visual retrieval "
        "research system."
    ),
    version="0.1.0",
)


pipeline = None
tracevision_service = None


app.include_router(health.router)
app.include_router(results.router)
app.include_router(processing.router)


@app.on_event("startup")
def startup_event():

    global pipeline
    global tracevision_service

    pipeline = build_tracevision_pipeline()

    tracevision_service = TraceVisionService(
        pipeline=pipeline
    )

    processing.set_service(
        tracevision_service
    )

    results.set_pipeline(
        pipeline
    )


@app.get("/")
def root():

    return {
        "service": "TraceVision",
        "version": "0.1.0",
        "status": "running",
    }
