
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.tracevision_service import TraceVisionService


router = APIRouter(
    prefix="/process",
    tags=["Processing"],
)


tracevision_service: TraceVisionService | None = None


def set_service(service: TraceVisionService):
    global tracevision_service
    tracevision_service = service


@router.post("/video")
async def process_video(
    file: UploadFile = File(...),
    frame_skip: int = 5,
):
    if tracevision_service is None:
        raise HTTPException(
            status_code=503,
            detail="TraceVision service is not initialized.",
        )

    if frame_skip < 1:
        raise HTTPException(
            status_code=400,
            detail="frame_skip must be >= 1.",
        )

    suffix = (
        Path(file.filename or ".mp4").suffix
        or ".mp4"
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

        result = tracevision_service.process_video(
            video_path=temp_path,
            frame_skip=frame_skip,
        )

        return {
            "status": "completed",
            "result": result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        if temp_path is not None:
            temp_path.unlink(
                missing_ok=True
            )

