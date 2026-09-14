import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.media_processor import MediaProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["media"])
processor = MediaProcessor()

@router.post("/process")
async def process_media(file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        text = await processor.process(content, file.content_type, file.filename)
        return {"text": text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Media processing error")
        raise HTTPException(status_code=500, detail="Internal error")
