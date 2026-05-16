from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import MAX_UPLOAD_SIZE, TARGET_API_URL, UPLOAD_DIR
from app.models.user import User
from app.repositories.post_repo import create_post, get_recent_posts


async def call_prediction_api(text: Optional[str], file: Optional[UploadFile]) -> tuple[dict, Optional[str]]:
    files = None
    data = {}
    image_url = None

    if text:
        data["text"] = text

    if file is not None:
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Ảnh vượt quá dung lượng tối đa 5MB")
        extension = Path(file.filename or "image.jpg").suffix or ".jpg"
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}{extension}"
        image_path = UPLOAD_DIR / filename
        image_path.write_bytes(file_content)
        image_url = f"/uploads/{filename}"
        files = {"file": (file.filename, file_content, file.content_type)}

    response = requests.post(
        TARGET_API_URL,
        data=data if data else None,
        files=files,
        timeout=60,
    )
    response.raise_for_status()
    return response.json(), image_url


async def predict_sentiment(text: Optional[str], file: Optional[UploadFile]) -> dict:
    try:
        prediction, _ = await call_prediction_api(text, file)
        return prediction
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi gọi Colab API: {str(exc)}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(exc)}") from exc


async def create_user_post(db: Session, current_user: User, text: Optional[str], file: Optional[UploadFile]) -> dict:
    if not text and file is None:
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung hoặc chọn ảnh")

    try:
        prediction, image_url = await call_prediction_api(text, file)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi gọi Colab API: {str(exc)}") from exc

    post = create_post(
        db,
        user_id=current_user.id,
        content=text,
        image_url=image_url,
        prediction=prediction,
    )
    return {"post": post, "prediction": prediction}


def list_posts(db: Session):
    return get_recent_posts(db)
