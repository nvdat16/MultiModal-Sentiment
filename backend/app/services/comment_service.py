from typing import Optional

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import MAX_UPLOAD_SIZE
from app.models.user import User
from app.repositories.comment_repo import create_comment, get_comments_by_post
from app.repositories.post_repo import get_post_by_id
from app.services.post_service import predict_sentiment


async def create_user_comment(
    db: Session,
    current_user: User,
    post_id: int,
    content: Optional[str],
    file: Optional[UploadFile],
) -> dict:
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung bình luận")

    post = get_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Bài viết không tồn tại")

    if file is not None:
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Ảnh vượt quá dung lượng tối đa 5MB")
        file.file.seek(0)

    prediction = await predict_sentiment(content.strip(), file)
    comment = create_comment(
        db,
        post_id=post_id,
        user_id=current_user.id,
        content=content.strip(),
        prediction=prediction,
    )
    return {"comment": comment, "prediction": prediction}


def list_post_comments(db: Session, post_id: int, only_positive: bool = True):
    return get_comments_by_post(db, post_id, show_all=not only_positive)
