from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.comment import CommentCreateResponse, CommentOut
from app.services.comment_service import create_user_comment, list_post_comments

router = APIRouter(prefix="/posts", tags=["comments"])


@router.post("/{post_id}/comments", response_model=CommentCreateResponse)
async def create_comment(
    post_id: int,
    content: str = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await create_user_comment(db, current_user, post_id, content, file)


@router.get("/{post_id}/comments", response_model=list[CommentOut])
def get_comments(
    post_id: int,
    only_positive: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_post_comments(db, post_id, only_positive=only_positive)
