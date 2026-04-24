from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.post import PostCreateResponse, PostOut
from app.services.post_service import create_user_post, list_posts, predict_sentiment

router = APIRouter(tags=["posts"])


@router.post("/predict")
async def predict(text: str = Form(None), file: UploadFile = File(None)):
    return await predict_sentiment(text, file)


@router.post("/posts", response_model=PostCreateResponse)
async def create_post(
    text: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await create_user_post(db, current_user, text, file)


@router.get("/posts", response_model=list[PostOut])
def get_posts(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return list_posts(db)
