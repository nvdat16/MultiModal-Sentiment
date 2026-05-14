from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.user import UserOut


class CommentOut(BaseModel):
    id: int
    post_id: int
    content: str
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    positive: Optional[float] = None
    neutral: Optional[float] = None
    negative: Optional[float] = None
    created_at: datetime
    user: UserOut

    class Config:
        from_attributes = True


class CommentCreateResponse(BaseModel):
    comment: CommentOut
    prediction: dict
