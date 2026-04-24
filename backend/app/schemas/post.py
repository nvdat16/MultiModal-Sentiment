from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.user import UserOut


class PostOut(BaseModel):
    id: int
    content: Optional[str] = None
    image_url: Optional[str] = None
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    positive: Optional[float] = None
    neutral: Optional[float] = None
    negative: Optional[float] = None
    created_at: datetime
    user: UserOut

    class Config:
        from_attributes = True


class PostCreateResponse(BaseModel):
    post: PostOut
    prediction: dict
