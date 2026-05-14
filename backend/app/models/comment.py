from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base import Base


class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    sentiment = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    positive = Column(Float, nullable=True)
    neutral = Column(Float, nullable=True)
    negative = Column(Float, nullable=True)
    prediction_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    post = relationship("Post", back_populates="comments")
    user = relationship("User", back_populates="comments")
