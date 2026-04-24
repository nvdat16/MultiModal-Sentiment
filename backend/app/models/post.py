from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base import Base


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    sentiment = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    positive = Column(Float, nullable=True)
    neutral = Column(Float, nullable=True)
    negative = Column(Float, nullable=True)
    prediction_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="posts")
