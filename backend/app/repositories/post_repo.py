from sqlalchemy.orm import Session

from app.models.post import Post


def create_post(
    db: Session,
    *,
    user_id: int,
    content: str | None,
    image_url: str | None,
    prediction: dict,
) -> Post:
    post = Post(
        user_id=user_id,
        content=content,
        image_url=image_url,
        sentiment=prediction.get("sentiment"),
        confidence=prediction.get("confidence"),
        positive=prediction.get("positive"),
        neutral=prediction.get("neutral"),
        negative=prediction.get("negative"),
        prediction_json=str(prediction),
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def get_recent_posts(db: Session, limit: int = 50) -> list[Post]:
    return db.query(Post).order_by(Post.created_at.desc()).limit(limit).all()
