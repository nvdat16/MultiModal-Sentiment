from sqlalchemy.orm import Session

from app.models.comment import Comment


def create_comment(
    db: Session,
    *,
    post_id: int,
    user_id: int,
    content: str,
    prediction: dict,
) -> Comment:
    comment = Comment(
        post_id=post_id,
        user_id=user_id,
        content=content,
        sentiment=prediction.get("sentiment"),
        confidence=prediction.get("confidence"),
        positive=prediction.get("positive"),
        neutral=prediction.get("neutral"),
        negative=prediction.get("negative"),
        prediction_json=str(prediction),
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


def get_comments_by_post(
    db: Session,
    post_id: int,
    limit: int = 100,
    show_all: bool = False,
) -> list[Comment]:
    query = db.query(Comment).filter(Comment.post_id == post_id)
    if not show_all:
        # Keep positive + neutral, hide negative by default.
        query = query.filter(~Comment.sentiment.ilike("%negative%"))
    return query.order_by(Comment.created_at.asc()).limit(limit).all()
