from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.repositories.user_repo import create_user, get_user_by_email
from app.schemas.user import UserCreate, UserLogin


def register_user(db: Session, payload: UserCreate) -> dict:
    existed = get_user_by_email(db, payload.email)
    if existed:
        raise HTTPException(status_code=400, detail="Email đã được sử dụng")

    user = create_user(db, payload, hash_password(payload.password))
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user": user}


def login_user(db: Session, payload: UserLogin) -> dict:
    user = get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng")

    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user": user}
