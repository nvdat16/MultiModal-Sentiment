from fastapi import APIRouter

from app.api.v1.endpoints import auth, comment, post, user

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(post.router)
api_router.include_router(comment.router)
api_router.include_router(user.router)
