import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

TARGET_API_URL = os.getenv("TARGET_API_URL", "https://hoa-unlicentiated-pablo.ngrok-free.dev/predict")
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:12345678@localhost:3306/multimodal_sentiment")
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
