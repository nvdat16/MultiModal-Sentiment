from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multimodal Bridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_API_URL = "https://hoa-unlicentiated-pablo.ngrok-free.dev/predict"

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        files = None
        data = {}

        # text
        if text is not None:
            data["text"] = text

        # image
        if file is not None:
            file_content = await file.read()
            files = {
                "file": (file.filename, file_content, file.content_type)
            }

        print("Forward request lên Colab...")

        response = requests.post(
            TARGET_API_URL,
            data=data if data else None,
            files=files,
            timeout=60
        )

        response.raise_for_status()

        print("✅ Nhận response từ Colab")

        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Lỗi gọi Colab API: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)