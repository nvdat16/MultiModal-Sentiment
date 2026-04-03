from fastapi import FastAPI, UploadFile, File, HTTPException
import requests

app = FastAPI(title="Image Prediction Bridge API")

# URL đích mà bạn cung cấp
TARGET_API_URL = "https://3b21-34-105-78-109.ngrok-free.app/predict"

@app.post("/upload-and-predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        print("Bắt đầu gọi api")
        # 1. Đọc nội dung file gửi lên từ client
        file_content = await file.read()
        print("Xử lý ảnh")
        # 2. Chuẩn bị file để gửi sang Ngrok API
        files = {"file": (file.filename, file_content, file.content_type)}
        print("Gọi lên colab")
        # 3. Thực hiện call API ngoại vi
        response = requests.post(TARGET_API_URL, files=files)
        
        # Kiểm tra lỗi nếu API ngrok không phản hồi tốt
        response.raise_for_status()
        print("Done")
        # 4. Trả kết quả về cho client của bạn
        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi API ngoại vi: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)