# MultiModal Sentiment

Hệ thống phân tích cảm xúc đa phương thức cho **text + image**. Dự án gồm:

- `src/`: pipeline huấn luyện, mô hình và công cụ dự đoán
- `backend/`: FastAPI backend, lưu dữ liệu, xác thực và gọi model
- `frontend/`: giao diện web Vite/React để đăng bài, bình luận và xem kết quả cảm xúc

## Tính năng

- Phân loại cảm xúc từ **text**, **image** hoặc **multimodal**
- Dự đoán 3 lớp: **positive / neutral / negative**
- Đăng bài, bình luận, xem lịch sử bài viết
- Giao diện web có hiển thị xác suất cảm xúc

## Dataset

- Bộ dữ liệu: **MVSA**
- Thư mục: `dataset/MVSA/`
- Nhãn cảm xúc: `positive`, `neutral`, `negative`

## Kết quả thực nghiệm

| Mô hình | Accuracy | F1-score |
|---|---:|---:|
| Image-only | 0.5831 | 0.6001 |
| Text-only | 0.6892 | 0.6870 |
| Multimodal | 0.6952 | 0.6926 |

> Lưu ý: kết quả có thể thay đổi theo cấu hình, seed, preprocessing và checkpoint sử dụng.

## Cấu trúc dự án

```text
MultiModal-Sentiment/
├── frontend/          # React + Vite UI
├── backend/           # FastAPI API
├── src/               # training / inference code
├── dataset/           # MVSA dataset
├── pretrained_models/ # checkpoint đã huấn luyện
├── figs/              # hình minh hoạ
└── notebook/          # notebook EDA / experiments
```

## Yêu cầu môi trường

- Python 3.10+
- Node.js 18+
- `pip` hoặc `venv`

## Cài đặt nhanh

### 1) Cài dependencies Python

```bash
pip install -r requirements.txt
```

### 2) Cài dependencies frontend

```bash
cd frontend
npm install
```

### 3) Cài dependencies backend

```bash
cd backend
pip install -r requirements.txt
```

## Chạy hệ thống

### Backend

```bash
cd backend
uvicorn app.main:app --reload
```

API mặc định chạy tại: `http://localhost:8000`

### Frontend

```bash
cd frontend
npm run dev
```

Frontend mặc định chạy tại: `http://localhost:5173`

## Biến môi trường

### Backend

- `DATABASE_URL`: chuỗi kết nối MySQL
- `TARGET_API_URL`: URL API dự đoán sentiment bên ngoài
- `SECRET_KEY`: key dùng cho JWT

Ví dụ:

```bash
export DATABASE_URL="mysql+pymysql://root:12345678@localhost:3306/multimodal_sentiment"
export TARGET_API_URL="https://your-colab-api/predict"
export SECRET_KEY="change-this-secret-key"
```

### Frontend

- `VITE_API_BASE`: URL backend, mặc định `http://localhost:8000`

Ví dụ:

```bash
export VITE_API_BASE="http://localhost:8000"
```

## Training model

Các lệnh train nằm trong `src/tools/train.py`.

Ví dụ:

```bash
python -m src.tools.train --datapath dataset --num_epoch 5 --batch_size 32 --mode multimodal
```

Các mode hỗ trợ:

- `text`
- `image`
- `multimodal`

## Dự đoán

```bash
python -m src.tools.predict --datapath dataset --model pretrained_models/MultiModal/best_multimodal.pth
```

## Mô hình

Trong `src/model/` có các kiến trúc:

- `TextClassifier`
- `ImageClassifier`
- `MultiModalClassifier`
- `CrossAttentionMultiModalClassifier`

## API chính

- `GET /` — kiểm tra API hoạt động
- `POST /auth/login`
- `POST /auth/register`
- `GET /posts`
- `POST /posts`
- `POST /posts/{id}/comments`
- `GET /posts/{id}/comments`
- `POST /predict`

## Ghi chú

- Ứng dụng hỗ trợ upload ảnh khi đăng bài hoặc bình luận
- Bảng bình luận mặc định hiển thị **positive + neutral**, và có thể bật để xem thêm **negative**
- Nếu dùng API Colab/ngrok, hãy đảm bảo `TARGET_API_URL` trỏ đúng endpoint `/predict`

## Tác giả

- Nguyen Van Dat
