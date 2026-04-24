# Back-end

FastAPI API cho đăng nhập, đăng bài, lưu MySQL và gọi model sentiment.

## Cấu trúc

```text
back-end/
├── app/
│   ├── api/v1/endpoints/
│   │   ├── auth.py
│   │   ├── post.py
│   │   └── user.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   ├── models/
│   │   ├── post.py
│   │   └── user.py
│   ├── repositories/
│   │   ├── post_repo.py
│   │   └── user_repo.py
│   ├── schemas/
│   │   ├── post.py
│   │   └── user.py
│   ├── services/
│   │   ├── auth_service.py
│   │   └── post_service.py
│   └── main.py
├── main.py
└── uploads/
```

## Chạy backend

Từ thư mục gốc project:

```bash
cd back-end
uvicorn main:app --reload
```

Hoặc chạy trực tiếp app package:

```bash
uvicorn app.main:app --reload
```

API chạy tại:

```text
http://localhost:8000
```

## MySQL

Tạo database trước:

```sql
CREATE DATABASE multimodal_sentiment CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Nếu cần đổi user/password:

```bash
export DATABASE_URL="mysql+pymysql://root:12345678@localhost:3306/multimodal_sentiment"
```
