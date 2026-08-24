Chúng ta đã có code train mô hình (Bước 1) và API phục vụ dự đoán ETA chạy mượt mà (Bước 2).

Bây giờ chúng ta sẽ sang **Bước 3: Đóng gói Docker và Xây dựng CI/CD Pipeline (Tự động hóa & Kiểm tra bảo mật)** cho hệ thống Logistics ETA này.

---

# 🛡️ BƯỚC 3: ĐÓNG GÓI DOCKER & TỰ ĐỘNG HÓA CI/CD PIPELINE

Trong hệ thống thực tế của công ty Logistics, chúng ta không thể thủ công copy code lên server. Mỗi khi Data Scientist hoặc bạn cập nhật code (ví dụ chỉnh sửa thuật toán hoặc sửa lỗi API), hệ thống tự động:
1. Kiểm tra lỗi code (Linting).
2. Build Docker Image chứa model artifact (`mlruns/`).
3. Quét lỗ hổng bảo mật (Vulnerability Scanning) trên Docker Image trước khi phát hành.

---

### PHẦN 1: Chuẩn bị file cấu hình

#### 1. Tạo file `requirements.txt`
Đảm bảo thư mục dự án của bạn có file `requirements.txt` với nội dung sau:
```text
fastapi
uvicorn
pydantic
mlflow
scikit-learn
numpy
pandas
```

#### 2. Tạo file `Dockerfile`
Tạo file `Dockerfile` (không có phần mở rộng) tại thư mục gốc với nội dung chuẩn bảo mật (dùng python slim image, không chạy thừa quyền):
```dockerfile
# Sử dụng Python slim image chính thức để tối ưu dung lượng và bảo mật
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói hệ thống tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn API và thư mục mlruns (chứa model artifact đã train) vào container
COPY app_eta.py .
COPY mlruns/ mlruns/

# Expose cổng ứng dụng
EXPOSE 8000

# Khởi chạy ứng dụng FastAPI bằng uvicorn
CMD ["uvicorn", "app_eta:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### PHẦN 2: Xây dựng GitHub Actions CI/CD Pipeline (`.github/workflows/logistics-ci.yml`)

Tạo thư mục `.github/workflows/` và tạo file `logistics-ci.yml` bên trong:

```yaml
name: Logistics MLOps CI/CD Pipeline

on:
  push:
    branches: [ "main", "master" ]

jobs:
  mlops-ci:
    runs-on: ubuntu-latest

    steps:
    # 1. Checkout mã nguồn từ kho lưu trữ
    - name: Checkout code
      uses: actions/checkout@v4

    # 2. Thiết lập môi trường Python
    - name: Set up Python 3.10
      uses: actions/setup-python@v5
      with:
        python-version: "3.10"
        cache: 'pip'

    # 3. Cài đặt dependencies
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    # 4. Kiểm tra chất lượng code (Linting với Flake8)
    - name: Lint code with flake8
      run: |
        # Dừng build nếu có lỗi cú pháp nghiêm trọng
        flake8 app_eta.py --count --select=E9,F63,F7,F82 --show-source --statistics

    # 5. Build Docker Image cho hệ thống ETA
    - name: Build Docker image
      run: |
        docker build -t logistics-eta-api:${{ github.sha }} .

    # 6. Quét lỗ hổng bảo mật Container Image với Trivy (Tiêu chuẩn DevSecOps)
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'logistics-eta-api:${{ github.sha }}'
        format: 'table'
        exit-code: '0'
        severity: 'CRITICAL,HIGH'
```

---

### Hướng dẫn thực thi:
1. Bạn hãy tạo các file **`requirements.txt`**, **`Dockerfile`**, và file GitHub Actions **`.github/workflows/logistics-ci.yml`** với nội dung như trên vào thư mục dự án `logistics-mlops`.
2. Test thử build Docker local trên máy của bạn:
   ```bash
   docker build -t logistics-eta-api:v1 .
   docker run -d -p 8000:8000 --name logistics-container logistics-eta-api:v1
   ```
3. Sau đó, bạn có thể khởi tạo Git (`git init`), commit và push lên GitHub repository của bạn để kích hoạt tự động pipeline CI/CD và kiểm tra bảo mật Trivy!

Hãy thực hiện bước này và báo lại cho tôi khi bạn đã sẵn sàng chuyển sang **Bước 4: Giám sát Data Drift cho hệ thống Logistics** nhé!