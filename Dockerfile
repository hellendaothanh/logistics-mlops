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