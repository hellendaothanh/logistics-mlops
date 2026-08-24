# Logistics MLOps & DevSecOps Pipeline
## 📂 Cấu trúc Thư mục Dự án Hoàn chỉnh
```text
logistics-mlops/
├── .github/
│   └── workflows/
│       └── logistics-ci.yml    # Pipeline tự động hóa CI/CD & Security Scanning
├── .gitignore                  # Cấu hình bỏ qua file rác và model artifacts nặng
├── Dockerfile                  # Đóng gói container an toàn và tối ưu
├── README.md                   # Tài liệu mô tả dự án chính thức
├── app_eta.py                  # API phục vụ dự đoán ETA (Model Serving)
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
└── train_eta.py                # Kịch bản huấn luyện mô hình & Tracking (MLflow)
```

---

## 📄 1. File `.gitignore`
*Dùng để loại bỏ các tệp rác hệ thống, môi trường ảo và thư mục `mlruns/` (vì mô hình sẽ được sinh tự động trong CI).*

```text
# --- Python ---
__pycache__/
*.py[cod]
*$py.class
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/

# --- Virtual Environments ---
.env
.venv
env/
venv/

# --- MLOps: MLflow & Artifacts ---
mlruns/
mlflow.db
*.sqlite

# --- IDE & OS ---
.vscode/
.idea/
.DS_Store
Thumbs.db
```

---

## 📄 2. File `requirements.txt`
*Danh sách các thư viện cần thiết cho cả quá trình huấn luyện và phục vụ API.*

```text
fastapi
uvicorn
pydantic
mlflow
scikit-learn
numpy
pandas
```

---

## 📄 3. File `train_eta.py`
*Kịch bản huấn luyện mô hình dự đoán thời gian giao hàng (ETA) kết hợp MLflow Tracking.*

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_eta_model():
    # 1. Giả lập dữ liệu Logistics thực tế (Dữ liệu từ WMS / GPS)
    np.random.seed(42)
    n_samples = 1500
    
    distance = np.random.uniform(5, 100, n_samples)       # Khoảng cách (km)
    stops = np.random.randint(1, 10, n_samples)            # Số điểm dừng
    
    # Công thức giả định tính thời gian: Thời gian = (Khoảng cách * 2.5) + (Số điểm dừng * 15) + Nhiễu
    time_minutes = (distance * 2.5) + (stops * 15) + np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'distance': distance,
        'stops': stops,
        'time_minutes': time_minutes
    })
    
    X = df[['distance', 'stops']]
    y = df['time_minutes']
    
    # 2. Chia tập Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. MLflow Experiment Tracking
    mlflow.set_experiment("logistics_eta_prediction")
    
    with mlflow.start_run() as run:
        print(f"--- Đang huấn luyện mô hình ETA với Run ID: {run.info.run_id} ---")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print(f"Kết quả đánh giá mô hình:")
        print(f" - MAE: {mae:.2f} phút")
        print(f" - RMSE: {rmse:.2f} phút")
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.sklearn.log_model(model, "eta_model")
        print("Huấn luyện hoàn tất và đã lưu model artifact vào MLflow!")

if __name__ == "__main__":
    train_eta_model()
```

---

## 📄 4. File `app_eta.py`
*Ứng dụng FastAPI phục vụ dự đoán ETA trực tuyến (Model Serving).*

```python
import glob
import os
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Logistics ETA Prediction API",
    description="API dự đoán thời gian giao hàng dự kiến (ETA) cho hệ thống Logistics",
    version="1.0.0"
)

def load_latest_mlflow_model():
    model_config_files = glob.glob("mlruns/**/artifacts/**/MLmodel", recursive=True)
    
    if not model_config_files:
        raise RuntimeError(
            f"Không tìm thấy model artifact trong thư mục mlruns tại: {os.getcwd()}. "
            "Hãy chắc chắn bạn đã chạy 'python train_eta.py' trước!"
        )
    
    latest_mlmodel_path = sorted(model_config_files)[-1]
    model_dir = os.path.dirname(latest_mlmodel_path)
    
    print(f"Đang tải mô hình ETA từ: {model_dir}")
    return mlflow.sklearn.load_model(model_dir)

model = load_latest_mlflow_model()

class ETARequest(BaseModel):
    distance_km: float = Field(..., gt=0, description="Khoảng cách giao hàng tính bằng km")
    stops: int = Field(..., ge=0, description="Số điểm dừng giao hàng")

@app.get("/")
def home():
    return {"message": "Logistics ETA MLOps Service is running securely!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict-eta")
def predict_eta(data: ETARequest):
    try:
        features = np.array([[data.distance_km, data.stops]])
        predicted_time = model.predict(features)
        eta_minutes = float(predicted_time[0])
        
        return {
            "distance_km": data.distance_km,
            "stops": data.stops,
            "estimated_time_minutes": round(eta_minutes, 2),
            "estimated_time_hours": round(eta_minutes / 60, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 📄 5. File `Dockerfile`
*Đóng gói ứng dụng FastAPI và mô hình vào Docker Container an toàn.*

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_eta.py .
COPY mlruns/ mlruns/

EXPOSE 8000

CMD ["uvicorn", "app_eta:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📄 6. File `.github/workflows/logistics-ci.yml`
*Cấu hình GitHub Actions CI/CD tự động lint code, train tự động, build Docker và quét bảo mật Trivy.*

```yaml
name: Logistics MLOps CI/CD Pipeline

on:
  push:
    branches: [ "main", "master" ]

jobs:
  mlops-ci:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v5
      with:
        python-version: "3.10"
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 mlflow scikit-learn pandas numpy fastapi uvicorn pydantic
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Lint code with flake8
      run: |
        flake8 app_eta.py train_eta.py --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Run automated model training
      run: |
        python train_eta.py

    - name: Build Docker image
      run: |
        docker build -t logistics-eta-api:${{ github.sha }} .

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'logistics-eta-api:${{ github.sha }}'
        format: 'table'
        exit-code: '0'
        severity: 'CRITICAL,HIGH'
```

---

## 📄 7. File `README.md` (Tài liệu tổng quan dự án)

```markdown
# 🚚 Logistics ETA MLOps & DevSecOps Pipeline

Hệ thống mẫu chuẩn Enterprise phục vụ dự đoán Thời gian giao hàng dự kiến (ETA) cho ngành Logistics, kết hợp quy trình vận hành MLOps tự động hóa và bảo mật từ đầu đến cuối.

## 🚀 Các tính năng chính
- **Experiment Tracking:** Quản lý vòng đời mô hình và ghi nhận chỉ số qua **MLflow**.
- **Model Serving:** Cung cấp REST API hiệu năng cao bằng **FastAPI**.
- **Containerization:** Đóng gói ứng dụng bất biến qua **Docker**.
- **Automated CI/CD & Security:** Tự động huấn luyện, kiểm tra code (Flake8), build Docker và quét lỗ hổng bảo mật (Trivy) thông qua **GitHub Actions**.

## 🛠️ Hướng dẫn Chạy Local
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Huấn luyện mô hình (sinh ra artifact `mlruns/`):
   ```bash
   python train_eta.py
   ```
3. Chạy API Server:
   ```bash
   uvicorn app_eta:app --reload
   ```
4. Truy cập giao diện test API tại: `http://127.0.0.1:8000/docs`
```