import glob
import os
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Khởi tạo FastAPI app
app = FastAPI(
    title="Logistics ETA Prediction API",
    description="API dự đoán thời gian giao hàng dự kiến (ETA) cho hệ thống Logistics",
    version="1.0.0"
)

# 2. Hàm tự động tải mô hình mới nhất từ MLflow Artifacts
def load_latest_mlflow_model():
    # Tìm kiếm file MLmodel được lưu trong thư mục mlruns của MLflow
    model_config_files = glob.glob("mlruns/**/artifacts/**/MLmodel", recursive=True)
    
    if not model_config_files:
        raise RuntimeError(
            f"Không tìm thấy model artifact trong thư mục mlruns tại: {os.getcwd()}. "
            "Hãy chắc chắn bạn đã chạy 'python train_eta.py' trước!"
        )
    
    # Lấy mô hình mới nhất dựa trên đường dẫn sắp xếp
    latest_mlmodel_path = sorted(model_config_files)[-1]
    model_dir = os.path.dirname(latest_mlmodel_path)
    
    print(f"Đang tải mô hình ETA từ: {model_dir}")
    return mlflow.sklearn.load_model(model_dir)

# Load model khi khởi động ứng dụng
model = load_latest_mlflow_model()

# 3. Định nghĩa cấu trúc dữ liệu đầu vào (Input Schema validation)
class ETARequest(BaseModel):
    distance_km: float = Field(..., gt=0, description="Khoảng cách giao hàng tính bằng km (phải lớn hơn 0)")
    stops: int = Field(..., ge=0, description="Số điểm dừng giao hàng (lớn hơn hoặc bằng 0)")

@app.get("/")
def home():
    return {"message": "Logistics ETA MLOps Service is running securely!"}

@app.get("/health")
def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống (Dùng cho Docker/Kubernetes Liveness Probe)"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict-eta")
def predict_eta(data: ETARequest):
    """
    Endpoint nhận thông tin khoảng cách và số điểm dừng, 
    trả về thời gian dự kiến giao hàng (phút).
    """
    try:
        # Chuẩn bị dữ liệu đầu vào cho mô hình [distance, stops]
        features = np.array([[data.distance_km, data.stops]])
        
        # Dự đoán thời gian (phút)
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