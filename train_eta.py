import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_eta_model():
    # 1. Giả lập dữ liệu Logistics thực tế (Dữ liệu từ WMS / GPS)
    # Features: [Khoảng cách (km), Số điểm dừng] -> Target: [Thời gian giao hàng (phút)]
    np.random.seed(42)
    n_samples = 1500
    
    distance = np.random.uniform(5, 100, n_samples)       # Khoảng cách từ 5km đến 100km
    stops = np.random.randint(1, 10, n_samples)            # Số điểm dừng từ 1 đến 9 điểm
    
    # Công thức giả định tính thời gian: Thời gian = (Khoảng cách * 2.5 phút) + (Số điểm dừng * 15 phút) + Nhiễu ngẫu nhiên
    time_minutes = (distance * 2.5) + (stops * 15) + np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'distance': distance,
        'stops': stops,
        'time_minutes': time_minutes
    })
    
    X = df[['distance', 'stops']]
    y = df['time_minutes']
    
    # 2. Chia tập dữ liệu Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Bắt đầu phiên MLflow Tracking (Động thái MLOps đầu tiên của bạn)
    mlflow.set_experiment("logistics_eta_prediction")
    
    with mlflow.start_run() as run:
        print(f"--- Đang huấn luyện mô hình ETA với Run ID: {run.info.run_id} ---")
        
        # Huấn luyện mô hình Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Đánh giá mô hình
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print(f"Kết quả đánh giá mô hình:")
        print(f" - MAE (Sai số tuyệt đối trung bình): {mae:.2f} phút")
        print(f" - RMSE (Sai số căn bậc hai bình phương): {rmse:.2f} phút")
        
        # Log tham số và metrics lên MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Lưu mô hình (Model Artifact) vào MLflow
        mlflow.sklearn.log_model(model, "eta_model")
        print("Huấn luyện hoàn tất và đã lưu model artifact vào MLflow!")

if __name__ == "__main__":
    train_eta_model()