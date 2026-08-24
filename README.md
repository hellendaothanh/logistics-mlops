# 🏗️ ENTERPRISE MLOPS ARCHITECTURE: HỆ THỐNG DỰ ĐOÁN ETA LOGISTICS

## 1. Sơ đồ Tổng quan Kiến trúc (System Architecture Flow)

```text
[ Data Sources ] (GPS Truck, IoT, WMS, Traffic DB, Weather API)
       │
       ▼ (Ingestion & Storage)
[ Data Lake / Cloud Storage (S3 / MinIO) ] ──> [ Feature Store (Feast / Hopsworks) ]
                                                            │
┌───────────────────────────────────────────────────────────┼────────────────────────────────────────┐
│ MLOPS PIPELINE (Orchestrated by Apache Airflow)           ▼                                        │
│                                                   [ Model Training ] <── (Experiment Tracking: MLflow)
│                                                           │                                        │
│                                                   [ Model Evaluation & Security Gate ]             │
│                                                           │                                        │
│                                                           ▼                                        │
│                                           [ Model Registry (MLflow + S3) ]                         │
└───────────────────────────────────────────────────────────┼────────────────────────────────────────┘
                                                            │
                                                            ▼ (CI/CD Deployment Pipeline)
                                            [ Model Serving Platform (BentoML / Triton on K8s) ]
                                                            │
       ┌────────────────────────────────────────────────────┴────────────────────────────────────┐
       ▼                                                                                         ▼
[ Real-time API Gateway ] (Client App / Driver App)                               [ Batch Inference ] (Nightly Batch Jobs)
       │                                                                                         │
       └──────────────────────────────────┐ ┌────────────────────────────────────────────────────┘
                                          ▼ ▼
                           [ Monitoring & Observability ] (Prometheus, Grafana, Evidently AI)
                                          │
                                          ▼ (Alert / Auto-Trigger)
                               [ Automated Retraining ]
```

---

## 2. Phân định Trách nhiệm (Who builds what?)

Trong hệ thống này, các bên phối hợp với nhau thông qua một **ML Platform** chung:

### 🔬 2.1. Vai trò Data Scientist (DS)
* **Trách nhiệm:** Nghiên cứu dữ liệu, xây dựng và tối ưu hóa thuật toán học máy.
* **Sản phẩm bàn giao:** Các file mã nguồn nghiên cứu (`notebooks/`, `src/features.py`, `src/model.py`), định nghĩa các đặc trưng (Features) và siêu tham số tốt nhất.
* **Không làm:** Không tự deploy code lên production, không tự quản lý hạ tầng cơ sở dữ liệu.

### ⚙️ 2.2. Vai trò MLOps Engineer (Bạn)
* **Trách nhiệm:** Biến code nghiên cứu của DS thành một dây chuyền tự động, an toàn và có khả năng scale.
* **Công việc cụ thể:**
  * **Orchestration:** Viết các DAGs trên **Apache Airflow** để tự động kéo dữ liệu mới, chạy huấn luyện định kỳ.
  * **CI/CD & Security:** Quản lý Git, cấu hình GitHub Actions chạy Unit Test, kiểm tra tính toàn vẹn của dữ liệu (Data Validation) và quét lỗ hổng bảo mật (Trivy) cho Container Image.
  * **Model Serving Infrastructure:** Xây dựng hạ tầng phục vụ mô hình trên cụm **Kubernetes (K8s)** sử dụng BentoML / Triton Inference Server.
  * **Monitoring & Alerting:** Thiết lập hệ thống giám sát Data Drift và Latency bằng Prometheus, Grafana và Evidently AI.

### 🛠️ 2.3. Vai trò Backend / Data Engineer
* **Trách nhiệm:** Xây dựng hạ tầng dữ liệu gốc, cung cấp Data Lake, đường ống truyền dữ liệu thời gian thực (Kafka/Flink) và tích hợp API dự đoán của mô hình vào hệ thống phần mềm quản lý kho/vận tải của công ty.

---

## 3. Chi tiết 5 Tầng của Kiến trúc Hệ thống Logistics MLOps

### Tầng 1: Data & Feature Engineering (Dữ liệu & Kho tính năng)
* **Nguồn dữ liệu:** Dữ liệu GPS xe tải thời gian thực (Kafka), lịch sử đơn hàng từ WMS (Warehouse Management System), dữ liệu thời tiết (OpenWeather API), tình trạng giao thông.
* **Feature Store (Feast):** Lưu trữ các đặc trưng đã được tính toán sẵn (ví dụ: *tốc độ trung bình của tuyến đường trong 7 ngày qua*, *tải trọng xe*). Cả lúc huấn luyện mô hình và lúc dự đoán trực tuyến đều gọi chung một Feature Store này để đảm bảo **không bị lệch dữ liệu (Training-Serving Skew)**.

### Tầng 2: Training & Orchestration (Huấn luyện & Điều phối)
* **Airflow DAG:** Cứ mỗi Chủ Nhật lúc 02:00 sáng, Airflow kích hoạt một pipeline:
  1. Kéo dữ liệu mới nhất từ Feature Store.
  2. Chạy script train của Data Scientist.
  3. Log các chỉ số (MAE, RMSE của dự đoán ETA) lên **MLflow**.
  4. Nếu mô hình mới có độ chính xác cao hơn mô hình đang chạy trên Production, tự động đẩy lên **Model Registry** dưới trạng thái `Staging`.

### Tầng 3: CI/CD & Security Gate (Cổng kiểm soát chất lượng & Bảo mật)
* Trước khi mô hình được đưa lên Production, một **Security & Quality Gate** tự động kiểm tra:
  * **Data Validation:** Dùng *Great Expectations* kiểm tra dữ liệu đầu vào không bị rác hoặc thiếu hụt.
  * **Model Fairness & Bias Check:** Kiểm tra mô hình có dự đoán sai lệch nghiêm trọng đối với một khu vực địa lý cụ thể nào không.
  * **Container Security Scan:** Quét lỗ hổng bảo mật Docker Image chứa mô hình trước khi deploy.

### Tầng 4: Serving & Inference (Phục vụ dự đoán)
* **Real-time Serving:** Mô hình được đóng gói thành microservice bằng **BentoML**, deploy lên cụm **Kubernetes**. Khi tài xế hoặc khách hàng mở app tra cứu đơn hàng, API Gateway gọi đến service này để trả về ETA trong vòng chưa đầy **50 milliseconds**.
* **Batch Inference:** Vào ban đêm, hệ thống chạy một batch job dự đoán trước ETA cho hàng triệu đơn hàng của ngày hôm sau để tối ưu hóa điều phối kho bãi.

### Tầng 5: Monitoring & Governance (Giám sát vận hành & Quản trị)
* **Data Drift Monitoring:** Hệ thống liên tục so sánh phân phối dữ liệu GPS thực tế hôm nay với dữ liệu lúc train. Nếu tài xế gặp tuyến đường mới hoặc thay đổi tốc độ chạy do mùa dịch/lễ hội, hệ thống phát hiện Data Drift.
* **Feedback Loop:** Khi đơn hàng hoàn thành, thời gian giao hàng thực tế (Actual ETA) được ghi nhận lại và đối chiếu với thời gian mô hình dự đoán (Predicted ETA) để tính toán sai số, làm dữ liệu đầu vòng cho chu kỳ train tiếp theo.

---

## 💡 Góc nhìn Tổng kết cho MLOps/DevSecOps
Trong kiến trúc này, bạn — với tư cách là một **MLOps Engineer** — chính là **kiến trúc sư vận hành cây cầu nối liền giữa Khoa học dữ liệu (Data Science) và Hạ tầng sản xuất (Production Infrastructure)**, đảm bảo hệ thống AI hoạt động **mượt mà, tự động, chịu tải cao và an toàn bảo mật**.