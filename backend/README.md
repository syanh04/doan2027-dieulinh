# ⚙️ Khối Backend (Nghiệp vụ AI & Xử lý Máy chủ)

Thư mục này quản lý toàn bộ logic tính toán, pipeline xử lý AI đa luồng, cơ sở dữ liệu và các thuật toán dự báo của hệ thống TrafficVision Enterprise:

## 📁 Cấu trúc thư mục:

```text
backend/
├── core/
│   ├── camera_manager.py      # Điều phối đa luồng CameraWorker, nhận diện YOLOv8 & ByteTrack, tính Q, K, V
│   ├── forecaster.py          # Mô hình dự báo chuỗi thời gian (Holt's Damped Trend, MA) & Backtest (MAE, RMSE, MAPE)
│   ├── traffic_db.py          # Quản trị SQLite traffic_metrics.db, xác thực người dùng, phân quyền, chuỗi số liệu
│   └── alerts.py              # Động cơ phân tích cảnh báo ùn tắc & đề xuất chu kỳ đèn tín hiệu
│
└── __init__.py                # Khởi tạo backend package
```

---

## 🛠️ Hướng dẫn bảo trì & sửa logic backend:
- **Thêm luồng camera / Sửa nhận diện xe / ROI**: Mở `backend/core/camera_manager.py`.
- **Sửa mô hình dự báo / Thêm thuật toán LSTM / Đổi khoảng dự báo**: Mở `backend/core/forecaster.py`.
- **Thêm cột dữ liệu / Quản lý tài khoản / Phân quyền**: Mở `backend/core/traffic_db.py`.
- **Sửa ngưỡng cảnh báo / Khuyến nghị đèn tín hiệu**: Mở `backend/core/alerts.py`.
