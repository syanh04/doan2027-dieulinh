# 🚦 TrafficVision Enterprise - Intelligent Transportation System (ITS)

Hệ thống giám sát, phân tích và dự báo lưu lượng giao thông đa luồng thời gian thực sử dụng **YOLOv8**, **ByteTrack**, và các mô hình chuỗi thời gian (**Holt's Damped Trend**, **Moving Average**).

---

## 📁 Cấu trúc thư mục dự án (Modular Architecture)

Dự án được tái cấu trúc theo mô hình phân lớp rõ ràng, chuyên nghiệp, giúp dễ dàng mở rộng và định vị lỗi (debug/fix bug):

```text
doan2027-dieulinh/
├── app.py                     # Entry point chính khởi chạy Flask Web Server & API
├── config.py                  # Cấu hình hệ thống (ROI, ngưỡng ùn tắc, hằng số PCU)
├── requirements.txt           # Danh sách các thư viện Python phụ thuộc
├── run.bat                    # Script khởi chạy nhanh 1-chạm trên Windows
├── .env.example               # Mẫu thiết lập biến môi trường
├── .gitignore                 # Bỏ qua tệp rác, virtualenv và trọng số nặng
├── README.md                  # Hướng dẫn chi tiết & tài liệu kiến trúc dự án
│
├── core/                      # [CORE DOMAIN] Mã nguồn nghiệp vụ cốt lõi
│   ├── __init__.py            # Khởi tạo core package
│   ├── camera_manager.py      # Đa luồng CameraWorker, nhận diện YOLOv8 & ByteTrack, tính Q, K, V
│   ├── forecaster.py          # Thuật toán dự báo lưu lượng tương lai & đánh giá Backtest (MAE, RMSE, MAPE)
│   ├── traffic_db.py          # Quản trị SQLite, xác thực người dùng, phân quyền, chuỗi thời gian
│   └── alerts.py              # Động cơ phân tích cảnh báo ùn tắc & đề xuất chu kỳ đèn
│
├── templates/                 # [JINJA2 TEMPLATES] Giao diện HTML
│   ├── index.html             # Bảng điều khiển trung tâm 7 Tabs (Tổng quan, Live, Phân tích, Dự báo...)
│   └── login.html             # Màn hình đăng nhập bảo mật (Dark SaaS theme)
│
├── static/                    # [ASSETS] Tệp tĩnh frontend
│   ├── css/
│   │   ├── style.css          # Bố cục giao diện, lưới camera, card KPI, bảng biểu
│   │   └── tokens.css         # Design tokens (Hệ màu sắc, Dark/Light theme, Spacing 8px)
│   ├── js/
│   │   ├── main.js            # Frontend controller (chuyển tab, SSE stream, biểu đồ động Chart.js)
│   │   ├── chart.min.js       # Thư viện Chart.js self-hosted
│   │   └── lucide.min.js      # Bộ icon Lucide SVG self-hosted
│   └── samples/               # Video CCTV mẫu các nút giao Hà Nội & ảnh kiểm thử
│
├── training/                  # [AI TRAINING PIPELINE] Quy trình huấn luyện mô hình YOLOv8
│   ├── README.md              # Hướng dẫn chi tiết pipeline huấn luyện
│   ├── 1_clean_data.py        # Làm sạch và lọc nhãn dữ liệu
│   ├── 2_split_data.py        # Chia tập Train / Val / Test
│   ├── 3_survey_data.py       # Khảo sát phân bố bounding box và lớp xe
│   ├── 4_train_model.py       # Huấn luyện mô hình YOLOv8
│   └── pipeline.py            # Chạy tự động chuỗi huấn luyện từ 1 -> 4
│
├── notebooks/                 # [RESEARCH] Jupyter Notebook nghiên cứu
│   └── Traffic_Congestion_Warning.ipynb   # Khảo sát dữ liệu, phân loại xe và huấn luyện
│
├── scripts/                   # [TOOLS & PIPELINES] Tiện ích dữ liệu & huấn luyện
│   ├── make_test_video.py     # Tạo video thử nghiệm đường phố
│   ├── create_sample_video.py # Sinh video mẫu giả lập
│   ├── diag_video.py          # Kiểm tra codec, FPS, độ phân giải video
│   ├── analyze_time.py        # Thống kê thời gian xử lý frame
│   ├── data_prep.py           # Tiền xử lý dữ liệu huấn luyện
│   ├── data_merge.py          # Hợp nhất các dataset giao thông
│   ├── run_training.py        # Script chạy huấn luyện YOLOv8
│   └── gpu_setup.py           # Kiểm tra CUDA & VRAM card đồ họa
│
├── tests/                     # [TESTS] Kiểm thử tự động
│   ├── run_browser_test.py    # Kiểm thử tự động toàn diện bằng Playwright (chụp ảnh & quay video)
│   ├── test_video.py          # Kiểm thử pipeline đọc video & YOLOv8
│   ├── test_external_image.py # Test nhận diện trên hình ảnh ngoài
│   └── test_external_video.py # Test nhận diện trên video ngoài
│
└── docker/                    # [DEPLOYMENT] Cấu hình Docker & Containerization
    ├── Dockerfile             # Build image ứng dụng ITS
    ├── docker-compose.yml     # Khởi chạy dịch vụ container
    └── .dockerignore          # Bỏ qua tệp khi build Docker image
```

---

## 🛠️ Hướng dẫn định vị & Fix Bug nhanh khi bảo trì

Khi phát sinh lỗi, bạn có thể định vị chính xác vị trí file cần chỉnh sửa:

| Hiện tượng / Mục tiêu sửa đổi | Tệp cần mở để sửa | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Thêm camera mới / Sửa góc quay ROI / Sửa đếm xe** | `core/camera_manager.py` | Tìm `stations_config` để đổi RTSP/file video hoặc chỉnh tọa độ `roi_norm`. |
| **Sửa thuật toán dự báo / Thêm mô hình LSTM / Đổi sai số** | `core/forecaster.py` | Sửa phương thức `predict()` và `_calculate_backtest_metrics()`. |
| **Quản lý tài khoản / Thêm bảng CSDL / Lưu chỉ số** | `core/traffic_db.py` | Quản lý kết nối SQLite `traffic_metrics.db`, xác thực mật khẩu. |
| **Đổi màu sắc / Tăng giảm cỡ chữ / Chỉnh kích thước khung video** | `static/css/tokens.css`<br>`static/css/style.css` | `tokens.css` quản lý biến màu Dark/Light; `style.css` quản lý layout lưới camera. |
| **Sửa biểu đồ động / Thêm tab mới / Sửa lỗi click giao diện** | `static/js/main.js` | Hàm `switchTab()`, `initAllCharts()`, và bộ timer `setInterval()`. |
| **Thêm route API mới cho Flask** | `app.py` | Quản lý các endpoint RESTful và luồng stream MJPEG. |

---

## 🚀 Hướng dẫn khởi chạy dự án

### 1. Chuẩn bị môi trường
```bash
# Tạo môi trường ảo Python 3.10 / 3.11
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell):
.venv\Scripts\Activate.ps1

# Cài đặt thư viện:
pip install -r requirements.txt
```

### 2. File trọng số mô hình YOLOv8
- Đặt file `yolov8s.pt` vào thư mục gốc của dự án.

### 3. Khởi chạy Server
```bash
python app.py
```
*(Hoặc nhấp đúp vào tệp `run.bat` trên Windows)*

### 4. Truy cập hệ thống
- Mở trình duyệt: **`http://127.0.0.1:5000`**
- Đăng nhập:
  - **Tài khoản**: `admin`
  - **Mật khẩu**: `admin@123`
