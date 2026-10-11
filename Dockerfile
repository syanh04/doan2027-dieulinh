# syntax=docker/dockerfile:1
# =============================================================================
# TrafficVision AI - Web app (Flask + YOLOv8)
#
# Dùng / build:
#   docker compose up --build          -> http://127.0.0.1:5000
#
# Mặc định: torch CPU (image nhẹ, không cần GPU).
# Build GPU (cần WSL2 + NVIDIA Container Toolkit):
#   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
# =============================================================================
FROM python:3.12-slim

# PYTHONUNBUFFERED : log ra ngay khi xem `docker logs`
# VN_FONT_PATH     : font DejaVu có dấu tiếng Việt (trong image không có font
#                    Windows Arial nên config.py cần chỉ đích danh font này)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VN_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

# libgl1, libglib2.0-0, libsm6, libxext6 : cv2 (opencv-python) chạy headless
# ffmpeg            : đọc/ghi video mp4 cho cv2.VideoCapture (video_feed)
# fonts-dejavu-core : vẽ chữ tiếng Việt có dấu lên frame bằng PIL
RUN apt-get update && apt-get install -y --no-cache-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch trước, tách lớp riêng để cache khi chỉ đổi code/requirements.
# Index CPU chỉ chứa torch/torchvision; extra PyPI để pip resolve các
# dependency kèm theo (sympy, filelock, ...).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision \
        --index-url "${TORCH_INDEX}" \
        --extra-index-url https://pypi.org/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chỉ copy những gì runtime cần (không copy cả repo -> image gọn hơn)
COPY app.py config.py congestion_alert.py ./
COPY templates/ ./templates/
COPY static/ ./static/

# Trọng số: detection (yolov8s.pt, fallback yolov8n.pt) + classifier.
# Đường dẫn best.pt PHẢI khớp CLS_WEIGHTS_PATH trong config.py.
COPY yolov8s.pt yolov8n.pt ./
COPY runs/classify/deep_train_results/yolov8s-cls_100ep/weights/best.pt \
     runs/classify/deep_train_results/yolov8s-cls_100ep/weights/best.pt

# Chạy bằng user thường (không phải root); uploads ghi vào volume gắn ngoài
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/uploads \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Model nạp lúc import -> cho 60s khởi động trước khi đánh dấu unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/', timeout=5)"

# Gunicorn:
#   workers=1      BẮT BUỘC - app.py giữ video_metrics_store / deque trong RAM,
#                  nhiều worker sẽ khiến /api/video_metrics trả 404 sai worker.
#   gthread/16     thread cho các stream /video_feed chạy song song.
#   timeout=120    không kill sớm request xử lý video dài.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--worker-class", "gthread", \
     "--workers", "1", \
     "--threads", "16", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "app:app"]
