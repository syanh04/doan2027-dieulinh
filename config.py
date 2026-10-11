"""
config.py - Cấu hình tập trung cho toàn bộ project TrafficVision AI.

Mọi script trong project đều import module này thay vì hardcode đường dẫn
ổ đĩa (D:\\ĐA GTTM, E:\\ĐA GTTM) hoặc hardcode API key.

Cách dùng:
    from config import WORKSPACE_DIR, get_api_key, CLS_WEIGHTS_PATH

Bí quết định cấu hình (theo thứ tự ưu tiên):
    1. Biến môi trường thật của hệ thống (đã set sẵn trong OS/CI)
    2. File .env nằm cạnh config.py (KHÔNG commit file này lên Git)
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ĐƯỜNG DẪN - tự động suy ra từ vị trí file, nên project chạy được ở mọi ổ đĩa
# ---------------------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = WORKSPACE_DIR / "datasets"
RUNS_DIR = WORKSPACE_DIR / "runs"
UPLOAD_FOLDER = WORKSPACE_DIR / "uploads"
WEIGHTS_DIR = WORKSPACE_DIR / "weights"
STATIC_DIR = WORKSPACE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"
TEMPLATES_DIR = WORKSPACE_DIR / "templates"
SURVEY_RESULTS_DIR = WORKSPACE_DIR / "survey_results"

# Dataset dùng cho train / test
BALANCED_DATASET_DIR = DATASETS_DIR / "vietnam_traffic_balanced"
DETECTION_DATASET_DIR = DATASETS_DIR / "vietnam_traffic"

# ---------------------------------------------------------------------------
# TRỌNG SỐ MÔ HÌNH
# ---------------------------------------------------------------------------
CLS_WEIGHTS_PATH = (
    RUNS_DIR / "classify" / "deep_train_results" / "yolov8s-cls_100ep" / "weights" / "best.pt"
)


def resolve_det_weights():
    """Tìm trọng số detection ưu tiên yolov8s.pt, không có thì yolov8n.pt."""
    for name in ("yolov8s.pt", "yolov8n.pt"):
        candidate = WORKSPACE_DIR / name
        if candidate.exists():
            return candidate
    return WORKSPACE_DIR / "yolov8s.pt"


DET_WEIGHTS_PATH = resolve_det_weights()

# ---------------------------------------------------------------------------
# NẠP .env (không bắt buộc phải cài python-dotenv)
# ---------------------------------------------------------------------------
ENV_FILE = WORKSPACE_DIR / ".env"


def load_dotenv(env_file=None):
    """
    Nạp biến từ file .env vào os.environ (không ghi đè biến đã tồn tại).
    Dùng python-dotenv nếu có, tự fallback sang parser đơn giản nếu không.
    """
    path = Path(env_file) if env_file else ENV_FILE
    if not path.exists():
        return False

    try:
        from dotenv import load_dotenv as _load
        _load(dotenv_path=str(path), override=False)
        return True
    except ImportError:
        pass

    # Fallback: parser tự viết, hỗ trợ KEY=VALUE và comment bắt đầu bằng '#'
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    return True


load_dotenv()


# ---------------------------------------------------------------------------
# ROI - HIỆU CHUẨN ĐOẠN ĐƯỜNG (mét) & NGƯỠNG ÙN TẮC
# ---------------------------------------------------------------------------
# Chiều dài đoạn đường trong ROI phụ thuộc vị trí/cao độ camera, KHÔNG phải
# hằng số vật lý. Đặt qua biến môi trường (hoặc .env) để hiệu chuẩn cho đúng:
#   ROI_REAL_LENGTH_M = <đo trên Google Maps / thước dây theo hướng xe chạy>
#   ROI_REAL_WIDTH_M  = <rộng mặt đường thực, ví dụ 12m = 3 làn x 4m>
# Để trống -> dùng giá trị mặc định bên dưới.
def _env_float(name, default):
    try:
        return float(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float_list(name, default):
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return list(default)
    try:
        vals = [float(v) for v in raw.replace(";", ",").split(",") if v.strip() != ""]
        return vals if vals else list(default)
    except ValueError:
        return list(default)


# Kích thước THỰC của đoạn đường nằm trong ROI (mét) - dùng để quy đổi mật độ.
ROI_REAL_WIDTH_M = _env_float("ROI_REAL_WIDTH_M", 12.0)    # chiều rộng mặt đường
ROI_REAL_LENGTH_M = _env_float("ROI_REAL_LENGTH_M", 60.0)  # chiều dài theo hướng xe chạy

# Ngưỡng phân loại ùn tắc: [level2, level3, level4]
DENSITY_THRESHOLDS = _env_float_list(
    "DENSITY_THRESHOLDS", [120.0, 220.0, 350.0])      # xe/km
OCCUPANCY_THRESHOLDS = _env_float_list(
    "OCCUPANCY_THRESHOLDS", [15.0, 30.0, 50.0])        # % diện tích ROI bị xe chiếm

# Trọng số PCU theo loại xe (có thể override, ví dụ "0.4,1.0,2.5,2.0")
_PCU_DEFAULT = [0.5, 1.0, 2.5, 2.0]
_PCU_ENV = _env_float_list("PCU_WEIGHTS", _PCU_DEFAULT)
PCU_WEIGHTS = {
    "xe_may": _PCU_ENV[0] if len(_PCU_ENV) > 0 else _PCU_DEFAULT[0],
    "o_to_con": _PCU_ENV[1] if len(_PCU_ENV) > 1 else _PCU_DEFAULT[1],
    "xe_khach": _PCU_ENV[2] if len(_PCU_ENV) > 2 else _PCU_DEFAULT[2],
    "xe_tai": _PCU_ENV[3] if len(_PCU_ENV) > 3 else _PCU_DEFAULT[3],
}

# Font TTF dùng vẽ chữ tiếng Việt có dấu lên frame (PIL). Ưu tiên font hệ
# thống; nếu không thấy, tự fallback sang DejaVuSans-Bold kèm ultralytics.
VN_FONT_PATHS = [
    str(os.environ.get("VN_FONT_PATH", "") or "").strip(),
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


# ---------------------------------------------------------------------------
# NHẬN DIỆN ĐA TẦM (gần + xa) - chỉnh qua .env, không cần sửa code
# ---------------------------------------------------------------------------
# Lượt 1 (chính): detect/track toàn frame với ngưỡng conf của người dùng.
# Lượt 2 (recall): predict ngưỡng THẤP trên crop vùng ROI để vớt xe xa/nhỏ
# rồi gộp bằng NMS (loại trùng với lượt 1). Tắt bằng DET_RECALL_PASS=0.
def _env_int(name, default):
    try:
        return int(float(str(os.environ.get(name, "")).strip() or default))
    except (TypeError, ValueError):
        return default


def _env_flag(name, default=True):
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


DET_IMGSZ = _env_int("DET_IMGSZ", 1280)            # kích thước infer lượt chính
DET_IOU = _env_float("DET_IOU", 0.5)              # NMS IoU lượt chính
DET_MAX_DET = _env_int("DET_MAX_DET", 300)        # số box tối đa / ảnh
DET_AGNOSTIC_NMS = _env_flag("DET_AGNOSTIC_NMS", True)  # NMS gộp liên lớp
DET_RECALL_PASS = _env_flag("DET_RECALL_PASS", True)   # bật lượt vớt xe xa
DET_RECALL_CONF = _env_float("DET_RECALL_CONF", 0.10)  # ngưỡng thấp lượt recall
DET_RECALL_IMGSZ = _env_int("DET_RECALL_IMGSZ", 960)   # infer trên crop ROI
DET_RECALL_PAD = _env_float("DET_RECALL_PAD", 0.10)    # nới crop ROI 10%
DET_MERGE_IOU = _env_float("DET_MERGE_IOU", 0.45)      # IoU coi là trùng khi gộp

# Lớp COCO được tính là phương tiện (đồng bộ với COCO_VEHICLE_MAP trong app.py).
# 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck. KHÔNG gồm 0=person.
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]


# ---------------------------------------------------------------------------
# API KEY - KHÔNG BAO GIỜ hardcode trong source code
# ---------------------------------------------------------------------------
def get_api_key(var_name="ROBOFLOW_API_KEY", required=False):
    """
    Lấy API key từ biến môi trường (đã nạp từ .env nếu có).

    Args:
        var_name: tên biến môi trường cần đọc.
        required: True -> thoát chương trình kèm hướng dẫn nếu thiếu.

    Returns:
        str: giá trị key, hoặc None nếu không tìm thấy và required=False.
    """
    key = os.environ.get(var_name, "").strip()
    if key:
        return key

    message = (
        f"\n[ERROR] Thiếu biến môi trường '{var_name}'.\n"
        f"        Hãy tạo file '{ENV_FILE.name}' cạnh config.py và điền:\n"
        f"            {var_name}=<your-key>\n"
        f"        (Xem mẫu trong '.env.example')\n"
        f"        Hoặc đặt biến môi trường trước khi chạy:\n"
        f"            set {var_name}=<your-key>   (Windows CMD)\n"
        f"            export {var_name}=<your-key>  (Linux/macOS)\n"
    )
    if required:
        print(message, file=sys.stderr)
        sys.exit(1)
    print("[WARNING] " + message.strip(), file=sys.stderr)
    return None