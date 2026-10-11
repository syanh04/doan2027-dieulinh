import os
import io
import time
import base64
import numpy as np
import cv2
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

# Đường dẫn & trọng số lấy từ config.py (tự suy ra theo vị trí file -> chạy được ở mọi ổ đĩa)
from config import (
    WORKSPACE_DIR, UPLOAD_FOLDER, CLS_WEIGHTS_PATH, DET_WEIGHTS_PATH,
    ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M,
    DENSITY_THRESHOLDS, OCCUPANCY_THRESHOLDS, PCU_WEIGHTS,
    VN_FONT_PATHS,
    DET_IMGSZ, DET_IOU, DET_MAX_DET, DET_AGNOSTIC_NMS,
    DET_RECALL_PASS, DET_RECALL_CONF, DET_RECALL_IMGSZ, DET_RECALL_PAD, DET_MERGE_IOU,
)

app = Flask(
    __name__,
    template_folder=str(WORKSPACE_DIR / "templates"),
    static_folder=str(WORKSPACE_DIR / "static"),
)

# Load models globally
print("[INFO] Loading YOLO models...")
cls_model = None
det_model = None

if os.path.exists(CLS_WEIGHTS_PATH):
    try:
        cls_model = YOLO(CLS_WEIGHTS_PATH)
        print("[SUCCESS] Loaded Custom Classification Model:", CLS_WEIGHTS_PATH)
    except Exception as e:
        print("[ERROR] Failed to load classification model:", e)

try:
    det_model = YOLO(DET_WEIGHTS_PATH)
    print("[SUCCESS] Loaded Object Detection Model:", DET_WEIGHTS_PATH)
except Exception as e:
    print("[ERROR] Failed to load detection model:", e)

# Mapping COCO vehicle class IDs to Vietnamese labels
COCO_VEHICLE_MAP = {
    2: ("Ô tô con", "o_to_con", (0, 215, 255)),     # car - Gold/Yellow
    3: ("Xe máy", "xe_may", (255, 144, 30)),        # motorcycle - Blue/Cyan
    5: ("Xe khách", "xe_khach", (226, 43, 138)),    # bus - Purple
    7: ("Xe tải", "xe_tai", (0, 69, 255)),          # truck - Red/Orange
    1: ("Xe đạp", "xe_may", (255, 144, 30))         # bicycle -> Xe máy category
}

CLS_VN_MAP = {
    "0_Xe_may": "Xe máy",
    "1_O_to_con": "Ô tô con",
    "2_Xe_khach": "Xe khách",
    "3_Xe_tai": "Xe tải"
}

# =====================================================================
# ROI (VÙNG QUAN TÂM) & HIỆU CHUẨN KHOẢNG CÁCH ĐƯỜNG
# ---------------------------------------------------------------------
# Đa giác ROI dùng toạ độ CHUẨN HOÁ (0..1) nên tự co giãn theo mọi kích
# thước ảnh/video. Thứ tự 4 đỉnh: dưới-trái, dưới-phải, trên-phải, trên-trái
# (dưới = gần camera, trên = xa camera) => tạo hình thang phối cảnh mặt đường.
ROI_NORM = np.array([
    [0.12, 0.99],   # dưới-trái  (gần camera)
    [0.88, 0.99],   # dưới-phải
    [0.66, 0.40],   # trên-phải  (xa camera)
    [0.34, 0.40],   # trên-trái
], dtype=np.float32)

# Kích thước THỰC & ngưỡng lấy từ config.py (có thể override qua .env).
# => Hãy chỉnh ROI_REAL_LENGTH_M cho khớp đoạn đường thực tế của bạn.
#    ROI_REAL_LENGTH_M=60  -> đoạn đường dài 60m hướng xe chạy
#    ROI_REAL_WIDTH_M=12   -> mặt đường rộng 12m
print(f"[INFO] ROI calib: width={ROI_REAL_WIDTH_M:.1f}m "
      f"length={ROI_REAL_LENGTH_M:.1f}m | "
      f"density_thr={DENSITY_THRESHOLDS} occ_thr={OCCUPANCY_THRESHOLDS}")

_DENSITY_KM = ROI_REAL_LENGTH_M / 1000.0        # chiều dài ROI đổi ra km


# ---------------------------------------------------------------------
# FONT TIẾNG VIỆT CHO FRAME (PIL + TTF, thay cv2.putText bị mất dấu)
# ---------------------------------------------------------------------
_VN_FONT_CACHE = {}


def _find_vn_font():
    """Tìm font TTF có dấu tiếng Việt: env -> Windows -> DejaVu của ultralytics."""
    for path in VN_FONT_PATHS:
        if path and os.path.exists(path):
            return path
    try:
        from ultralytics.utils.checks import check_font
        return check_font("DejaVuSans-Bold.ttf")
    except Exception:
        return None


def _vn_font(px):
    """Lấy font PIL theo cỡ pixel (cache theo cỡ, có fallback không dấu nếu mất font)."""
    px = max(8, int(px))
    if px not in _VN_FONT_CACHE:
        path = _find_vn_font()
        try:
            _VN_FONT_CACHE[px] = (ImageFont.truetype(path, px) if path
                                  else ImageFont.load_default())
        except Exception:
            _VN_FONT_CACHE[px] = ImageFont.load_default()
    return _VN_FONT_CACHE[px]


def draw_vn_text(img, text, org, px=20, color=(255, 255, 255), thickness=1):
    """
    Vẽ chữ TIẾNG VIỆT CÓ DẤU lên ảnh BGR (ghi trực tiếp vào img).
    org = (x, y) góc TRÊN-TRÁI của chữ; thickness>1 -> tô đậm thủ công.
    Trả về (w, h) kích thước hộp chữ (tương đương cv2.getTextSize).
    """
    x, y = int(org[0]), int(org[1])
    font = _vn_font(px)
    for dx, dy in ([(0, 0)] if thickness <= 1 else
                   [(dx, dy) for dx in range(thickness) for dy in range(thickness)]):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        d.text((x + dx, y + dy), str(text), font=font,
               fill=(int(color[2]), int(color[1]), int(color[0])))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    try:
        left, top, right, bottom = font.getbbox(str(text))
        return max(0, right - left), max(0, bottom - top)
    except Exception:
        return 0, 0


def measure_vn_text(text, px=20):
    """Đo (w, h) hộp chữ tiếng Việt mà không cần vẽ (dùng khi kẻ tag nền)."""
    font = _vn_font(px)
    try:
        left, top, right, bottom = font.getbbox(str(text))
        return max(0, right - left), max(0, bottom - top)
    except Exception:
        return 0, 0


def draw_vn_texts(img, items):
    """
    Vẽ NHIỀU chữ tiếng Việt trong 1 lần chuyển đổi PIL (hiệu năng cho tag bbox).
    items: list các tuple (text, (x, y), px, color(BGR), thickness).
    """
    if not items:
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, org, px, color, thickness in items:
        font = _vn_font(int(px))
        x, y = int(org[0]), int(org[1])
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        steps = [(0, 0)] if int(thickness) <= 1 else [
            (dx, dy) for dx in range(int(thickness)) for dy in range(int(thickness))]
        for dx, dy in steps:
            d.text((x + dx, y + dy), str(text), font=font, fill=rgb)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def get_roi_pixels(w, h):
    """Trả về 4 đỉnh ROI (pixel) cho ảnh kích thước (w, h)."""
    pts = ROI_NORM.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.int32)


def build_homography(w, h):
    """Ma trận biến đổi phối cảnh: pixel ảnh -> mặt phẳng đường (mét)."""
    src = (ROI_NORM * np.array([w, h], dtype=np.float32)).astype(np.float32)
    dst = np.array([
        [0, 0],
        [ROI_REAL_WIDTH_M, 0],
        [ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M],
        [0, ROI_REAL_LENGTH_M],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def ground_point_m(w, h, x, y, H=None):
    """
    Chuyển 1 điểm ảnh (x, y) -> toạ độ mét (X, Y) trên mặt đường qua homography.
    Y là khoảng cách dọc ĐOẠN ĐƯỜNG tính từ mép gần của ROI.
    """
    if H is None:
        H = build_homography(w, h)
    p = cv2.perspectiveTransform(
        np.array([[[float(x), float(y)]]], dtype=np.float32), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def measure_roi_length_m(w, h):
    """
    Đo chiều dài ROI (mét) BẰNG CHÍNH HOMOGRAPHY: lấy 2 điểm giữa đường ở
    mép gần / mép xa, chuyển sang mét rồi tính khoảng cách. Trả về số dương.
    Đây là con số thực mà mật độ xe/km dùng để chia, không phải số giả định.
    """
    roi_px = get_roi_pixels(w, h)
    near = ((float(roi_px[0, 0]) + float(roi_px[1, 0])) / 2.0,
            (float(roi_px[0, 1]) + float(roi_px[1, 1])) / 2.0)
    far = ((float(roi_px[2, 0]) + float(roi_px[3, 0])) / 2.0,
           (float(roi_px[2, 1]) + float(roi_px[3, 1])) / 2.0)
    H = build_homography(w, h)
    x1, y1 = ground_point_m(w, h, near[0], near[1], H)
    x2, y2 = ground_point_m(w, h, far[0], far[1], H)
    length = float(np.hypot(x2 - x1, y2 - y1))
    return length if length > 0 else float(ROI_REAL_LENGTH_M)


def solve_roi_length_m(w, h, ref_px_a, ref_px_b, ref_real_m,
                       lo=1.0, hi=500.0, iters=40):
    """
    Hiệu chuẩn KHÔNG VÒNG TRÒN: tìm L (mét) sao cho khoảng cách homography
    giữa 2 điểm ảnh tham chiếu đúng bằng khoảng cách thật đã đo (mét).
    Dùng khi bạn đo được 1 đoạn chuẩn trên đường (vạch kẻ liền, đuôi 2 xe...).
    Không dùng ở luồng realtime, chỉ gọi 1 lần khi hiệu chuẩn, ví dụ:
        L = solve_roi_length_m(1280, 720, (640, 700), (640, 300), 25.0)
        # -> đặt ROI_REAL_LENGTH_M=L trong .env
    """
    global ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M, _DENSITY_KM
    ax, ay = float(ref_px_a[0]), float(ref_px_a[1])
    bx, by = float(ref_px_b[0]), float(ref_px_b[1])
    for _ in range(max(1, int(iters))):
        mid = (lo + hi) / 2.0
        saved = ROI_REAL_LENGTH_M
        ROI_REAL_LENGTH_M = mid
        try:
            H = build_homography(w, h)
            x1, y1 = ground_point_m(w, h, ax, ay, H)
            x2, y2 = ground_point_m(w, h, bx, by, H)
            dist = float(np.hypot(x2 - x1, y2 - y1))
        finally:
            ROI_REAL_LENGTH_M = saved
        if dist < float(ref_real_m):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def point_in_roi(roi_px, x, y):
    """Kiểm tra điểm (x, y) có nằm trong đa giác ROI hay không."""
    return cv2.pointPolygonTest(roi_px, (float(x), float(y)), False) >= 0


def counting_line(w, h):
    """Vạch đếm lưu lượng: đường ngang giữa ROI. Trả về (y_line, x_left, x_right)."""
    roi_px = get_roi_pixels(w, h)
    y_line = int((int(roi_px[:, 1].min()) + int(roi_px[:, 1].max())) / 2)
    x_l = int(np.interp(y_line, [roi_px[3, 1], roi_px[0, 1]], [roi_px[3, 0], roi_px[0, 0]]))
    x_r = int(np.interp(y_line, [roi_px[2, 1], roi_px[1, 1]], [roi_px[2, 0], roi_px[1, 0]]))
    return y_line, x_l, x_r


def _pseudo_roi_slots(w, h, cols=6, rows=4):
    """
    Chia lưới điểm tiếp đất NẰM TRONG ROI cho chế độ classification (không bbox).
    Trả về danh sách (cx, cy) đã trộn, dùng để rải khối xe ước lượng.
    """
    roi_px = get_roi_pixels(w, h)
    xs = np.linspace(float(roi_px[:, 0].min()), float(roi_px[:, 0].max()), cols + 2)[1:-1]
    ys = np.linspace(float(roi_px[:, 1].min()), float(roi_px[:, 1].max()), rows + 2)[1:-1]
    slots = [(float(x), float(y)) for y in ys for x in xs
             if point_in_roi(roi_px, float(x), float(y))]
    slots.sort(key=lambda p: (p[1], abs(p[0] - w / 2.0)))
    return slots


def compute_roi_metrics(det_boxes, w, h):
    """
    Tính các chỉ số giao thông TRONG ROI từ danh sách bounding box.
    det_boxes: list các tuple (x1, y1, x2, y2, cls_id).
    Trả về: counts, roi_count, roi_pcu, occupancy(%), density(xe/km), density_pcu.
    """
    roi_px = get_roi_pixels(w, h)
    roi_area = float(cv2.contourArea(roi_px))
    if roi_area <= 0:
        roi_area = float(w * h)

    # Chiều dài đoạn đường (mét) = ĐO TRỰC TIẾP bằng homography từ kích thước
    # frame hiện tại, thay vì chia cứng theo số giả định.
    roi_length_m = measure_roi_length_m(w, h)
    roi_km = roi_length_m / 1000.0 if roi_length_m > 0 else _DENSITY_KM

    counts = {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0}
    roi_count = 0
    roi_pcu = 0.0
    occupied_area = 0.0

    for (x1, y1, x2, y2, cls_id) in det_boxes:
        if cls_id not in COCO_VEHICLE_MAP:
            continue
        # Điểm tiếp đất = trung điểm cạnh dưới bbox (nơi xe chạm mặt đường)
        cx, cy = (x1 + x2) / 2.0, float(y2)
        if not point_in_roi(roi_px, cx, cy):
            continue

        _, cat_key, _ = COCO_VEHICLE_MAP[cls_id]
        counts[cat_key] += 1
        roi_count += 1
        roi_pcu += PCU_WEIGHTS[cat_key]

        # Diện tích phần bbox nằm trong ROI (đo độ chiếm mặt đường)
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        inter_area, _ = cv2.intersectConvexConvex(rect, roi_px.astype(np.float32))
        occupied_area += max(0.0, inter_area)

    occupancy = (occupied_area / roi_area) * 100.0        # % diện tích ROI
    density = roi_count / roi_km                             # xe/km (chia theo mét ĐO ĐƯỢC)
    density_pcu = roi_pcu / roi_km                           # PCU/km

    return {
        "counts": counts,
        "roi_count": roi_count,
        "roi_pcu": round(roi_pcu, 1),
        "occupancy": round(occupancy, 1),
        "density": round(density, 1),
        "density_pcu": round(density_pcu, 1),
        "roi_area_px": round(roi_area),
        "roi_length_m": round(float(roi_length_m), 1),
    }


def analyze_congestion(roi_count, density, occupancy, flow_vph=None):
    """
    Phân loại mức độ giao thông dựa trên MẬT ĐỘ (xe/km) và ĐỘ CHIẾM MẶT
    ĐƯỜNG (%) trong ROI. Mức nào nặng hơn thì lấy mức đó (nhờ vậy vừa bắt
    được đám đông xe máy nhỏ, vừa bắt được vài xe to chiếm đường).
    """
    if roi_count <= 0:
        return {
            "name": "Không phát hiện phương tiện",
            "level": 0,
            "color": "#94a3b8",
            "bg_color": "rgba(148, 163, 184, 0.15)",
            "badge": "none",
            "description": "Không có phương tiện nào trong vùng ROI (khung mặt đường)."
        }

    # Mức theo mật độ (xe/km)
    if density < DENSITY_THRESHOLDS[0]:
        lvl_density = 1
    elif density < DENSITY_THRESHOLDS[1]:
        lvl_density = 2
    elif density < DENSITY_THRESHOLDS[2]:
        lvl_density = 3
    else:
        lvl_density = 4

    # Mức theo độ chiếm mặt đường (%)
    if occupancy < OCCUPANCY_THRESHOLDS[0]:
        lvl_occ = 1
    elif occupancy < OCCUPANCY_THRESHOLDS[1]:
        lvl_occ = 2
    elif occupancy < OCCUPANCY_THRESHOLDS[2]:
        lvl_occ = 3
    else:
        lvl_occ = 4

    level = max(lvl_density, lvl_occ)

    metric_txt = f"Mật độ {density:.0f} xe/km | Độ chiếm mặt đường {occupancy:.0f}%"
    if flow_vph:
        metric_txt += f" | Lưu lượng {flow_vph:.0f} xe/h"

    presets = {
        1: {"name": "Thông thoáng", "color": "#10b981", "bg_color": "rgba(16, 185, 129, 0.15)",
            "badge": "success",
            "desc": "Giao thông thông thoáng, xe di chuyển thuận lợi và an toàn."},
        2: {"name": "Đông đúc / Mật độ cao", "color": "#f59e0b", "bg_color": "rgba(245, 158, 11, 0.15)",
            "badge": "warning",
            "desc": "Mật độ phương tiện tăng cao, cần giữ khoảng cách an toàn."},
        3: {"name": "Ùn tắc giao thông", "color": "#ef4444", "bg_color": "rgba(239, 68, 68, 0.2)",
            "badge": "danger",
            "desc": "CẢNH BÁO: Ùn tắc cục bộ! Xe di chuyển rất chậm."},
        4: {"name": "Ùn tắc nghiêm trọng", "color": "#dc2626", "bg_color": "rgba(220, 38, 38, 0.3)",
            "badge": "critical",
            "desc": "BÁO ĐỘNG ĐỎ: Ùn tắc nghiêm trọng! Xe hầu như dừng hẳn."},
    }
    p = presets[level]
    return {
        "name": p["name"],
        "level": level,
        "color": p["color"],
        "bg_color": p["bg_color"],
        "badge": p["badge"],
        "description": f"{p['desc']} ({metric_txt})"
    }


def draw_roi_overlay(img, w, h, level):
    """Vẽ vùng ROI + vạch đếm lưu lượng lên ảnh/frame (ghi trực tiếp vào img)."""
    roi_px = get_roi_pixels(w, h)

    if level <= 0:
        color = (150, 150, 150)          # xám - không có xe
    elif level == 1:
        color = (0, 200, 0)              # xanh - thông thoáng
    elif level == 2:
        color = (0, 215, 255)            # vàng - đông đúc
    else:
        color = (0, 0, 255)              # đỏ  - ùn tắc

    # Tô mờ vùng ROI + viền
    overlay = img.copy()
    cv2.fillPoly(overlay, [roi_px], color)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    cv2.polylines(img, [roi_px], True, color, 2)

    # Vạch đếm lưu lượng (đường ngang giữa ROI)
    y_line, x_l, x_r = counting_line(w, h)
    cv2.line(img, (x_l, y_line), (x_r, y_line), (0, 255, 255), 2)

    cv2.putText(img, "ROI", (roi_px[3][0] + 6, roi_px[3][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    sample_dir = os.path.join(app.static_folder, "samples")
    samples = []
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                samples.append({
                    "filename": f,
                    "url": f"/static/samples/{f}",
                    "title": f.replace("sample_", "").replace(".jpg", "").replace("_", " ").title()
                })
    return jsonify({"samples": samples})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "accuracy": "97.12%",
        "model_name": "YOLOv8s-cls (100 Epochs) & YOLOv8 Detection",
        "supported_classes": ["Xe máy", "Ô tô con", "Xe khách", "Xe tải"],
        "cls_loaded": cls_model is not None,
        "det_loaded": det_model is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # Check image payload
        image_bytes = None
        if "file" in request.files:
            file = request.files["file"]
            image_bytes = file.read()
        elif request.json and "image_base64" in request.json:
            b64_data = request.json["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            image_bytes = base64.b64decode(b64_data)
        else:
            return jsonify({"success": False, "error": "Không nhận được hình ảnh tải lên"}), 400

        mode = "detection"
        conf_thresh = 0.25

        if request.is_json:
            mode = request.json.get("mode", "detection")
            conf_thresh = float(request.json.get("conf", 0.25))
        else:
            mode = request.form.get("mode", "detection")
            if request.form.get("conf"):
                conf_thresh = float(request.form.get("conf"))

        # Decode image using OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Không thể giải mã định dạng ảnh"}), 400

        h, w = img_bgr.shape[:2]
        annotated_img = img_bgr.copy()
        
        counts = {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0}
        detections_list = []
        det_boxes = []          # (x1, y1, x2, y2, cls_id) để tính chỉ số ROI
        cls_result_data = None

        if mode == "classification" and cls_model is not None:
            # Classification mode using trained model
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = cls_model.predict(source=img_rgb, conf=conf_thresh, verbose=False)[0]
            top1_id = results.probs.top1
            top1_conf = float(results.probs.top1conf)
            class_raw = results.names[top1_id]
            class_vn = CLS_VN_MAP.get(class_raw, class_raw)

            probs_dict = {}
            for idx, p in enumerate(results.probs.data):
                c_name = results.names[idx]
                probs_dict[CLS_VN_MAP.get(c_name, c_name)] = round(float(p) * 100, 2)

            # Draw banner on image (PIL để giữ dấu tiếng Việt)
            label_text = f"{class_vn}: {top1_conf*100:.1f}%"
            cv2.rectangle(annotated_img, (0, 0), (w, 50), (20, 20, 20), -1)
            draw_vn_text(annotated_img, label_text, (20, 12), px=24,
                         color=(0, 255, 127), thickness=2)

            # Increment count
            if "may" in class_raw.lower(): counts["xe_may"] += 1
            elif "to" in class_raw.lower(): counts["o_to_con"] += 1
            elif "khach" in class_raw.lower(): counts["xe_khach"] += 1
            elif "tai" in class_raw.lower(): counts["xe_tai"] += 1

            cls_result_data = {
                "class_name": class_vn,
                "confidence": round(top1_conf * 100, 2),
                "probabilities": probs_dict
            }
        else:
            # Detection mode (Default & multi-vehicle counting)
            if det_model is not None:
                results = det_model.predict(
                    source=img_bgr, conf=conf_thresh, iou=DET_IOU, imgsz=DET_IMGSZ,
                    classes=list(COCO_VEHICLE_MAP.keys()), max_det=DET_MAX_DET,
                    agnostic_nms=DET_AGNOSTIC_NMS, verbose=False)[0]

                main_boxes = []
                if results.boxes is not None and len(results.boxes) > 0:
                    main_boxes = [[float(v) for v in row] for row in results.boxes.xyxy.cpu().numpy()]
                recall_boxes = _recall_boxes_on_roi(
                    img_bgr, conf_thresh, list(COCO_VEHICLE_MAP.keys()),
                    DET_RECALL_IMGSZ, DET_MAX_DET, DET_MERGE_IOU, main_boxes) if DET_RECALL_PASS else []
                
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id in COCO_VEHICLE_MAP:
                        label_vn, cat_key, color = COCO_VEHICLE_MAP[cls_id]
                        counts[cat_key] += 1
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw sleek bounding box
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label header (đo + vẽ bằng chữ Việt, không vỡ dấu)
                        tag_text = f"{label_vn} {conf*100:.0f}%"
                        t_w, t_h = measure_vn_text(tag_text, px=15)
                        cv2.rectangle(annotated_img, (x1, max(0, y1 - 24)), (x1 + t_w + 12, y1), color, -1)
                        draw_vn_text(annotated_img, tag_text, (x1 + 6, max(2, y1 - 22)),
                                     px=15, color=(255, 255, 255), thickness=1)
                        
                        detections_list.append({
                            "class_name": label_vn,
                            "confidence": round(conf * 100, 1),
                            "box": [x1, y1, x2, y2]
                        })
                        det_boxes.append((x1, y1, x2, y2, cls_id))

                # Lượt recall: thêm xe xa/nhỏ (tag "+xa") vào cùng vẽ + đếm
                for (gx1, gy1, gx2, gy2, cls_id, conf, _tid) in recall_boxes:
                    label_vn, cat_key, color = COCO_VEHICLE_MAP[cls_id]
                    counts[cat_key] += 1
                    x1, y1, x2, y2 = map(int, (gx1, gy1, gx2, gy2))
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    tag_text = f"{label_vn} {conf*100:.0f}%+xa"
                    t_w, t_h = measure_vn_text(tag_text, px=13)
                    cv2.rectangle(annotated_img, (x1, max(0, y1 - 22)), (x1 + t_w + 12, y1), color, -1)
                    draw_vn_text(annotated_img, tag_text, (x1 + 6, max(2, y1 - 20)),
                                 px=13, color=(255, 255, 255), thickness=1)
                    detections_list.append({
                        "class_name": label_vn,
                        "confidence": round(conf * 100, 1),
                        "box": [x1, y1, x2, y2]
                    })
                    det_boxes.append((x1, y1, x2, y2, cls_id))

        # ---- Chỉ số PCU (Passenger Car Unit) ----
        pcu = (counts["xe_may"] * PCU_WEIGHTS["xe_may"]) + (counts["o_to_con"] * PCU_WEIGHTS["o_to_con"]) \
              + (counts["xe_khach"] * PCU_WEIGHTS["xe_khach"]) + (counts["xe_tai"] * PCU_WEIGHTS["xe_tai"])
        total_vehicles = sum(counts.values())

        # ---- Chỉ số theo VÙNG ROI (mật độ / độ chiếm mặt đường) ----
        if det_boxes:
            roi_metrics = compute_roi_metrics(det_boxes, w, h)
        else:
            # Chế độ phân loại (không có bbox thật): rải "khối xe ước lượng"
            # BÊN TRONG đúng vùng ROI rồi đo bằng cùng 1 hàm. Tránh 2 lỗi cũ:
            # (1) chia tổng xe toàn frame, (2) occupancy luôn bằng 0.
            pseudo_boxes = []
            slots = _pseudo_roi_slots(w, h)
            grid_map = {"xe_may": 3, "o_to_con": 2, "xe_khach": 5, "xe_tai": 7}
            for cat_key in ("xe_may", "o_to_con", "xe_khach", "xe_tai"):
                for _ in range(int(counts.get(cat_key, 0))):
                    if not slots:
                        break
                    cx, cy = slots.pop(0)
                    bw = w * 0.06 if cat_key == "xe_may" else w * 0.16
                    bh = h * 0.10 if cat_key == "xe_may" else h * 0.22
                    pseudo_boxes.append((cx - bw / 2, cy - bh, cx + bw / 2, cy,
                                         grid_map[cat_key]))
            if pseudo_boxes:
                roi_metrics = compute_roi_metrics(pseudo_boxes, w, h)
                roi_metrics["counts"] = counts
            else:
                roi_metrics = compute_roi_metrics([], w, h)

        congestion_info = analyze_congestion(
            roi_metrics["roi_count"], roi_metrics["density"], roi_metrics["occupancy"]
        )

        # Vẽ vùng ROI + vạch đếm lưu lượng lên ảnh kết quả
        draw_roi_overlay(annotated_img, w, h, congestion_info["level"])

        # Convert annotated image to Base64 JPEG
        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            "success": True,
            "mode": mode,
            "annotated_image": annotated_b64,
            "total_vehicles": total_vehicles,
            "counts": {
                "xe_may": counts["xe_may"],
                "o_to_con": counts["o_to_con"],
                "xe_khach": counts["xe_khach"],
                "xe_tai": counts["xe_tai"]
            },
            "pcu_score": round(pcu, 1),
            "roi_count": roi_metrics["roi_count"],
            "density": roi_metrics["density"],
            "occupancy": roi_metrics["occupancy"],
            "roi_length_m": roi_metrics.get("roi_length_m", round(float(ROI_REAL_LENGTH_M), 1)),
            "flow": None,   # Ảnh tĩnh không tính được lưu lượng (xe/giờ)
            "congestion": congestion_info,
            "detections": detections_list,
            "classification": cls_result_data,
            "process_time_ms": elapsed_ms
        })

    except Exception as e:
        print("[ERROR] Exception during prediction:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# Video Upload & Streaming Support
# UPLOAD_FOLDER đã được import sẵn từ config.py
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Giới hạn kích thước upload & vòng lặp stream ---
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024     # 5 GB - trần cứng của Flask
MAX_UPLOAD_MB = 2048                             # giới hạn mềm do ứng dụng đặt ra (2048 MB = 2 GB)
MAX_VIDEO_LOOPS = 3                              # số lần tua video trước khi dừng stream
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.errorhandler(413)
def handle_file_too_large(e):
    """Trả về JSON thay vì HTML khi request vượt quá MAX_CONTENT_LENGTH."""
    return jsonify({
        "success": False,
        "error": f"File vượt quá giới hạn cho phép ({MAX_UPLOAD_MB} MB). "
                 f"Vui lòng dùng video ngắn hoặc nén lại trước khi tải lên."
    }), 413


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Không nhận được file video"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Chưa chọn file video"}), 400

        # Kiểm tra định dạng file (chặn file nhị phân/rủi ro bị upload nhầm)
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXT:
            return jsonify({
                "success": False,
                "error": f"Định dạng '{ext}' không được hỗ trợ. "
                         f"Các định dạng hợp lệ: {', '.join(sorted(ALLOWED_VIDEO_EXT))}"
            }), 400

        filename = f"video_{int(time.time())}_{os.path.basename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Lưu theo lôi để kiểm tra dung lượng thực tế, tránh ghi 1 file quá lớn
        file.save(filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            os.remove(filepath)
            return jsonify({
                "success": False,
                "error": f"Video dung lượng {size_mb:.0f} MB vượt quá giới hạn {MAX_UPLOAD_MB} MB. "
                         f"Vui lòng rút ngắn hoặc nén video trước khi tải lên."
            }), 400

        return jsonify({
            "success": True,
            "filename": filename,
            "size_mb": round(size_mb, 1),
            "url": f"/video_feed/{filename}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

video_metrics_store = {}


def _finalize_stream(filename, reason=""):
    """
    Đánh dấu stream đã kết thúc.

    Giữ nguyên chỉ số cuối cùng trong store (kèm cờ stream_ended) để frontend
    nhận được thông báo và dừng polling, thay vì poll mãi vào 404.
    """
    metrics = video_metrics_store.get(filename)
    if metrics is None:
        metrics = {"total_vehicles": 0, "roi_count": 0,
                   "counts": {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0},
                   "pcu_score": 0.0, "density": 0.0, "occupancy": 0.0, "flow": 0,
                   "roi_length_m": round(float(ROI_REAL_LENGTH_M), 1),
                   "process_time_ms": 0.0}
    metrics["stream_ended"] = True
    metrics["stream_message"] = reason
    video_metrics_store[filename] = metrics


def _reset_tracker():
    """
    Xoá tracker của ByteTrack để video tua vòng lại bắt đầu ID mới.

    ⚠️ BẮT BUỘC phải `del predictor.trackers`, KHÔNG được gán `= None`.

    Lý do (ultralytics >= 8.4):
        on_predict_start() kiểm tra `hasattr(predictor, "trackers") and persist`
        để quyết định dựng lại tracker hay không.
        - Gán None  -> hasattr vẫn True  -> KHÔNG dựng lại
        - on_predict_postprocess_end() gọi `type(predictor.trackers[0])`
          -> None[0] -> TypeError -> SẬP TOÀN BỘ luồng stream.
        - Xoá hẳn attribute -> hasattr False -> ultralytics dựng tracker mới.
    """
    predictor = getattr(det_model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        try:
            del predictor.trackers
        except (AttributeError, TypeError):
            pass


def _iou_xyxy(a, b):
    """IoU 2 box (x1,y1,x2,y2). Tra ve 0.0 neu khong giao nhau."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    aa = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / max(1e-6, aa + bb - inter)


def _recall_boxes_on_roi(frame, conf_thresh, classes, imgsz, max_det, merge_iou, main_boxes):
    """Luot 2 (tam): predict nguong thap tren crop ROI roi gop loai trung luot 1."""
    h, w = frame.shape[:2]
    roi_px = get_roi_pixels(w, h)
    x1 = max(0, int(np.min(roi_px[:, 0])))
    y1 = max(0, int(np.min(roi_px[:, 1])))
    x2 = min(w, int(np.max(roi_px[:, 0])))
    y2 = min(h, int(np.max(roi_px[:, 1])))
    pad_x = int((x2 - x1) * DET_RECALL_PAD)
    pad_y = int((y2 - y1) * DET_RECALL_PAD)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return []
    # 2 crop: toàn ROI (xe gần) + nửa trên ROI (phóng to xe xa ở chân trời).
    regions = [(x1, y1, x2, y2)]
    mid_y = y1 + (y2 - y1) // 2
    if y2 - mid_y >= 32:
        regions.append((x1, mid_y, x2, y2))
    out = []
    for rx1, ry1, rx2, ry2 in regions:
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        res = det_model.predict(
            source=crop, conf=min(conf_thresh, DET_RECALL_CONF), iou=DET_IOU,
            imgsz=imgsz, classes=classes, max_det=max_det,
            agnostic_nms=DET_AGNOSTIC_NMS, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in COCO_VEHICLE_MAP:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
            gx1, gy1, gx2, gy2 = bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1
            # Bỏ box chạm mép crop (vật bị cắt dở, dễ là false-positive).
            m = 3.0
            if gx1 <= rx1 + m or gy1 <= ry1 + m or gx2 >= rx2 - m or gy2 >= ry2 - m:
                continue
            cand = (gx1, gy1, gx2, gy2)
            if any(_iou_xyxy(cand, m) >= merge_iou for m in main_boxes):
                continue
            if any(_iou_xyxy(cand, (ox1, oy1, ox2, oy2)) >= merge_iou
                   for (ox1, oy1, ox2, oy2, _c, _f, _t) in out):
                continue
            out.append((gx1, gy1, gx2, gy2, cid, float(b.conf[0]), None))
    return out


def _detect_frame(frame, conf_thresh):
    """Chạy detection (có tracking) trên 1 frame."""
    return _detect_frame_with_extras(frame, conf_thresh)[0]


def _detect_frame_with_extras(frame, conf_thresh):
    """
    Chạy detection (có tracking) trên 1 frame + lượt recall vớt xe xa/nhỏ.

    Trả về (results, extras), extras là list tuple
    (x1, y1, x2, y2, cls_id, conf, track_id=None) KHÔNG có trong results.
    Bọc try/except để một frame lỗi không làm sập cả luồng stream.
    """
    if det_model is None:
        return None, []
    try:
        results = det_model.track(
            source=frame, persist=True, tracker="bytetrack.yaml",
            classes=list(COCO_VEHICLE_MAP.keys()), conf=conf_thresh,
            iou=DET_IOU, imgsz=DET_IMGSZ, max_det=DET_MAX_DET,
            agnostic_nms=DET_AGNOSTIC_NMS, verbose=False)[0]
    except Exception:
        try:
            results = det_model.predict(
                source=frame, conf=conf_thresh, iou=DET_IOU, imgsz=DET_IMGSZ,
                classes=list(COCO_VEHICLE_MAP.keys()), max_det=DET_MAX_DET,
                agnostic_nms=DET_AGNOSTIC_NMS, verbose=False)[0]
        except Exception as e:
            print(f"[WARNING] Bo qua frame do loi su luong: {e}")
            return None, []
    if not DET_RECALL_PASS:
        return results, []
    try:
        main_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            main_boxes = [[float(v) for v in row] for row in results.boxes.xyxy.cpu().numpy()]
        extras = _recall_boxes_on_roi(
            frame, conf_thresh, list(COCO_VEHICLE_MAP.keys()),
            DET_RECALL_IMGSZ, DET_MAX_DET, DET_MERGE_IOU, main_boxes)
        return results, extras
    except Exception as e:
        print(f"[WARNING] Bo qua luot recall: {e}")
        return results, []


def generate_video_frames(filename, filepath, conf_thresh=0.25):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return

    frame_skip = 2  # Xử lý 1/2 số frame để stream mượt
    frame_count = 0
    loop_count = 0
    stream_start = time.time()

    # Trạng thái đếm lưu lượng (xe cắt qua vạch đếm trong ROI)
    track_last_y = {}
    counted_ids = set()
    crossing_times = deque(maxlen=5000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Hết video -> tua lại từ đầu, nhưng chỉ trong giới hạn cho phép
            if loop_count >= MAX_VIDEO_LOOPS:
                print(f"[INFO] [{filename}] Đã tua video {MAX_VIDEO_LOOPS} lần, kết thúc stream.")
                _finalize_stream(filename, "Đã kết thúc phát video sau "
                                           f"{MAX_VIDEO_LOOPS + 1} lần phát lại.")
                break

            loop_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            track_last_y.clear()
            counted_ids.clear()
            crossing_times.clear()
            _reset_tracker()
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        annotated_frame = frame.copy()
        h_f, w_f = annotated_frame.shape[:2]
        det_boxes = []
        vn_text_queue = []

        if det_model is not None:
            results, recall_extras = _detect_frame_with_extras(frame, conf_thresh)

            if results is None:
                continue   # frame lỗi -> sang frame tiếp theo, không dừng stream

            y_line, x_l, x_r = counting_line(w_f, h_f)
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

                for (xi1, yi1, xi2, yi2), cid, tid in zip(xyxy, clss, ids):
                    xi1, yi1, xi2, yi2 = int(xi1), int(yi1), int(xi2), int(yi2)
                    cid = int(cid)
                    if cid not in COCO_VEHICLE_MAP:
                        continue
                    label_vn, cat_key, color = COCO_VEHICLE_MAP[cid]
                    det_boxes.append((xi1, yi1, xi2, yi2, cid))

                    cv2.rectangle(annotated_frame, (xi1, yi1), (xi2, yi2), color, 2)
                    t_w, t_h = measure_vn_text(label_vn, px=14)
                    cv2.rectangle(annotated_frame, (xi1, max(0, yi1 - 22)), (xi1 + t_w + 10, yi1), color, -1)
                    vn_text_queue.append((label_vn, (xi1 + 5, max(2, yi1 - 20)),
                                          14, (255, 255, 255), 1))

                    # Đếm lưu lượng: xe cắt qua vạch đếm trong ROI
                    if tid is not None:
                        tid = int(tid)
                        cx, cy = (xi1 + xi2) / 2.0, (yi1 + yi2) / 2.0
                        prev_y = track_last_y.get(tid)
                        if prev_y is not None and x_l <= cx <= x_r:
                            if (prev_y < y_line <= cy) or (prev_y > y_line >= cy):
                                if tid not in counted_ids:
                                    counted_ids.add(tid)
                                    crossing_times.append(time.time())
                        track_last_y[tid] = cy

            # Lượt recall: vẽ box từng xe xa/nhỏ thêm vào (tag "+xa"), cùng cho đếm ROI.
            # Đặt sau lượt chính nên không ảnh hưởng tracker/ByteTrack.
            for (gx1, gy1, gx2, gy2, cls_id, conf, _tid) in (recall_extras or []):
                if cls_id not in COCO_VEHICLE_MAP:
                    continue
                label_vn, cat_key, color = COCO_VEHICLE_MAP[cls_id]
                xi1, yi1, xi2, yi2 = int(gx1), int(gy1), int(gx2), int(gy2)
                det_boxes.append((xi1, yi1, xi2, yi2, cls_id))

                cv2.rectangle(annotated_frame, (xi1, yi1), (xi2, yi2), color, 2)
                tag_text = f"{label_vn} {conf*100:.0f}%+xa"
                t_w, t_h = measure_vn_text(tag_text, px=13)
                cv2.rectangle(annotated_frame, (xi1, max(0, yi1 - 22)), (xi1 + t_w + 10, yi1), color, -1)
                vn_text_queue.append((tag_text, (xi1 + 5, max(2, yi1 - 20)),
                                      13, (255, 255, 255), 1))

        # ---- Chỉ số theo ROI (mật độ / độ chiếm mặt đường) ----
        roi_metrics = compute_roi_metrics(det_boxes, w_f, h_f)

        # ---- Lưu lượng (xe/giờ) theo cửa sổ trượt 30 giây ----
        now = time.time()
        window_s = 30.0
        recent = [t for t in crossing_times if now - t <= window_s]
        eff = min(window_s, max(now - stream_start, 1.0))
        flow_vph = len(recent) / eff * 3600.0

        cg = analyze_congestion(roi_metrics["roi_count"], roi_metrics["density"],
                                roi_metrics["occupancy"], flow_vph)
        frame_pcu = sum(roi_metrics["counts"][k] * PCU_WEIGHTS[k] for k in roi_metrics["counts"])

        # Store latest frame metrics for real-time frontend polling
        video_metrics_store[filename] = {
            "total_vehicles": roi_metrics["roi_count"],
            "roi_count": roi_metrics["roi_count"],
            "counts": roi_metrics["counts"],
            "pcu_score": round(frame_pcu, 1),
            "density": roi_metrics["density"],
            "occupancy": roi_metrics["occupancy"],
            "roi_length_m": roi_metrics.get("roi_length_m", round(float(ROI_REAL_LENGTH_M), 1)),
            "flow": round(flow_vph),
            "congestion": cg,
            "process_time_ms": 33.3,
            "stream_ended": False
        }

        # Vẽ ROI + vạch đếm + HUD (HUD dùng chữ Việt có dấu qua PIL)
        draw_roi_overlay(annotated_frame, w_f, h_f, cg["level"])
        hud_text = (f"ROI: {roi_metrics['roi_count']} xe | Mật độ: {roi_metrics['density']:.0f} xe/km"
                    f" | Lưu lượng: {flow_vph:.0f} xe/h | {cg['name'].upper()}")
        cv2.rectangle(annotated_frame, (10, 10), (w_f - 10, 52), (15, 15, 25), -1)
        vn_text_queue.append((hud_text, (20, 20), 18, (0, 242, 254), 2))
        draw_vn_texts(annotated_frame, vn_text_queue)

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed/<filename>")
def video_feed(filename):
    if filename == "sample_traffic.mp4":
        filepath = os.path.join(app.static_folder, "samples", "sample_traffic.mp4")
    else:
        filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return "Video file not found", 404

    conf = float(request.args.get("conf", 0.25))
    return app.response_class(generate_video_frames(filename, filepath, conf),
                              mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/video_metrics/<filename>", methods=["GET"])
def get_video_metrics(filename):
    metrics = video_metrics_store.get(filename)
    if metrics:
        return jsonify({"success": True, "metrics": metrics})
    return jsonify({"success": False, "error": "No metrics available yet"}), 404

if __name__ == "__main__":
    print("\n=======================================================")
    print("🚀 TRAFFIC AI WEB APP SERVER STARTING AT http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
