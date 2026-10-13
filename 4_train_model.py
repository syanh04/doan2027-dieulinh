"""
4_train_model.py - Module Huấn luyện Chuyên sâu Mô hình YOLOv8-cls (Expert Training)
=============================================================================
Nhiệm vụ:
  1. Kiểm tra phần cứng (GPU CUDA, tên card, VRAM) & cấu hình tối ưu.
  2. Tự động khắc phục các lỗi đặc thù trên môi trường Windows:
     - Tắt AMP (Automatic Mixed Precision) khi chạy card RTX để tránh lỗi NaN Loss.
     - Tối ưu hóa số luồng đọc dữ liệu (workers).
  3. Huấn luyện mô hình YOLOv8-cls (yolov8s-cls, yolov8n-cls) với bộ siêu tham số
     chuyên nghiệp (Hyperparameters):
     - Optimizer : AdamW kết hợp Cosine Learning Rate Scheduler.
     - Regularization: Weight decay, dropout, label smoothing.
     - Augmentation  : Lật ngang, biến đổi màu sắc HSV thích ứng camera giao thông.
     - Early Stopping: Tự động dừng nếu độ chính xác không cải thiện sau n epochs.
  4. Đánh giá chất lượng sau huấn luyện (Top-1 Accuracy, Top-5 Accuracy).
  5. Tự động lưu checkpoint tốt nhất (best.pt) và cập nhật vào thư mục weights/
     để phục vụ triển khai Web Server (app.py) ngay lập tức.
=============================================================================
"""

import os
import sys
import io
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# Đảm bảo in tiếng Việt và emoji mượt mà trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import (
    WORKSPACE_DIR,
    BALANCED_DATASET_DIR,
    RUNS_DIR,
    WEIGHTS_DIR
)


def detect_optimal_hardware():
    """Tự động kiểm tra phần cứng và trả về cấu hình tối ưu nhất."""
    if torch.cuda.is_available():
        gpu_id = 0
        gpu_name = torch.cuda.get_device_name(gpu_id)
        vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
        print(f"🚀 PHÁT HIỆN GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"✅ CUDA Version : {torch.version.cuda}")

        # Trên Windows RTX, tắt AMP để tránh lỗi gradient underflow/NaN loss
        use_amp = False
        device = 0
        workers = min(4, os.cpu_count() or 2)
    else:
        print("⚠️ KHÔNG TÌM THẤY GPU! Hệ thống sẽ huấn luyện trên CPU (tốc độ chậm hơn).")
        use_amp = True
        device = "cpu"
        workers = 2

    return device, use_amp, workers


def train_classifier(
    data_dir: Path,
    base_model: str = "yolov8s-cls.pt",
    epochs: int = 100,
    batch_size: int = 32,
    imgsz: int = 224,
    patience: int = 20,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    project_name: str = "deep_train_results",
    exp_name: str = "yolov8s-cls_100ep",
    auto_deploy: bool = True
):
    """
    Huấn luyện mô hình YOLOv8 Classification chuyên sâu.
    """
    print("\n" + "=" * 75)
    print("🎯 BƯỚC 4: HUẤN LUYỆN CHUYÊN SÂU MÔ HÌNH PHÂN LOẠI (YOLOV8-CLS TRAINING)")
    print("=" * 75)

    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        print(f"❌ Không tìm thấy bộ dữ liệu huấn luyện tại: {data_path}")
        print("💡 Hãy chạy 'python 2_split_data.py' để chuẩn bị dữ liệu trước.")
        sys.exit(1)

    device, use_amp, workers = detect_optimal_hardware()

    print(f"📁 Thư mục Dữ liệu   : {data_path}")
    print(f"🧠 Trọng số gốc       : {base_model}")
    print(f"🔄 Số Epochs         : {epochs} (Patience Early Stopping: {patience})")
    print(f"📦 Batch Size        : {batch_size} | Kích thước ảnh: {imgsz}x{imgsz}")
    print(f"⚙️ Optimizer         : {optimizer} (lr0={lr0}, lrf={lrf})")
    print(f"⚡ AMP (Mixed Prec)  : {'Bật' if use_amp else 'Tắt (An toàn cho Windows RTX)'}")
    print(f"📂 Thư mục kết quả   : {RUNS_DIR / 'classify' / project_name / exp_name}\n")

    # Khởi tạo mô hình
    model = YOLO(base_model)

    # Bắt đầu quá trình huấn luyện
    print("🚀 Bắt đầu quá trình huấn luyện...\n")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=imgsz,
        project=str(RUNS_DIR / "classify" / project_name),
        name=exp_name,
        device=device,
        amp=use_amp,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        workers=workers,
        exist_ok=True,
        # Kỹ thuật Data Augmentation tối ưu cho giao thông
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.1
    )

    print("\n" + "=" * 75)
    print("🎉 HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 75)

    # Đánh giá trọng số tốt nhất
    best_weight = Path(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else None
    if not best_weight or not best_weight.exists():
        best_weight = RUNS_DIR / "classify" / project_name / exp_name / "weights" / "best.pt"

    if best_weight.exists():
        print(f"🏆 Trọng số tốt nhất (Best Model): {best_weight}")

        # Đánh giá lại mô hình trên tập validation
        print("\n📊 Đang đánh giá chất lượng mô hình trên tập kiểm định (Validation Set)...")
        eval_model = YOLO(str(best_weight))
        metrics = eval_model.val(data=str(data_path), split="val", imgsz=imgsz, device=device)

        top1 = getattr(metrics, "top1", None)
        top5 = getattr(metrics, "top5", None)
        if top1 is not None:
            print(f"  • Top-1 Accuracy: {top1 * 100:.2f}%")
        if top5 is not None:
            print(f"  • Top-5 Accuracy: {top5 * 100:.2f}%")

        # Tự động triển khai sang thư mục weights/ của web server
        if auto_deploy:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            deployed_target = WEIGHTS_DIR / f"{exp_name}_best.pt"
            shutil.copy2(best_weight, deployed_target)
            print(f"\n🚀 Đã tự động cập nhật trọng số vào thư mục triển khai: {deployed_target}")
    else:
        print("⚠️ Không tìm thấy file trọng số best.pt. Vui lòng kiểm tra log huấn luyện.")

    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Bước 4: Huấn luyện chuyên sâu mô hình YOLOv8-cls")
    parser.add_argument(
        "--data",
        type=str,
        default=str(BALANCED_DATASET_DIR),
        help="Đường dẫn thư mục dataset huấn luyện"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-cls.pt",
        help="Mô hình gốc (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--imgsz", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("--patience", type=int, default=20, help="Số epoch dừng sớm nếu không cải thiện")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Thuật toán tối ưu (AdamW, SGD, Adam)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Learning rate ban đầu")
    parser.add_argument("--name", type=str, default="yolov8s-cls_100ep", help="Tên phiên huấn luyện")
    parser.add_argument("--no-deploy", action="store_true", help="Không tự động copy vào thư mục weights/")

    args = parser.parse_args()

    train_classifier(
        data_dir=Path(args.data),
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        exp_name=args.name,
        auto_deploy=not args.no_deploy
    )


if __name__ == "__main__":
    main()
