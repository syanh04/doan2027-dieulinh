"""
2_split_data.py - Module Phân chia & Cân bằng Dữ liệu (Data Splitting & Balancing)
=============================================================================
Nhiệm vụ:
  1. Đọc dữ liệu sạch từ thư mục đầu vào.
  2. Thực hiện phân tầng (Stratified Splitting) đảm bảo tỉ lệ giữa các lớp
     đồng đều trên các tập:
     - Tập Huấn luyện (Train): Mặc định 70%
     - Tập Kiểm định   (Val)  : Mặc định 20%
     - Tập Kiểm thử    (Test) : Mặc định 10%
  3. Cơ chế Cân bằng dữ liệu (Class Balancing):
     - Tùy chọn cân bằng số lượng mẫu giữa các lớp nhằm hạn chế tình trạng
       lớp áp đảo (ví dụ xe máy quá nhiều so với xe tải/xe khách).
  4. Cố định Seed ngẫu nhiên để kết quả có thể tái lập (Reproducibility).
  5. Xuất ra cấu trúc thư mục chuẩn định dạng YOLOv8-cls:
     output_dir/
       ├── train/
       │   ├── 0_Xe_may/
       │   ├── 1_O_to_con/
       │   ├── 2_Xe_khach/
       │   └── 3_Xe_tai/
       ├── val/
       └── test/
=============================================================================
"""

import os
import sys
import io
import shutil
import random
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt và emoji mượt mà trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        total = len(iterable) if hasattr(iterable, "__len__") else 0
        for idx, item in enumerate(iterable):
            if total > 0 and (idx % max(1, total // 20) == 0 or idx == total - 1):
                print(f"[{desc}] Tiến trình: {idx + 1}/{total} ({(idx + 1) * 100 // total}%)")
            yield item

from config import DATASETS_DIR, BALANCED_DATASET_DIR

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STANDARD_CLASSES = ["0_Xe_may", "1_O_to_con", "2_Xe_khach", "3_Xe_tai"]


def collect_images_by_class(input_dir: Path):
    """Quét và gom nhóm ảnh theo từng lớp."""
    data_by_class = {cls: [] for cls in STANDARD_CLASSES}

    if not input_dir.exists():
        print(f"❌ Không tìm thấy thư mục đầu vào: {input_dir}")
        return data_by_class

    for cls in STANDARD_CLASSES:
        cls_dir = input_dir / cls
        if cls_dir.exists() and cls_dir.is_dir():
            for f in cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() in VALID_EXTS:
                    data_by_class[cls].append(f)

    # Nếu input_dir có cấu trúc train/val sẵn
    if sum(len(v) for v in data_by_class.values()) == 0:
        for split in ["train", "val", "valid", "test"]:
            split_dir = input_dir / split
            if not split_dir.exists():
                continue
            for cls in STANDARD_CLASSES:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    for f in cls_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in VALID_EXTS:
                            data_by_class[cls].append(f)

    return data_by_class


def split_and_balance(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    balance: bool = True,
    max_per_class: int = None,
    seed: int = 42
):
    """
    Phân chia dữ liệu theo chiến lược phân tầng kèm cân bằng.
    """
    print("\n" + "=" * 70)
    print("✂️ BƯỚC 2: PHÂN CHIA & CÂN BẰNG DỮ LIỆU (DATA SPLITTING & BALANCING)")
    print("=" * 70)
    print(f"📁 Thư mục nguồn      : {input_dir}")
    print(f"📁 Thư mục đích chuẩn : {output_dir}")
    print(f"📊 Tỉ lệ phân chia    : Train={train_ratio*100:.0f}% | Val={val_ratio*100:.0f}% | Test={test_ratio*100:.0f}%")
    print(f"⚖️ Cân bằng dữ liệu   : {'Bật' if balance else 'Tắt'}")
    print(f"🎲 Random Seed        : {seed}\n")

    random.seed(seed)

    # 1. Thu thập ảnh
    class_images = collect_images_by_class(input_dir)
    total_found = sum(len(v) for v in class_images.values())

    if total_found == 0:
        print(f"❌ Không tìm thấy ảnh nào tại {input_dir}!")
        print("💡 Hãy chạy 'python 1_clean_data.py' trước hoặc kiểm tra đường dẫn.")
        return

    print("📊 Số lượng ảnh phát hiện theo từng lớp ban đầu:")
    for cls in STANDARD_CLASSES:
        print(f"  • {cls:<12}: {len(class_images[cls]):>6,} ảnh")
    print(f"  Tổng cộng    : {total_found:>6,} ảnh\n")

    # 2. Xử lý cân bằng dữ liệu nếu được bật
    if balance:
        min_count = min(len(v) for v in class_images.values() if len(v) > 0)
        target_count = min_count
        if max_per_class:
            target_count = min(target_count, max_per_class)
        print(f"⚖️ Áp dụng cân bằng: Giới hạn mỗi lớp tối đa {target_count:,} ảnh để tránh lệch mẫu.")
        for cls in STANDARD_CLASSES:
            random.shuffle(class_images[cls])
            class_images[cls] = class_images[cls][:target_count]
    elif max_per_class:
        for cls in STANDARD_CLASSES:
            random.shuffle(class_images[cls])
            class_images[cls] = class_images[cls][:max_per_class]

    # Chuẩn bị thư mục đích
    splits = ["train", "val"]
    if test_ratio > 0:
        splits.append("test")

    for s in splits:
        for cls in STANDARD_CLASSES:
            (output_dir / s / cls).mkdir(parents=True, exist_ok=True)

    summary = {s: {cls: 0 for cls in STANDARD_CLASSES} for s in splits}
    all_tasks = []

    # 3. Phân chia phân tầng theo từng lớp
    for cls in STANDARD_CLASSES:
        imgs = class_images[cls]
        random.shuffle(imgs)
        n = len(imgs)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio) if test_ratio > 0 else (n - n_train)
        n_test = n - n_train - n_val if test_ratio > 0 else 0

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:] if test_ratio > 0 else []

        split_map = [("train", train_imgs), ("val", val_imgs)]
        if test_ratio > 0:
            split_map.append(("test", test_imgs))

        for s_name, s_imgs in split_map:
            for idx, img_path in enumerate(s_imgs):
                dest_name = f"{cls}_{idx:05d}_{img_path.name}"
                dest_path = output_dir / s_name / cls / dest_name
                all_tasks.append((img_path, dest_path))
                summary[s_name][cls] += 1

    # 4. Sao chép tệp
    print(f"🚚 Đang sao chép {len(all_tasks):,} tệp vào cấu trúc YOLOv8-cls...")
    for src, dst in tqdm(all_tasks, desc="Đang phân bổ dữ liệu"):
        shutil.copy2(src, dst)

    # 5. In bảng kết quả thống kê dạng ma trận
    print("\n" + "-" * 75)
    print("📊 BẢNG THỐNG KÊ CHI TIẾT SAU KHI PHÂN CHIA (DATA SPLIT SUMMARY):")
    print("-" * 75)
    header = f"{'Lớp (Class)':<14} | " + " | ".join(f"{s.upper():>8}" for s in splits) + " | {'TỔNG':>8} | {'TỈ LỆ %':>8}"
    print(header)
    print("-" * 75)

    grand_total = sum(sum(summary[s].values()) for s in splits)
    for cls in STANDARD_CLASSES:
        cls_total = sum(summary[s][cls] for s in splits)
        cls_pct = (cls_total / grand_total * 100) if grand_total > 0 else 0
        row = f"{cls:<14} | " + " | ".join(f"{summary[s][cls]:>8,}" for s in splits) + f" | {cls_total:>8,} | {cls_pct:>7.1f}%"
        print(row)

    print("-" * 75)
    total_row = f"{'TỔNG CỘNG':<14} | " + " | ".join(f"{sum(summary[s].values()):>8,}" for s in splits) + f" | {grand_total:>8,} | {'100.0%':>8}"
    print(total_row)
    print("=" * 75)
    print(f"✅ Hoàn tất! Bộ dữ liệu huấn luyện chuẩn đã sẵn sàng tại: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Bước 2: Phân chia & Cân bằng dữ liệu giao thông")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Thư mục chứa dữ liệu sạch (Mặc định tự tìm cleaned_traffic hoặc raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BALANCED_DATASET_DIR),
        help="Thư mục đích lưu cấu trúc train/val/test"
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Tỉ lệ tập train (mặc định 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Tỉ lệ tập val (mặc định 0.20)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Tỉ lệ tập test (mặc định 0.10, set 0 để chia 70/30)")
    parser.add_argument("--no-balance", action="store_true", help="Không cân bằng mẫu giữa các lớp")
    parser.add_argument("--max-per-class", type=int, default=None, help="Giới hạn tối đa mẫu mỗi lớp")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")

    args = parser.parse_args()

    # Tự động xác định input dir nếu không truyền
    if not args.input_dir:
        cleaned_dir = DATASETS_DIR / "cleaned_traffic"
        if cleaned_dir.exists() and any(cleaned_dir.iterdir()):
            input_path = cleaned_dir
        else:
            # Fallback sang dataset sẵn có
            input_path = DATASETS_DIR / "vietnam_traffic_balanced"
            if not input_path.exists():
                input_path = DATASETS_DIR / "vietnam_traffic_cls"
    else:
        input_path = Path(args.input_dir)

    split_and_balance(
        input_dir=input_path,
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        balance=not args.no_balance,
        max_per_class=args.max_per_class,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
