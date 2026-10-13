"""
1_clean_data.py - Module Làm sạch Dữ liệu Giao thông (Data Cleaning)
=============================================================================
Nhiệm vụ:
  1. Quét toàn bộ hình ảnh thô từ các nguồn dữ liệu (raw datasets).
  2. Kiểm tra tính toàn vẹn của tệp:
     - Loại bỏ file rỗng (0 bytes).
     - Kiểm tra header và cấu trúc ảnh (loại bỏ file corrupt / không đọc được).
     - Kiểm tra kích thước tối thiểu (loại bỏ ảnh quá nhỏ < min_size).
  3. Phát hiện và loại bỏ hình ảnh trùng lặp bằng thuật toán mã băm (MD5 hash).
  4. Chuẩn hóa nhãn (Label Standardization) về 4 lớp giao thông Việt Nam:
     - 0_Xe_may    (Xe máy, mô tô, xe đạp, xe điện)
     - 1_O_to_con  (Ô tô con, xe du lịch, taxi, sedan, SUV)
     - 2_Xe_khach  (Xe khách, xe bus, xe đưa đón, minibus)
     - 3_Xe_tai    (Xe tải, xe container, bán tải, xe tải nặng HGV/LGV)
  5. Xuất tập dữ liệu sạch vào thư mục đích chuẩn bị cho bước chia tập (split).
=============================================================================
"""

import os
import sys
import io
import shutil
import hashlib
import argparse
from pathlib import Path
from PIL import Image

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

from config import DATASETS_DIR, WORKSPACE_DIR

# Định dạng ảnh hợp lệ
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bảng quy đổi nhãn linh hoạt
LABEL_RULES = {
    "0_Xe_may": [
        "motor", "bike", "scooter", "moto", "cycle", "xe_may", "xemay"
    ],
    "1_O_to_con": [
        "passenger-car", "car", "taxi", "suv", "sedan", "coupe",
        "hatchback", "o_to", "oto", "otocon"
    ],
    "2_Xe_khach": [
        "bus", "coach", "tourist-bus", "rta-bus", "school-bus",
        "labor-bus", "medium-bus", "micro-bus", "minibus", "xe_khach", "xekhach"
    ],
    "3_Xe_tai": [
        "truck", "hgv", "lgv", "van", "pickup", "lorry",
        "articulated", "rigid", "xe_tai", "xetai", "container"
    ]
}


def map_label(raw_name: str) -> str:
    """
    Ánh xạ tên thư mục/nhãn thô thành 1 trong 4 lớp chuẩn giao thông Việt Nam.
    """
    clean_name = raw_name.lower().replace("_", "-").replace(" ", "-")

    # Ưu tiên kiểm tra xe khách trước (tránh trường hợp nhầm với car)
    for kw in LABEL_RULES["2_Xe_khach"]:
        if kw in clean_name:
            return "2_Xe_khach"

    # Xe tải
    for kw in LABEL_RULES["3_Xe_tai"]:
        if kw in clean_name:
            return "3_Xe_tai"

    # Xe máy
    for kw in LABEL_RULES["0_Xe_may"]:
        if kw in clean_name:
            return "0_Xe_may"

    # Ô tô con
    for kw in LABEL_RULES["1_O_to_con"]:
        if kw in clean_name:
            return "1_O_to_con"

    # Mặc định an toàn
    return "1_O_to_con"


def compute_file_hash(filepath: Path) -> str:
    """Tính mã băm MD5 để nhận diện các tệp trùng lặp."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image(filepath: Path, min_size: int = 24) -> bool:
    """
    Kiểm tra ảnh có hợp lệ, không bị corrupt và có độ phân giải tối thiểu.
    """
    try:
        if filepath.stat().st_size == 0:
            return False
        with Image.open(filepath) as img:
            img.verify()
        # Mở lại để kiểm tra kích thước và màu
        with Image.open(filepath) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                return False
        return True
    except Exception:
        return False


def clean_dataset(
    raw_dirs: list,
    output_dir: Path,
    min_size: int = 24,
    remove_duplicates: bool = True,
    copy_files: bool = True
):
    """
    Thực hiện quy trình làm sạch dữ liệu toàn diện.
    """
    print("\n" + "=" * 70)
    print("🧹 BƯỚC 1: LÀM SẠCH DỮ LIỆU GIAO THÔNG (DATA CLEANING)")
    print("=" * 70)
    print(f"📁 Thư mục nguồn quét  : {[str(d) for d in raw_dirs]}")
    print(f"📁 Thư mục đích lưu sạch: {output_dir}")
    print(f"⚙️ Kích thước tối thiểu : {min_size}x{min_size}px")
    print(f"⚙️ Lọc trùng lặp        : {'Bật' if remove_duplicates else 'Tắt'}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    for cls in LABEL_RULES.keys():
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    stats = {
        "total_scanned": 0,
        "corrupted_or_too_small": 0,
        "duplicates": 0,
        "valid_saved": 0,
        "by_class": {cls: 0 for cls in LABEL_RULES.keys()}
    }

    # Quét toàn bộ tệp từ các thư mục nguồn
    candidate_files = []
    for r_dir in raw_dirs:
        r_path = Path(r_dir)
        if not r_path.exists():
            print(f"⚠️ Cảnh báo: Thư mục nguồn không tồn tại: {r_path}")
            continue

        for root, dirs, files in os.walk(r_path):
            folder_name = Path(root).name
            # Bỏ qua các thư mục cache, runs, temp
            if folder_name.startswith((".", "__")) or folder_name in ("runs", "cache", "weights"):
                continue

            target_class = map_label(folder_name)
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in VALID_EXTENSIONS:
                    candidate_files.append((p, target_class))

    stats["total_scanned"] = len(candidate_files)
    print(f"🔍 Đã phát hiện tổng cộng {len(candidate_files):,} tệp hình ảnh ứng viên. Bắt đầu kiểm tra...\n")

    if not candidate_files:
        print("❌ Không tìm thấy tệp ảnh nào trong các thư mục nguồn được chỉ định!")
        return stats

    for filepath, target_class in tqdm(candidate_files, desc="Đang làm sạch"):
        # 1. Kiểm tra tính toàn vẹn
        if not verify_image(filepath, min_size=min_size):
            stats["corrupted_or_too_small"] += 1
            continue

        # 2. Kiểm tra trùng lặp
        if remove_duplicates:
            f_hash = compute_file_hash(filepath)
            if f_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(f_hash)

        # 3. Lưu vào thư mục sạch
        dest_filename = f"{target_class}_{stats['by_class'][target_class]:05d}_{filepath.name}"
        dest_path = output_dir / target_class / dest_filename

        if copy_files:
            shutil.copy2(filepath, dest_path)

        stats["valid_saved"] += 1
        stats["by_class"][target_class] += 1

    # In báo cáo tổng kết
    print("\n" + "-" * 70)
    print("📊 BÁO CÁO KẾT QUẢ LÀM SẠCH:")
    print("-" * 70)
    print(f"  • Tổng số ảnh quét được     : {stats['total_scanned']:,}")
    print(f"  • Ảnh lỗi / hỏng / quá nhỏ  : {stats['corrupted_or_too_small']:,}")
    print(f"  • Ảnh trùng lặp bị loại bỏ  : {stats['duplicates']:,}")
    print(f"  • Số ảnh SẠCH giữ lại       : {stats['valid_saved']:,}")
    print("-" * 70)
    print("  Phân bố theo từng lớp sau khi làm sạch:")
    for cls, count in stats["by_class"].items():
        pct = (count / stats["valid_saved"] * 100) if stats["valid_saved"] > 0 else 0
        print(f"    - {cls:<12}: {count:>6,} ảnh ({pct:>5.1f}%)")
    print("=" * 70)
    print(f"✅ Hoàn tất! Dữ liệu sạch đã sẵn sàng tại: {output_dir}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Bước 1: Làm sạch dữ liệu giao thông")
    parser.add_argument(
        "--raw-dirs",
        nargs="+",
        default=[
            str(DATASETS_DIR / "vehicles-classification-1"),
            str(DATASETS_DIR / "Vehicle-Models-1"),
        ],
        help="Danh sách thư mục dữ liệu thô cần quét"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATASETS_DIR / "cleaned_traffic"),
        help="Thư mục lưu trữ dữ liệu sau khi làm sạch"
    )
    parser.add_argument("--min-size", type=int, default=24, help="Kích thước tối thiểu (px)")
    parser.add_argument("--keep-duplicates", action="store_true", help="Không lọc trùng lặp")

    args = parser.parse_args()

    clean_dataset(
        raw_dirs=args.raw_dirs,
        output_dir=Path(args.output_dir),
        min_size=args.min_size,
        remove_duplicates=not args.keep_duplicates,
        copy_files=True
    )


if __name__ == "__main__":
    main()
