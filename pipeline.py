"""
pipeline.py - Trình Điều Phối Toàn Diện Quy Trình Xử Lý & Huấn Luyện (Master Pipeline)
=============================================================================
Cung cấp giao diện dòng lệnh tập trung để thực thi độc lập hoặc liên hoàn 4 bước:
  1. python pipeline.py --step 1   (Làm sạch dữ liệu)
  2. python pipeline.py --step 2   (Phân chia & Cân bằng dữ liệu)
  3. python pipeline.py --step 3   (Khảo sát & Trực quan hóa dữ liệu EDA)
  4. python pipeline.py --step 4   (Huấn luyện chuyên sâu mô hình YOLOv8-cls)
  Hoặc:
  python pipeline.py --all         (Chạy liên hoàn cả 4 bước tự động)
=============================================================================
"""

import sys
import io
import subprocess
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt và emoji mượt mà trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PYTHON_EXE = sys.executable

STEPS = {
    1: ("1_clean_data.py", "LÀM SẠCH DỮ LIỆU (DATA CLEANING)"),
    2: ("2_split_data.py", "PHÂN CHIA & CÂN BẰNG DỮ LIỆU (DATA SPLIT & BALANCE)"),
    3: ("3_survey_data.py", "KHẢO SÁT & TRỰC QUAN HÓA DỮ LIỆU (DATA SURVEY & EDA)"),
    4: ("4_train_model.py", "HUẤN LUYỆN CHUYÊN SÂU MÔ HÌNH (EXPERT MODEL TRAINING)")
}


def run_step(step_num: int, extra_args=None):
    """Chạy một bước cụ thể với logging rõ ràng."""
    if step_num not in STEPS:
        print(f"❌ Bước không hợp lệ: {step_num}. Vui lòng chọn từ 1 đến 4.")
        return False

    script_name, description = STEPS[step_num]
    script_path = Path(__file__).resolve().parent / script_name

    if not script_path.exists():
        print(f"❌ Không tìm thấy tệp mã nguồn: {script_path}")
        return False

    print("\n" + "━" * 80)
    print(f"▶️ ĐANG THỰC HIỆN BƯỚC {step_num}: {description}")
    print(f"📄 Script: {script_name}")
    print("━" * 80 + "\n")

    cmd = [PYTHON_EXE, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"\n❌ Bước {step_num} kết thúc với mã lỗi {ret}!")
        return False

    print(f"\n✅ BƯỚC {step_num} ĐÃ HOÀN THÀNH XUẤT SẮC!")
    return True


def interactive_menu():
    """Hiển thị menu tương tác thân thiện khi người dùng chạy python pipeline.py."""
    while True:
        print("\n" + "╔" + "═" * 70 + "╗")
        print("║      HỆ THỐNG XỬ LÝ DỮ LIỆU & HUẤN LUYỆN TRAFFICVISION AI           ║")
        print("╠" + "═" * 70 + "╣")
        print("║  1. [Bước 1] Làm sạch dữ liệu (Data Cleaning & Filter)               ║")
        print("║  2. [Bước 2] Phân chia & Cân bằng dữ liệu (Train/Val/Test Split)     ║")
        print("║  3. [Bước 3] Khảo sát & Trực quan hóa dữ liệu (EDA & Charts)         ║")
        print("║  4. [Bước 4] Huấn luyện chuyên sâu mô hình YOLOv8-cls (Training)     ║")
        print("║  5. [Chạy tất cả] Thực hiện tự động cả 4 bước liên hoàn (Full Flow)  ║")
        print("║  0. Thoát chương trình                                               ║")
        print("╚" + "═" * 70 + "╝")

        choice = input("\n👉 Nhập lựa chọn của bạn (0 - 5): ").strip()
        if choice == "0":
            print("Tạm biệt!")
            break
        elif choice in ("1", "2", "3", "4"):
            run_step(int(choice))
        elif choice == "5":
            for s in (1, 2, 3, 4):
                ok = run_step(s)
                if not ok:
                    print(f"⚠️ Dừng quy trình tại bước {s}.")
                    break
        else:
            print("⚠️ Lựa chọn không hợp lệ. Vui lòng nhập số từ 0 đến 5.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Điều Phối Quy Trình AI Giao Thông")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Chọn chạy riêng lẻ bước 1, 2, 3 hoặc 4")
    parser.add_argument("--all", action="store_true", help="Chạy tự động liên hoàn cả 4 bước")

    args, extra_args = parser.parse_known_args()

    if args.all:
        for s in (1, 2, 3, 4):
            ok = run_step(s, extra_args)
            if not ok:
                sys.exit(1)
        print("\n🎉 TOÀN BỘ QUY TRÌNH 4 BƯỚC ĐÃ HOÀN TẤT THÀNH CÔNG!")
    elif args.step:
        ok = run_step(args.step, extra_args)
        if not ok:
            sys.exit(1)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
