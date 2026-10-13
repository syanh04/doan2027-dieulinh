"""
3_survey_data.py - Module Khảo sát & Phân tích Dữ liệu Khám phá (Data Survey & EDA)
=============================================================================
Nhiệm vụ:
  1. Khảo sát thống kê chi tiết bộ dữ liệu:
     - Số lượng mẫu theo từng lớp (Class) và từng tập (Train/Val/Test).
     - Đo lường chỉ số mất cân bằng dữ liệu (Imbalance Ratio).
  2. Phân tích đặc trưng kỹ thuật của hình ảnh:
     - Thống kê độ phân giải (Width, Height, Mean, Median, Min, Max).
     - Phân tích tỉ lệ khung hình (Aspect Ratio = W/H).
  3. Tự động vẽ và xuất các biểu đồ trực quan hóa chuyên nghiệp (lưu vào reports/):
     - Biểu đồ 1: Cột nhóm so sánh phân bố giữa các tập dữ liệu.
     - Biểu đồ 2: Biểu đồ tròn (Pie chart) tỉ lệ phần trăm các lớp phương tiện.
     - Biểu đồ 3: Biểu đồ phân bố kích thước ảnh (Resolution & Aspect ratio).
     - Biểu đồ 4: Thư viện ảnh mẫu thực tế đại diện cho 4 lớp phương tiện.
  4. Xuất file báo cáo khoa học dạng Markdown và JSON phục vụ làm đồ án/báo cáo.
=============================================================================
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt và emoji mượt mà trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from config import BALANCED_DATASET_DIR, WORKSPACE_DIR

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES_VN = {
    "0_Xe_may": "Xe máy",
    "1_O_to_con": "Ô tô con",
    "2_Xe_khach": "Xe khách",
    "3_Xe_tai": "Xe tải"
}
PALETTE = ["#2b6cb0", "#319795", "#d69e2e", "#e53e3e"]


def scan_dataset(dataset_dir: Path):
    """Quét và thu thập metadata của toàn bộ ảnh trong bộ dữ liệu."""
    records = []
    splits = [d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "valid", "test")]

    if not splits:
        # Nếu thư mục chứa trực tiếp các class
        splits = ["all"]

    print(f"🔍 Đang quét và phân tích thuộc tính hình ảnh từ: {dataset_dir} ...")

    for split in splits:
        split_path = dataset_dir / split if split != "all" else dataset_dir
        for cls_dir in sorted(split_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]

            for f in files:
                try:
                    with Image.open(f) as img:
                        w, h = img.size
                        mode = img.mode
                    records.append({
                        "filepath": str(f),
                        "filename": f.name,
                        "split": "val" if split == "valid" else split,
                        "class": cls_name,
                        "class_vn": CLASS_NAMES_VN.get(cls_name, cls_name),
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / h, 2) if h > 0 else 1.0,
                        "mode": mode,
                        "size_kb": round(f.stat().st_size / 1024, 1)
                    })
                except Exception:
                    continue

    return pd.DataFrame(records)


def generate_survey_report(df: pd.DataFrame, output_dir: Path, sample_count: int = 4):
    """Tạo biểu đồ và xuất báo cáo khảo sát chi tiết."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("📈 BƯỚC 3: KHẢO SÁT & TRỰC QUAN HÓA DỮ LIỆU (DATA SURVEY & EDA)")
    print("=" * 70)

    if df.empty:
        print("❌ Không có dữ liệu để khảo sát!")
        return

    total_images = len(df)
    unique_classes = sorted(df["class"].unique())
    splits = sorted(df["split"].unique())

    # 1. Bảng ma trận phân bố
    pivot_df = df.pivot_table(index="class", columns="split", values="filepath", aggfunc="count", fill_value=0)
    pivot_df["Tổng cộng"] = pivot_df.sum(axis=1)
    pivot_df["Tỉ lệ (%)"] = (pivot_df["Tổng cộng"] / total_images * 100).round(1)

    print(f"📊 TỔNG SỐ ẢNH TOÀN BỘ DATASET: {total_images:,} ảnh")
    print(f"📊 CÁC TẬP DỮ LIỆU HIỆN CÓ     : {', '.join(splits)}")
    print("-" * 70)
    print(pivot_df.to_string())
    print("-" * 70)

    # Đo lường Imbalance Ratio
    counts = pivot_df["Tổng cộng"].values
    imbalance_ratio = max(counts) / min(counts) if len(counts) > 0 and min(counts) > 0 else 1.0
    print(f"⚖️ Tỉ lệ mất cân bằng mẫu (Max/Min): {imbalance_ratio:.2f}")
    if imbalance_ratio <= 1.5:
        print("  -> Đánh giá: Dataset rất cân bằng (Lý tưởng cho huấn luyện).")
    elif imbalance_ratio <= 3.0:
        print("  -> Đánh giá: Độ lệch ở mức chấp nhận được.")
    else:
        print("  -> Cảnh báo: Dataset lệch nhiều, nên áp dụng Class Weights hoặc Oversampling.")

    # 2. Thống kê kích thước ảnh
    print("\n📐 THỐNG KÊ KÍCH THƯỚC HÌNH ẢNH (RESOLUTION STATS):")
    res_stats = {
        "Width (Min - Max - Mean)": f"{df['width'].min()} - {df['width'].max()} - {df['width'].mean():.1f} px",
        "Height (Min - Max - Mean)": f"{df['height'].min()} - {df['height'].max()} - {df['height'].mean():.1f} px",
        "Aspect Ratio (Mean)": f"{df['aspect_ratio'].mean():.2f}",
        "File Size Trung bình": f"{df['size_kb'].mean():.1f} KB"
    }
    for k, v in res_stats.items():
        print(f"  • {k:<28}: {v}")

    # =========================================================================
    # VẼ BIỂU ĐỒ 1: PHÂN BỐ LỚP THEO TẬP (SPLIT COMPARISON BAR CHART)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = pivot_df.drop(columns=["Tổng cộng", "Tỉ lệ (%)"], errors="ignore")
    plot_data.plot(kind="bar", ax=ax, colormap="viridis", width=0.7, edgecolor="black", alpha=0.85)
    plt.title("Phân bố Số lượng Ảnh từng Lớp theo Tập Dữ liệu", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Lớp phương tiện (Vehicle Class)", fontsize=11, labelpad=8)
    plt.ylabel("Số lượng ảnh", fontsize=11, labelpad=8)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Tập (Split)")
    plt.tight_layout()
    chart1_path = plots_dir / "1_class_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # =========================================================================
    # VẼ BIỂU ĐỒ 2: BIỂU ĐỒ TRÒN TỔNG THỂ (OVERALL PIE CHART)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 7))
    class_totals = df["class_vn"].value_counts()
    ax.pie(
        class_totals.values,
        labels=[f"{k}\n({v:,} ảnh)" for k, v in class_totals.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=PALETTE,
        wedgeprops=dict(edgecolor="black", width=0.6)  # Donut style
    )
    plt.title("Tỉ lệ Phần trăm Phương tiện trong Toàn bộ Dataset", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    chart2_path = plots_dir / "2_overall_pie_chart.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # =========================================================================
    # VẼ BIỂU ĐỒ 3: PHÂN BỐ KÍCH THƯỚC & TỈ LỆ KHUNG HÌNH (SCATTER & HISTOGRAM)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter Width vs Height
    sample_df = df.sample(min(len(df), 2000), random_state=42)
    ax1.scatter(sample_df["width"], sample_df["height"], alpha=0.35, color="#2b6cb0", edgecolors="none", s=25)
    ax1.set_title("Phân bố Kích thước Ảnh (Width vs Height)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Chiều rộng (Width px)", fontsize=10)
    ax1.set_ylabel("Chiều cao (Height px)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Histogram Aspect Ratio
    ax2.hist(df["aspect_ratio"], bins=30, color="#319795", edgecolor="black", alpha=0.75)
    ax2.set_title("Phân bố Tỉ lệ Khung hình (Aspect Ratio = W/H)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tỉ lệ (W/H)", fontsize=10)
    ax2.set_ylabel("Tần số (Số lượng ảnh)", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart3_path = plots_dir / "3_image_resolutions.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()

    # =========================================================================
    # VẼ BIỂU ĐỒ 4: BỘ MẪU ẢNH THỰC TẾ (SAMPLE GALLERY GRID)
    # =========================================================================
    n_classes = len(unique_classes)
    fig, axes = plt.subplots(n_classes, sample_count, figsize=(sample_count * 3, n_classes * 2.8))

    for row_idx, cls_name in enumerate(unique_classes):
        cls_df = df[df["class"] == cls_name]
        sample_imgs = cls_df.sample(min(len(cls_df), sample_count), random_state=42)

        for col_idx in range(sample_count):
            ax = axes[row_idx, col_idx] if n_classes > 1 else axes[col_idx]
            if col_idx < len(sample_imgs):
                img_row = sample_imgs.iloc[col_idx]
                try:
                    img = mpimg.imread(img_row["filepath"])
                    ax.imshow(img)
                    ax.set_title(f"{img_row['width']}x{img_row['height']}", fontsize=9)
                except Exception:
                    ax.text(0.5, 0.5, "Lỗi đọc", ha="center")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(CLASS_NAMES_VN.get(cls_name, cls_name), fontsize=11, fontweight="bold", rotation=0, labelpad=40, va="center")

    plt.suptitle("Bộ Mẫu Hình Ảnh Thực Tế Đại Diện Từng Lớp Giao Thông", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart4_path = plots_dir / "4_sample_gallery.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()

    # =========================================================================
    # XUẤT BÁO CÁO MARKDOWN
    # =========================================================================
    report_md_path = output_dir / "data_survey_report.md"
    report_content = f"""# Báo Cáo Khảo Sát & Phân Tích Dữ Liệu Giao Thông (EDA Report)

- **Thời gian lập báo cáo:** Tự động tạo bởi `3_survey_data.py`
- **Tổng số lượng ảnh phân tích:** {total_images:,} ảnh
- **Các lớp phương tiện:** 0_Xe_may, 1_O_to_con, 2_Xe_khach, 3_Xe_tai
- **Tỉ lệ mất cân bằng (Imbalance Ratio):** {imbalance_ratio:.2f}

## 1. Bảng Thống Kê Chi Tiết Số Lượng Ảnh

| Lớp Phương Tiện | {" | ".join(splits)} | Tổng cộng | Tỉ lệ % |
| :--- | {" | ".join([":---:" for _ in splits])} | :---: | :---: |
"""
    for cls in unique_classes:
        row_vals = [str(pivot_df.loc[cls, s]) for s in splits]
        tot = pivot_df.loc[cls, "Tổng cộng"]
        pct = pivot_df.loc[cls, "Tỉ lệ (%)"]
        report_content += f"| **{cls}** ({CLASS_NAMES_VN.get(cls, '')}) | {' | '.join(row_vals)} | **{tot:,}** | {pct}% |\n"

    report_content += f"""
## 2. Đặc Trưng Kỹ Thuật Của Hình Ảnh
- **Độ phân giải trung bình:** {df['width'].mean():.1f} x {df['height'].mean():.1f} px
- **Độ phân giải nhỏ nhất:** {df['width'].min()} x {df['height'].min()} px
- **Độ phân giải lớn nhất:** {df['width'].max()} x {df['height'].max()} px
- **Tỉ lệ khung hình trung bình (W/H):** {df['aspect_ratio'].mean():.2f}
- **Dung lượng file trung bình:** {df['size_kb'].mean():.1f} KB

## 3. Các Biểu Đồ Trực Quan Hóa Đã Xuất
1. `figures/1_class_distribution.png`: Biểu đồ cột phân bố lớp theo từng tập.
2. `figures/2_overall_pie_chart.png`: Biểu đồ tròn tỉ lệ các phương tiện.
3. `figures/3_image_resolutions.png`: Phân bố kích thước ảnh và aspect ratio.
4. `figures/4_sample_gallery.png`: Lưới hình ảnh thực tế đại diện cho từng lớp.
"""
    report_md_path.write_text(report_content, encoding="utf-8")

    # Xuất JSON
    summary_json_path = output_dir / "survey_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": total_images,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "splits": splits,
            "classes": unique_classes,
            "resolution_stats": res_stats
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Hoàn tất khảo sát! Báo cáo & Biểu đồ đã lưu tại:")
    print(f"  • Thư mục biểu đồ: {plots_dir}")
    print(f"  • Báo cáo Markdown: {report_md_path}")
    print(f"  • Báo cáo JSON    : {summary_json_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Bước 3: Khảo sát & Phân tích Dữ liệu Giao thông")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(BALANCED_DATASET_DIR),
        help="Thư mục dataset cần khảo sát"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_DIR / "reports" / "eda"),
        help="Thư mục lưu biểu đồ và báo cáo khảo sát"
    )
    parser.add_argument("--samples", type=int, default=4, help="Số ảnh mẫu mỗi lớp hiển thị trong grid")

    args = parser.parse_args()

    df = scan_dataset(Path(args.dataset_dir))
    generate_survey_report(df, Path(args.output_dir), sample_count=args.samples)


if __name__ == "__main__":
    main()
