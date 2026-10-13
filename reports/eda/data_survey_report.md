# Báo Cáo Khảo Sát & Phân Tích Dữ Liệu Giao Thông (EDA Report)

- **Thời gian lập báo cáo:** Tự động tạo bởi `3_survey_data.py`
- **Tổng số lượng ảnh phân tích:** 12,000 ảnh
- **Các lớp phương tiện:** 0_Xe_may, 1_O_to_con, 2_Xe_khach, 3_Xe_tai
- **Tỉ lệ mất cân bằng (Imbalance Ratio):** 1.00

## 1. Bảng Thống Kê Chi Tiết Số Lượng Ảnh

| Lớp Phương Tiện | test | train | val | Tổng cộng | Tỉ lệ % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0_Xe_may** (Xe máy) | 450 | 2100 | 450 | **3,000** | 25.0% |
| **1_O_to_con** (Ô tô con) | 450 | 2100 | 450 | **3,000** | 25.0% |
| **2_Xe_khach** (Xe khách) | 450 | 2100 | 450 | **3,000** | 25.0% |
| **3_Xe_tai** (Xe tải) | 450 | 2100 | 450 | **3,000** | 25.0% |

## 2. Đặc Trưng Kỹ Thuật Của Hình Ảnh
- **Độ phân giải trung bình:** 640.0 x 640.0 px
- **Độ phân giải nhỏ nhất:** 640 x 640 px
- **Độ phân giải lớn nhất:** 640 x 640 px
- **Tỉ lệ khung hình trung bình (W/H):** 1.00
- **Dung lượng file trung bình:** 78.9 KB

## 3. Các Biểu Đồ Trực Quan Hóa Đã Xuất
1. `figures/1_class_distribution.png`: Biểu đồ cột phân bố lớp theo từng tập.
2. `figures/2_overall_pie_chart.png`: Biểu đồ tròn tỉ lệ các phương tiện.
3. `figures/3_image_resolutions.png`: Phân bố kích thước ảnh và aspect ratio.
4. `figures/4_sample_gallery.png`: Lưới hình ảnh thực tế đại diện cho từng lớp.
