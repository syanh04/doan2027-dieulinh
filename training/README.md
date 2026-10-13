# 🧠 Pipeline Huấn luyện Mô hình AI (Training Pipeline)

Thư mục này gom toàn bộ các tập lệnh xử lý dữ liệu và huấn luyện mô hình YOLOv8:

- `1_clean_data.py`: Làm sạch và chuẩn hóa dữ liệu ảnh/nhãn
- `2_split_data.py`: Phân chia tập dữ liệu thành Train / Val / Test
- `3_survey_data.py`: Khảo sát phân bố kích thước bbox và số lượng lớp
- `4_train_model.py`: Chạy huấn luyện mô hình YOLOv8 trên GPU/CPU
- `pipeline.py`: Kịch bản thực thi tự động toàn bộ quy trình từ 1 đến 4
