# 🎨 Khối Frontend (Giao diện Người dùng & Client)

Thư mục này quản lý toàn bộ các thành phần hiển thị, tương tác và phong cách giao diện của hệ thống TrafficVision Enterprise:

## 📁 Cấu trúc thư mục:

```text
frontend/
├── templates/                 # Các trang giao diện HTML Jinja2
│   ├── index.html             # Dashboard điều hành chính 7 Tabs (Tổng quan, Live, Phân tích, Dự báo...)
│   └── login.html             # Màn hình đăng nhập bảo mật (Dark SaaS theme)
│
└── static/                    # Tài nguyên tĩnh phục vụ trình duyệt
    ├── css/
    │   ├── style.css          # Bố cục giao diện, lưới camera 240px, thẻ KPI, bảng biểu
    │   └── tokens.css         # Hệ thống Design Tokens (Màu sắc Dark/Light, Spacing 8px)
    │
    ├── js/
    │   ├── main.js            # Controller frontend (chuyển tab, SSE stream, 4 biểu đồ động)
    │   ├── chart.min.js       # Thư viện Chart.js self-hosted
    │   └── lucide.min.js      # Bộ icon Lucide SVG self-hosted
    │
    └── samples/               # Video CCTV mẫu các nút giao Hà Nội & hình ảnh kiểm thử
```

---

## 🛠️ Hướng dẫn bảo trì & sửa giao diện:
- **Đổi màu sắc / Tùy chỉnh chế độ Tối - Sáng**: Mở `static/css/tokens.css`.
- **Chỉnh sửa kích thước khung video / Bố cục thẻ**: Mở `static/css/style.css`.
- **Thêm tính năng biểu đồ / Cập nhật tự động**: Mở `static/js/main.js`.
- **Chỉnh sửa giao diện nội dung các Tab**: Mở `templates/index.html`.
