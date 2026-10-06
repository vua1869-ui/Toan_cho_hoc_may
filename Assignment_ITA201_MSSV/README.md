# POLY-MART / AURA Studio: Toán cho học máy (ITA201)

Hệ thống phân tích dữ liệu cho cửa hàng thời trang trực tuyến, gồm 4 module tự cài đặt bằng **Python 3 thuần**
(không dùng numpy, networkx hay thư viện giải thuật cho phần lõi).

## Yêu cầu
- Python 3.8 trở lên. **Không cần cài thêm thư viện nào** để chạy chương trình.
- (Tùy chọn) `numpy`, `scipy`: chỉ dùng trong `test_modules.py` để đối chiếu kết quả. Không có thì các test đó tự bỏ qua.

## Cách chạy
```bash
python main_ecommerce.py        # menu console (main_dashboard.py là bí danh, chạy giống hệt)
python test_modules.py          # chạy 17 bài kiểm thử tự động
python customer_analytics.py    # hoặc chạy riêng demo từng module
```

Menu:
```
1. Demo Module 1: Chuẩn hóa khoảng cách & Nén chiều PCA
2. Demo Module 2: Sinh tập con, Naive Bayes & Tính Entropy/IG
3. Demo Module 3: Duyệt đồ thị BFS/DFS & Dự báo Markov
4. Demo Module 4: Phân bổ ngân sách LP & Gradient Descent
5. Chạy toàn bộ Pipeline tổng hợp
0. Thoát chương trình
```

## Cấu trúc
| File | Nội dung | Bài |
|---|---|---|
| `customer_analytics.py` | Khoảng cách L1/L2, nhân ma trận A·x, PCA thuần (power iteration) | 1, 2, 3 |
| `review_classifier.py` | Quay lui sinh tập con, Naive Bayes + Laplace, Entropy và Information Gain | 4, 5 |
| `behavior_network.py` | Danh sách/ma trận kề, BFS, DFS, chuỗi Markov, đồ thị hai phía (tô màu BFS) | 6, 7 |
| `revenue_optimizer.py` | Quy hoạch tuyến tính duyệt đỉnh, dạng chính tắc, đối ngẫu, Gradient Descent | 8 |
| `main_ecommerce.py` | Menu console điều khiển | |
| `linalg_utils.py` | Hàm ma trận dùng chung (tích vô hướng, nhân ma trận, khử Gauss...) | |
| `sample_data.py` | Dữ liệu mẫu theo bối cảnh cửa hàng quần áo | |
| `test_modules.py` | 17 test tự động | |
| `BaoCao_Assignment_MSSV.docx` | Báo cáo phân tích toán học và kết quả | |

## Ghi chú
- **Ràng buộc quy hoạch tuyến tính** (Module 4.1) là số giả định vì đề không nêu giá trị. Để đổi sang số của đề, sửa
  `PRIMAL_A`, `PRIMAL_B`, `PRIMAL_C` ở đầu file `revenue_optimizer.py`.
- **PC1, PC2** được chương trình tự tính. Nếu có sẵn hai trục, truyền vào `pca(X, 2, components=[PC1, PC2])`.
- Dữ liệu mẫu nằm trong `sample_data.py`, có thể thay bằng dữ liệu của website (đơn hàng, đánh giá, lượt xem).
- Kết quả demo (PCA, Naive Bayes, Markov, LP) khớp với numpy/scipy khi đối chiếu trong test.
