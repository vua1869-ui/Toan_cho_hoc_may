# ============================================
# LAB 2 - BÀI 1: Tính Tổ hợp Tuyến tính
# và Tọa độ theo Cơ sở mới
# ============================================

def compute_linear_combination(B, c):
    """
    Tính tổ hợp tuyến tính: v = c1*b1 + c2*b2 + ... + cn*bn

    Tham số:
        B : danh sách các vector cơ sở (mỗi vector là một dòng của B)
        c : danh sách hệ số tọa độ tương ứng
    Trả về:
        v : vector kết quả của tổ hợp tuyến tính
    """
    # Bước 1: Xác định số chiều của không gian
    dim = len(B[0])

    # Bước 2: Khởi tạo vector kết quả v gồm dim phần tử bằng 0
    v = [0.0] * dim

    # Bước 3: Duyệt qua từng vector cơ sở B[i] và hệ số c[i]
    for i in range(len(B)):
        # Bước 4: Cộng dồn c[i] * B[i][j] vào vị trí v[j]
        for j in range(dim):
            v[j] += c[i] * B[i][j]

    # Bước 5: Trả về vector kết quả
    return v


# ---------- Dữ liệu đầu vào mẫu ----------
B = [
    [1, 0],   # b1
    [1, 1]    # b2
]
c = [-2, 7]

# ---------- Chạy và in kết quả ----------
v = compute_linear_combination(B, c)
print(f"Vector kết quả v: {v}")