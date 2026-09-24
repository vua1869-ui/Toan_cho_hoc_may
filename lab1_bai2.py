# ============================================
# LAB 1 - BÀI 2: TÍNH CHUẨN VECTOR L1 VÀ L2
# ============================================

def norm_l1(v):
    """Chuẩn Manhattan: ||v||_1 = sum(|v_i|) — đo tổng độ lớn sai số."""
    total = 0
    for x in v:
        total += abs(x)
    return total


def norm_l2(v):
    """Chuẩn Euclidean: ||v||_2 = sqrt(sum(v_i^2)) — dùng trong MSE, khoảng cách thực."""
    sum_sq = 0
    for x in v:
        sum_sq += x ** 2
    return sum_sq ** 0.5


# ---------- Chạy thử nghiệm ----------
error_vector = [3, -4]
print(f"L1 Norm: {norm_l1(error_vector)}")   # Kết quả: 7   (= |3| + |-4|)
print(f"L2 Norm: {norm_l2(error_vector)}")   # Kết quả: 5.0 (= sqrt(9 + 16))

# Ví dụ trong ngữ cảnh học máy: e = y - y_hat
y_hat  = [2.0, 3.0, 5.0]   # vector dự đoán
y      = [1.0, 7.0, 5.0]   # vector thực tế
e = [y[i] - y_hat[i] for i in range(len(y))]    # e = [-1, 4, 0]
print("Vector sai số e =", e)
print(f"L1 Norm của e: {norm_l1(e)}")   # 5
print(f"L2 Norm của e: {norm_l2(e)}")   # ≈ 4.123