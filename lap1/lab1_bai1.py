# ============================================
# LAB 1 - BÀI 1: KHỞI TẠO VÀ CHUYỂN VỊ MA TRẬN
# ============================================

def transpose_matrix(A):
    """Nhận ma trận A (m x n), trả về ma trận chuyển vị A^T (n x m)."""
    # Bước 1: Xác định số hàng và số cột của ma trận gốc
    rows = len(A)
    cols = len(A[0])

    # Bước 2: Khởi tạo A^T kích thước (cols x rows), gán toàn bộ = 0
    A_T = [[0 for _ in range(rows)] for _ in range(cols)]

    # Bước 3 + 4: Duyệt 2 vòng lặp lồng nhau, hoán đổi chỉ số hàng/cột
    for i in range(rows):          # vòng ngoài: chỉ số hàng i
        for j in range(cols):      # vòng trong: chỉ số cột j
            A_T[j][i] = A[i][j]    # gán phần tử tương ứng

    # Bước 5: Trả về ma trận kết quả
    return A_T


# ---------- Chạy thử nghiệm ----------
A = [
    [1, 2, 3],
    [4, 5, 6]
]

print("Ma trận gốc A:")
for row in A:
    print(row)

print("\nMa trận chuyển vị A_T:")
for row in transpose_matrix(A):
    print(row)

# Kết quả kỳ vọng:
# [1, 4]
# [2, 5]
# [3, 6]