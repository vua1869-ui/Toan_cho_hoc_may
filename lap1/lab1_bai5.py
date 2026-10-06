"""Bài 5: Khử Gauss với chọn phần tử trụ từng phần (partial pivoting)."""


def gaussian_elimination(aug_matrix):
    """Trả về bản sao dạng bậc thang, làm tròn 2 chữ số thập phân."""
    if not aug_matrix:
        return []
    rows = len(aug_matrix)
    cols = len(aug_matrix[0])
    if cols < 2:
        raise ValueError("Ma trận bổ sung cần ít nhất một cột hệ số và cột b.")
    for row in aug_matrix:
        if len(row) != cols:
            raise ValueError("Các hàng phải có cùng số cột.")

    # Sao chép để không thay đổi ma trận đầu vào.
    A = [[float(value) for value in row] for row in aug_matrix]
    pivot_row = 0
    eps = 1e-12  # Ngưỡng coi số rất nhỏ là 0 khi tính bằng số thực.

    # Tách chỉ số hàng trụ và cột để xử lý cả ma trận thiếu hạng/chữ nhật.
    for col in range(cols):
        if pivot_row == rows:
            break
        best_row = pivot_row
        for i in range(pivot_row + 1, rows):
            if abs(A[i][col]) > abs(A[best_row][col]):
                best_row = i

        # Cột không có trụ: bỏ qua cột, giữ nguyên hàng trụ.
        if abs(A[best_row][col]) <= eps:
            for i in range(pivot_row, rows):
                A[i][col] = 0.0
            continue

        A[pivot_row], A[best_row] = A[best_row], A[pivot_row]
        for i in range(pivot_row + 1, rows):
            factor = A[i][col] / A[pivot_row][col]
            A[i][col] = 0.0
            for j in range(col + 1, cols):
                A[i][j] -= factor * A[pivot_row][j]
        pivot_row += 1

    # Chỉ làm tròn sau khi hoàn thành khử để tránh tích lũy sai số.
    for i in range(rows):
        for j in range(cols):
            A[i][j] = round(A[i][j], 2)
            if A[i][j] == 0:
                A[i][j] = 0.0  # Loại bỏ cách hiển thị -0.0.
    return A


# Với ma trận hệ số vuông n x n (ma trận bổ sung n x (n+1)):
# - Tìm trụ trên tất cả các cột: O(n^2).
# - Khử xuôi: n bước, mỗi bước cập nhật tối đa n hàng * (n+1) cột.
#   Tổng chi phí là O(n^3), chi phối độ phức tạp thời gian.
# - Bản sao ma trận dùng O(n^2) bộ nhớ bổ sung.


if __name__ == "__main__":
    augmented_matrix = [
        [2.0, 1.0, -1.0, 8.0],
        [-3.0, -1.0, 2.0, -11.0],
        [-2.0, 1.0, 2.0, -3.0],
    ]
    print("Ma trận bậc thang (làm tròn 2 chữ số thập phân):")
    for row in gaussian_elimination(augmented_matrix):
        print("[" + ", ".join(f"{value:.2f}" for value in row) + "]")
    # [-3.00, -1.00, 2.00, -11.00]
    # [ 0.00,  1.67, 0.67,   4.33]
    # [ 0.00,  0.00, 0.20,  -0.20]
    # Thế ngược trên kết quả chưa làm tròn cho nghiệm x=2, y=3, z=-1.
