"""Tiện ích đại số tuyến tính dùng chung cho các module (Python thuần, không numpy)."""
import math


def dot(u, v):
    """Tích vô hướng của hai vector."""
    return sum(a * b for a, b in zip(u, v))


def norm(v):
    """Độ dài (chuẩn L2) của vector."""
    return math.sqrt(dot(v, v))


def transpose(A):
    """Ma trận chuyển vị."""
    return [list(row) for row in zip(*A)]


def mat_vec(A, x):
    """Nhân ma trận A (m x n) với vector cột x (n) -> vector (m)."""
    return [dot(row, x) for row in A]


def mat_mul(A, B):
    """Nhân hai ma trận A (m x n) và B (n x p) -> ma trận (m x p)."""
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]


def vec_mat(v, A):
    """Nhân vector hàng v (m) với ma trận A (m x n) -> vector hàng (n)."""
    return mat_vec(transpose(A), v)


def identity(n):
    """Ma trận đơn vị n x n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def solve_linear_system(A, b):
    """Giải hệ A x = b bằng khử Gauss có chọn phần tử trụ (partial pivoting)."""
    n = len(A)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Hệ phương trình suy biến, không có nghiệm duy nhất")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def print_matrix(M, fmt="{:9.3f}", labels=None):
    """In ma trận gọn gàng ra console."""
    for i, row in enumerate(M):
        prefix = f"{labels[i]:>8} " if labels else "  "
        print(prefix + " ".join(fmt.format(v) for v in row))
