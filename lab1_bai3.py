"""Bài 3: Nhân ma trận trọng số W với vector đầu vào x."""


def matrix_vector_multiply(W, x):
    if not W or not W[0]:
        print("Lỗi: Ma trận W phải có ít nhất một hàng và một cột.")
        return None
    m = len(W)
    n = len(W[0])
    for row in W:
        if len(row) != n:
            print("Lỗi: Các hàng của W phải có cùng số cột.")
            return None
    if n != len(x):
        print("Lỗi: Số cột của W phải bằng số phần tử của x.")
        return None

    y = [0 for _ in range(m)]
    for i in range(m):
        for j in range(n):
            y[i] += W[i][j] * x[j]
    return y


if __name__ == "__main__":
    W = [[0.2, 0.5, -0.1], [0.8, -0.3, 0.4]]
    x = [10, 2, 5]
    print("y =", matrix_vector_multiply(W, x))  # [2.5, 9.4]
