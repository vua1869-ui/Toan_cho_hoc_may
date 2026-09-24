"""Bài 4: Nhân hai ma trận và đếm số phép nhân số học."""


def matrix_multiply(A, B):
    if not A or not B or not A[0] or not B[0]:
        return None
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    for row in A:
        if len(row) != n:
            return None
    for row in B:
        if len(row) != p:
            return None
    if n != len(B):
        return None

    C = [[0 for _ in range(p)] for _ in range(m)]
    multiplication_count = 0
    # Mỗi phần tử C[i][j] là tích vô hướng hàng i của A và cột j của B.
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                multiplication_count += 1
    print("Tổng số phép nhân số học:", multiplication_count)
    return C


if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 1], [2, 3]]
    C = matrix_multiply(A, B)  # 2 * 2 * 3 = 12 phép nhân.
    print("Ma trận C:")
    if C is not None:
        for row in C:
            print(row)
    # Kết quả: [31, 19], [85, 55].
