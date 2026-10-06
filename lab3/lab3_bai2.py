def mat_mul_2x2(A, B):
    """Hàm bổ trợ nhân hai ma trận kích thước 2x2."""
    res = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(2):
        for j in range(2):
            res[i][j] = sum(A[i][k] * B[k][j] for k in range(2))
    return res


def matrix_power_fast(P, D_diag, P_inv, k):
    """
    Tính nhanh A^k qua công thức chéo hóa: A^k = P * (D^k) * P^(-1)[cite: 2]
    """
    # Bước 1 & 2: Tính D^k với ma trận đường chéo[cite: 2]
    D_k = [
        [D_diag[0] ** k, 0.0],
        [0.0, D_diag[1] ** k]
    ]
    
    # Bước 3: Nhân 3 ma trận P * D^k * P_inv[cite: 2]
    return mat_mul_2x2(mat_mul_2x2(P, D_k), P_inv)


if __name__ == "__main__":
    # Ma trận A = [[4, 2], [1, 3]] có P, D_diag, P_inv tương ứng:
    P = [[2.0, 1.0], [1.0, -1.0]]
    D_diag = [5.0, 2.0]
    P_inv = [[1/3, 1/3], [1/3, -2/3]]
    k = 3
    
    A_k = matrix_power_fast(P, D_diag, P_inv, k)
    
    print(f"Ma trận A^{k} (tính nhanh qua chéo hóa):")
    for row in A_k:
        print([round(val, 4) for val in row])