def verify_eigen(A, x, lambda_val, eps=1e-6):
    """
    Kiểm tra x có phải là vector riêng của ma trận A tương ứng với trị riêng lambda_val hay không.
    Điều kiện: A * x == lambda_val * x (xử lý sai số số thực với eps)[cite: 1]
    """
    n = len(A)
    
    # Bước 1: Tính vế trái LHS = A * x[cite: 1]
    Ax = []
    for i in range(n):
        Ax.append(sum(A[i][j] * x[j] for j in range(len(x))))
        
    # Bước 2: Tính vế phải RHS = lambda * x[cite: 1]
    lambda_x = [lambda_val * val for val in x]
    
    # Bước 3 & 4: So sánh sai số abs(LHS[i] - RHS[i]) < eps[cite: 1]
    is_valid = all(abs(Ax[i] - lambda_x[i]) < eps for i in range(n))
    
    return is_valid, Ax, lambda_x


if __name__ == "__main__":
    A = [[4, 2], [1, 3]]
    x = [2, 1]
    lambda_val = 5
    
    is_valid, Ax, lambda_x = verify_eigen(A, x, lambda_val)
    
    print(f"A * x      : {Ax}")
    print(f"Lambda * x : {lambda_x}")
    print(f"x là vector riêng của A: {is_valid}")