def mean_centering(X):
    """
    Tính trung bình từng cột và trả về ma trận đã chuẩn hóa tâm X_centered.[cite: 2]
    Không sử dụng thư viện numpy.[cite: 2]
    """
    M = len(X)
    N = len(X[0])
    
    # Tính trung bình từng cột
    means = [0.0] * N
    for j in range(N):
        column_sum = sum(X[i][j] for i in range(M))
        means[j] = column_sum / M
        
    # Trừ trung bình
    X_centered = []
    for i in range(M):
        row = [X[i][j] - means[j] for j in range(N)]
        X_centered.append(row)
        
    return X_centered


def compute_covariance_matrix(X_centered):
    """
    Tính ma trận hiệp phương sai Cov (N x N) theo công thức:
    Cov[i][j] = sum(X_centered[k][i] * X_centered[k][j] for k in range(M)) / (M - 1)[cite: 2]
    """
    M = len(X_centered)
    N = len(X_centered[0])
    
    cov_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            cov_sum = sum(X_centered[k][i] * X_centered[k][j] for k in range(M))
            cov_matrix[i][j] = cov_sum / (M - 1)
            
    return cov_matrix


if __name__ == "__main__":
    medical_data = [
        [120, 95, 210, 24.5],
        [140, 130, 250, 29.0],
        [110, 85, 180, 21.5],
        [155, 160, 280, 32.0],
        [130, 105, 220, 26.0]
    ]
    
    X_centered = mean_centering(medical_data)
    cov = compute_covariance_matrix(X_centered)
    
    print("Ma trận sau khi trừ trung bình (X_centered):")
    for row in X_centered:
        print([round(val, 2) for val in row])
        
    print("\nMa trận hiệp phương sai Cov (4x4):")
    for row in cov:
        print([round(val, 2) for val in row])