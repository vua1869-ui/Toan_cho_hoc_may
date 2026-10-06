def project_data_1d(X_centered, pc_vector):
    """
    Chiếu ma trận dữ liệu đã chuẩn hóa tâm X_centered (M x N) lên vector riêng pc_vector (độ dài N).
    Trả về danh sách 1D gồm M tọa độ mới tương ứng với tích vô hướng.[cite: 2]
    """
    M = len(X_centered)
    N = len(X_centered[0])
    
    projected = []
    for i in range(M):
        dot_product = sum(X_centered[i][j] * pc_vector[j] for j in range(N))
        projected.append(dot_product)
        
    return projected


if __name__ == "__main__":
    from lab3_bai3 import mean_centering
    
    medical_data = [
        [120, 95, 210, 24.5],
        [140, 130, 250, 29.0],
        [110, 85, 180, 21.5],
        [155, 160, 280, 32.0],
        [130, 105, 220, 26.0]
    ]
    
    X_centered = mean_centering(medical_data)
    
    # Giả sử vector thành phần chính PC1 đã chuẩn hóa
    pc1_vector = [0.33, 0.58, 0.74, 0.08]
    
    projected_1d = project_data_1d(X_centered, pc1_vector)
    print("Tọa độ dữ liệu sau khi chiếu lên PC1:")
    for i, val in enumerate(projected_1d):
        print(f"Bệnh nhân {i+1}: {val:.4f}")