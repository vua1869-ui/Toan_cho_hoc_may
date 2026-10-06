from lab3_bai3 import mean_centering, compute_covariance_matrix
from lab3_bai4 import project_data_1d


def pca_reduce_2d(data, pc1=None, pc2=None):
    """
    Quy trình PCA thu gọn từ 4D về 2D:[cite: 3]
    1. Trừ trung bình dữ liệu[cite: 1]
    2. Tính ma trận hiệp phương sai[cite: 1]
    3. Chiếu dữ liệu lên 2 trục vector riêng PC1, PC2[cite: 3]
    """
    # 1. Trừ trung bình[cite: 1]
    X_centered = mean_centering(data)
    
    # 2. Ma trận hiệp phương sai 4x4[cite: 3]
    cov_matrix = compute_covariance_matrix(X_centered)
    
    # Vector riêng mặc định ứng với 2 trị riêng lớn nhất của bộ dữ liệu[cite: 3]
    if pc1 is None or pc2 is None:
        pc1 = [0.3323, 0.5794, 0.7432, 0.0815]
        pc2 = [0.8175, -0.5724, 0.0805, -0.0101]
        
    # 3. Chiếu lên PC1 và PC2[cite: 3]
    proj_pc1 = project_data_1d(X_centered, pc1)
    proj_pc2 = project_data_1d(X_centered, pc2)
    
    data_2d = [[proj_pc1[i], proj_pc2[i]] for i in range(len(data))]
    return data_2d, cov_matrix


def calculate_explained_variance_ratio(lambdas):
    """Tính tỷ lệ phương sai giải thích được: Ratio = (lambda_1 + lambda_2) / sum(lambda)[cite: 3]"""
    return (lambdas[0] + lambdas[1]) / sum(lambdas)


if __name__ == "__main__":
    medical_data = [
        [120, 95, 210, 24.5],
        [140, 130, 250, 29.0],
        [110, 85, 180, 21.5],
        [155, 160, 280, 32.0],
        [130, 105, 220, 26.0]
    ]
    
    # 1. Chạy quy trình PCA giảm về 2D[cite: 3]
    data_2d, cov_matrix = pca_reduce_2d(medical_data)
    
    print("--- 1. MA TRẬN HIỆP PHƯƠNG SAI 4x4 ---")
    for row in cov_matrix:
        print([round(val, 2) for val in row])
        
    print("\n--- 2. DỮ LIỆU ĐÃ GIẢM CHIỀU (2D) ---")
    for i, point in enumerate(data_2d):
        print(f"Bệnh nhân {i+1}: PC1 = {point[0]:8.4f} | PC2 = {point[1]:8.4f}")
        
    # 2. Tính Tỷ lệ phương sai giải thích được[cite: 3]
    lambdas = [145.2, 32.8, 4.5, 1.2]
    ratio = calculate_explained_variance_ratio(lambdas)
    
    print("\n--- 3. TỶ LỆ PHƯƠNG SAI GIẢI THÍCH (EXPLAINED VARIANCE RATIO) ---")
    print(f"Lambdas: {lambdas}")
    print(f"Ratio   : ({lambdas[0]} + {lambdas[1]}) / {sum(lambdas):.1f} = {ratio:.4f} ({ratio * 100:.2f}%)")


# ==============================================================================
# 3. PHÂN TÍCH Ý NGHĨA HỌC MÁY (MACHINE LEARNING ANALYSIS)[cite: 3]
# ==============================================================================
# Câu 1: Tỷ lệ thông tin đạt ~96.90% (> 90%), việc loại bỏ 2 chiều cuối KHÔNG
# làm mất bản chất dữ liệu.
# Giải thích: Hai thành phần chính PC1 và PC2 đã đại diện cho 96.90% tổng độ
# biến thiên (phương sai) của bộ dữ liệu gốc. 2 chiều còn lại chỉ chiếm vỏn vẹn
# ~3.10% thông tin (chủ yếu là nhiễu hoặc sự dư thừa dữ liệu).

# Câu 2: Lợi ích khi trực quan hóa 2D cho chuyên gia y tế:
# - Trực quan trực tiếp: Cho phép bác sĩ quan sát cụm bệnh nhân (clustering) và
#   phát hiện các mẫu bệnh lý trên biểu đồ 2 chiều thay vì phân tích bảng 4 chỉ số.
# - Phát hiện bất thường (Outlier Detection): Dễ dàng nhận diện các bệnh nhân
#   có chỉ số sức khỏe nguy cơ cao nằm tách biệt hẳn so với nhóm còn lại.
# - Giảm chi phí tính toán: Giúp các thuật toán phân loại/phân cụm phía sau chạy
#   nhanh hơn và tránh hiện tượng "lời nguyền chiều dữ liệu" (Curse of Dimensionality).
# ==============================================================================