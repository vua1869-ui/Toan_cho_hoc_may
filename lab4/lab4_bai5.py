import math

def count_configurations(param_grid):
    """
    Tính số lượng cấu hình siêu tham số dựa trên Nguyên lý Nhân.
    """
    total = 1
    for key, values in param_grid.items():
        total *= len(values)
    return total


def custom_grid_search(param_grid):
    """
    Sinh toàn bộ các tổ hợp siêu tham số bằng Thuật toán Quay lui.
    """
    keys = list(param_grid.keys())
    all_configs = []

    def backtrack(index, current_config):
        if index == len(keys):
            all_configs.append(dict(current_config))
            return

        key = keys[index]
        for val in param_grid[key]:
            current_config[key] = val
            backtrack(index + 1, current_config)
            del current_config[key]  # Quay lui

    backtrack(0, {})
    return all_configs


def dirichlet_collision_analysis(num_models, num_buckets):
    """
    Phân tích đụng độ theo Nguyên lý Dirichlet mở rộng ceil(N / k).
    """
    return math.ceil(num_models / num_buckets)


if __name__ == "__main__":
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'optimizer': ['Adam', 'SGD']
    }

    # 1. Phân tích Nguyên lý Nhân
    total_calc = count_configurations(param_grid)
    print(f"1. Tổng số cấu hình (Nguyên lý Nhân): {total_calc}")

    # 2. Sinh Grid Search bằng Quay lui
    configs = custom_grid_search(param_grid)
    print(f"   Số lượng cấu hình sinh ra bằng Quay lui: {len(configs)}")
    print("   Ví dụ 3 cấu hình đầu tiên:")
    for cfg in configs[:3]:
        print("  ", cfg)

    # 3. Phân tích Đụng độ theo Nguyên lý Dirichlet
    N = 105  # Số lượng mô hình
    k = 10   # Số cụm máy chủ (Server Buckets)
    max_models_in_bucket = dirichlet_collision_analysis(N, k)

    print(f"\n3. Phân tích Nguyên lý Dirichlet:")
    print(f"   - Số mô hình N = {N}")
    print(f"   - Số cụm máy chủ k = {k}")
    print(f"   - Công thức Dirichlet mở rộng: ceil(N / k) = ceil({N} / {k}) = {max_models_in_bucket}")
    print(f"   => Kết luận: Chắc chắn có ít nhất 1 cụm máy chủ phải chịu tải tối thiểu {max_models_in_bucket} mô hình.")
    
    print("\n   [Ý nghĩa trong Cân bằng tải Hệ thống AI]")
    print("   - Dự báo điểm nghẽn (Bottleneck): Cho dù thuật toán băm (hashing) hoạt động tốt đến đâu, vẫn luôn tồn tại ít nhất 1 cụm máy chủ phải xử lý tải đỉnh (peak load).")
    print("   - Thiết kế hạ tầng: Giúp các kỹ sư AI / DevOps chủ động cấp phát dung lượng bộ nhớ và tính toán cho server để tránh tình trạng quá tải (Out of Memory/CPU bottleneck) khi lưu trữ/huấn luyện hàng loạt mô hình.")