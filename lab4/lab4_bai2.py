def generate_subsets_backtracking(features):
    """
    Sinh tất cả các tập con đặc trưng (bao gồm tập rỗng)
    sử dụng thuật toán Quay lui (Backtracking).
    """
    all_subsets = []

    def backtrack(start_index, current_path):
        # Lưu tập con hiện tại vào kết quả
        all_subsets.append(list(current_path))

        # Duyệt qua các đặc trưng tiếp theo
        for i in range(start_index, len(features)):
            # 1. Chọn đặc trưng
            current_path.append(features[i])
            # 2. Đệ quy bước tiếp theo
            backtrack(i + 1, current_path)
            # 3. Quay lui (Undo lựa chọn)
            current_path.pop()

    backtrack(0, [])
    return all_subsets


if __name__ == "__main__":
    features = ['Age', 'Income', 'Score']
    subsets = generate_subsets_backtracking(features)
    print(f"Tổng số tập con sinh được (2^{len(features)} = {len(subsets)}): {len(subsets)}")
    for s in subsets:
        print(s)