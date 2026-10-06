def solve_n_queens(n):
    """
    Giải bài toán Xếp N Quân hậu sử dụng Quay lui kết hợp Cắt tỉa nhánh cận O(1).
    """
    solutions = []
    cols = set()      # Quản lý các cột đã có quân hậu
    diag1 = set()     # Đường chéo chính: r - c
    diag2 = set()     # Đường chéo phụ: r + c

    board = [['.'] * n for _ in range(n)]

    def backtrack(r):
        if r == n:
            # Lưu cấu hình bàn cờ hiện tại
            solutions.append(["".join(row) for row in board])
            return

        for c in range(n):
            # Cắt tỉa nhánh cận nếu cột hoặc đường chéo bị đụng độ
            if c in cols or (r - c) in diag1 or (r + c) in diag2:
                continue

            # Đặt quân hậu
            cols.add(c)
            diag1.add(r - c)
            diag2.add(r + c)
            board[r][c] = 'Q'

            # Gọi đệ quy cho hàng tiếp theo
            backtrack(r + 1)

            # Quay lui (Undo)
            cols.remove(c)
            diag1.remove(r - c)
            diag2.remove(r + c)
            board[r][c] = '.'

    backtrack(0)
    return solutions


if __name__ == "__main__":
    for n in [4, 8]:
        solutions = solve_n_queens(n)
        print(f"\n=== BÀI TOÁN {n}-QUEENS ===")
        print(f"Tổng số cách xếp hợp lệ: {len(solutions)}")
        if n == 4:
            print("Các cách biểu diễn bàn cờ:")
            for idx, sol in enumerate(solutions, 1):
                print(f"Cách {idx}:")
                for row in sol:
                    print(row)
                print()