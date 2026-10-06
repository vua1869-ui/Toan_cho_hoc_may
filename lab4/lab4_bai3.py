def generate_permutations(n):
    """
    Sinh toàn bộ n! hoán vị của {1, 2, ..., n} theo thứ tự từ điển
    bằng thuật toán sinh (vòng lặp while, không dùng itertools).
    """
    a = list(range(1, n + 1))  # Cấu hình ban đầu: [1, 2, ..., n]
    permutations = []

    while True:
        permutations.append(list(a))

        # Bước 1: Tìm i lớn nhất sao cho a[i] < a[i + 1]
        i = n - 2
        while i >= 0 and a[i] >= a[i + 1]:
            i -= 1

        # Nếu không tìm thấy, đã đạt cấu hình cuối cùng (ví dụ: [3, 2, 1])
        if i < 0:
            break

        # Bước 2: Tìm k lớn nhất sao cho a[k] > a[i]
        k = n - 1
        while a[k] <= a[i]:
            k -= 1

        # Bước 3: Đổi chỗ a[i] và a[k]
        a[i], a[k] = a[k], a[i]

        # Bước 4: Lật ngược đoạn từ a[i + 1] đến a[n - 1]
        a[i + 1:] = reversed(a[i + 1:])

    return permutations


if __name__ == "__main__":
    n = 3
    perms = generate_permutations(n)
    print(f"Tổng số hoán vị ({n}! = {len(perms)}): {len(perms)}")
    for p in perms:
        print(p)