def generate_binary_strings(n):
    """
    Sinh toàn bộ 2^n chuỗi nhị phân theo thứ tự từ điển 
    bằng Thuật toán Sinh (vòng lặp thuần, không đệ quy).
    """
    results = []
    a = [0] * n  # Bước 1: Cấu hình đầu tiên toàn số 0

    while True:
        # Bước 3: Lưu cấu hình hiện tại
        results.append("".join(str(bit) for bit in a))

        # Bước 4: Tìm bit 0 đầu tiên từ phải sang trái
        i = n - 1
        while i >= 0 and a[i] == 1:
            a[i] = 0
            i -= 1

        # Bước 5: Điều kiện dừng (đã đạt toàn bit 1)
        if i < 0:
            break

        # Bước 6: Đổi bit 0 thành 1
        a[i] = 1

    return results


if __name__ == "__main__":
    n = 3
    binary_list = generate_binary_strings(n)
    print(f"Tổng số chuỗi nhị phân sinh được (2^{n} = {len(binary_list)}): {len(binary_list)}")
    print("Danh sách chuỗi:", binary_list)