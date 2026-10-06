# ============================================
# LAB 2 - BÀI 3: Tìm Hạt nhân Ker(f) và Số chiều
# Nullity qua Hệ thuần nhất A·x = 0 (f: R^3 -> R^2)
# ============================================

def rref(A):
    """
    Đưa ma trận A về dạng bậc thang rút gọn (RREF)
    bằng thuật toán khử Gauss - Jordan (biến đổi dòng sơ cấp).
    """
    R = [row[:] for row in A]          # bản sao, không làm thay đổi A gốc
    rows, cols = len(R), len(R[0])
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break

        # 1) Tìm dòng có phần tử trụ (pivot) khác 0 ở cột 'lead'
        i = r
        while abs(R[i][lead]) < 1e-9:
            i += 1
            if i == rows:              # cả cột = 0 -> sang cột kế tiếp
                i = r
                lead += 1
                if lead == cols:
                    return R

        # 2) Đổi chỗ để đưa dòng chứa trụ lên vị trí r
        R[r], R[i] = R[i], R[r]

        # 3) Chuẩn hóa dòng trụ: chia cả dòng cho phần tử trụ -> trụ = 1
        pivot = R[r][lead]
        R[r] = [x / pivot for x in R[r]]

        # 4) Khử phần tử cùng cột ở các dòng còn lại (khử Gauss-Jordan)
        for i in range(rows):
            if i != r and abs(R[i][lead]) > 1e-9:
                factor = R[i][lead]
                R[i] = [R[i][j] - factor * R[r][j] for j in range(cols)]

        lead += 1

    return R


def find_kernel_basis_2x3(A):
    """
    Tìm vector cơ sở của Ker(f) = {x : A·x = 0} và số chiều dim(Ker).

    Các bước:
      1. Đưa A về RREF:  [1 0 c1]
                         [0 1 c2]
      2. Đặt biến tự do x3 = 1.0  ->  x1 = -c1, x2 = -c2
      3. Vector cơ sở: [-c1, -c2, 1.0],  dim(Ker) = 3 - rank(A)

    Trả về: (basis, dim_ker)
    """
    R = rref(A)

    print("Ma trận A sau khi đưa về RREF:")
    for row in R:
        print("   ", [round(x, 4) for x in row])

    rows, cols = len(R), len(R[0])

    # Xác định các cột trụ (có leading 1)
    pivot_cols = []
    r = 0
    for c in range(cols):
        if r < rows and abs(R[r][c] - 1.0) < 1e-9:
            pivot_cols.append(c)
            r += 1

    # Các cột còn lại là biến tự do
    free_cols = [c for c in range(cols) if c not in pivot_cols]

    # Với mỗi biến tự do, dựng một vector cơ sở của hạt nhân
    basis = []
    for fc in free_cols:
        x = [0.0] * cols
        x[fc] = 1.0                                  # đặt biến tự do = 1
        for r_idx, pc in enumerate(pivot_cols):
            x[pc] = -R[r_idx][fc]                    # biến trụ = -hệ số tương ứng
        basis.append(x)

    dim_ker = len(basis)                             # Nullity = số biến tự do
    return basis, dim_ker


# ---------- Ví dụ: A là ma trận 2x3 (f: R^3 -> R^2) ----------
A = [
    [1, 1, -2],
    [2, 1,  1]
]

basis, dim_ker = find_kernel_basis_2x3(A)

print("\nVector cơ sở của Ker(f):", basis)
print("Số chiều Nullity dim(Ker):", dim_ker)

# Kiểm chứng: A·x phải bằng 0
for x in basis:
    check = [sum(A[i][j] * x[j] for j in range(3)) for i in range(2)]
    print("Kiểm chứng A·x =", [round(v, 9) for v in check])