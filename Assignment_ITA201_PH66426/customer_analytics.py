"""Module 1: Tiền xử lý ma trận & nén chiều dữ liệu PCA (Bài 1, 2, 3).

Mọi thuật toán tự cài đặt bằng Python thuần, không dùng numpy.
"""
import math

from linalg_utils import dot, norm, transpose, mat_vec, mat_mul, print_matrix
from sample_data import CUSTOMERS, FEATURE_NAMES


# ---------- Chức năng 1.1: khoảng cách giữa các khách hàng (Bài 1) ----------
def l1_distance(u, v):
    """Khoảng cách Manhattan: tổng |u_i - v_i|."""
    return sum(abs(a - b) for a, b in zip(u, v))


def l2_distance(u, v):
    """Khoảng cách Euclid: căn của tổng (u_i - v_i)^2."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def distance_matrix(X, metric=l2_distance):
    """Ma trận khoảng cách M x M giữa mọi cặp khách hàng."""
    return [[metric(a, b) for b in X] for a in X]


def nearest_customer(X, i, metric=l2_distance):
    """Chỉ số khách hàng gần khách hàng i nhất (không tính chính nó)."""
    candidates = [(metric(X[i], X[j]), j) for j in range(len(X)) if j != i]
    return min(candidates)[1]


# ---------- Chức năng 1.2: biến đổi A * x (Bài 2) ----------
def transform_all(A, X):
    """Áp dụng phép biến đổi tuyến tính A lên từng vector khách hàng x."""
    return [mat_vec(A, x) for x in X]


# ---------- Chức năng 1.3: PCA thuần (Bài 3) ----------
def column_means(X):
    M = len(X)
    return [sum(col) / M for col in zip(*X)]


def mean_center(X):
    """Bước 1: trừ trung bình từng cột. Trả về (Xc, means)."""
    means = column_means(X)
    return [[x - m for x, m in zip(row, means)] for row in X], means


def covariance_matrix(Xc):
    """Bước 2: Cov = (1/(M-1)) * Xc^T * Xc  (ma trận N x N)."""
    M = len(Xc)
    C = mat_mul(transpose(Xc), Xc)
    return [[v / (M - 1) for v in row] for row in C]


def power_iteration(C, max_iter=100000, tol=1e-13):
    """Tìm trị riêng lớn nhất và vector riêng tương ứng của ma trận đối xứng C."""
    n = len(C)
    v = [1.0 / (i + 1) for i in range(n)]  # vector khởi tạo không đối xứng
    nv = norm(v)
    v = [x / nv for x in v]
    lam = 0.0
    for _ in range(max_iter):
        w = mat_vec(C, v)
        nw = norm(w)
        if nw < 1e-15:
            return 0.0, v
        w = [x / nw for x in w]
        new_lam = dot(w, mat_vec(C, w))  # thương Rayleigh
        v = w
        if abs(new_lam - lam) < tol:
            lam = new_lam
            break
        lam = new_lam
    # quy ước dấu: thành phần có |giá trị| lớn nhất dương
    big = max(range(n), key=lambda i: abs(v[i]))
    if v[big] < 0:
        v = [-x for x in v]
    return lam, v


def deflate(C, lam, v):
    """Khử trị riêng đã tìm: C' = C - lam * v v^T."""
    n = len(C)
    return [[C[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]


def top_eigenpairs(C, k=2):
    """k cặp (trị riêng, vector riêng) lớn nhất bằng power iteration + deflation."""
    pairs, work = [], [row[:] for row in C]
    for _ in range(k):
        lam, v = power_iteration(work)
        pairs.append((lam, v))
        work = deflate(work, lam, v)
    return pairs


def project(Xc, components):
    """Bước 3: chiếu dữ liệu đã trừ trung bình lên các trục chính (mỗi trục là 1 vector N chiều)."""
    return mat_mul(Xc, transpose(components))


def pca(X, k=2, components=None):
    """PCA đầy đủ. Nếu truyền `components` (PC1, PC2 cho sẵn) thì dùng luôn, ngược lại tự tính."""
    Xc, means = mean_center(X)
    C = covariance_matrix(Xc)
    if components is None:
        pairs = top_eigenpairs(C, k)
        eigenvalues = [p[0] for p in pairs]
        components = [p[1] for p in pairs]
    else:
        eigenvalues = [dot(v, mat_vec(C, v)) for v in components]
    trace = sum(C[i][i] for i in range(len(C)))
    return {
        "means": means,
        "cov": C,
        "eigenvalues": eigenvalues,
        "components": components,
        "explained": [lam / trace for lam in eigenvalues],
        "Z": project(Xc, components),
    }


def demo():
    names = list(CUSTOMERS)
    X = [CUSTOMERS[n] for n in names]
    print("=== MODULE 1: CHUẨN HÓA KHOẢNG CÁCH & NÉN CHIỀU PCA ===")
    print("Đặc trưng:", FEATURE_NAMES)
    print("\n[1.1] Ma trận khoảng cách Manhattan (L1):")
    print_matrix(distance_matrix(X, l1_distance), "{:7.1f}", names)
    print("\n[1.1] Ma trận khoảng cách Euclid (L2):")
    print_matrix(distance_matrix(X, l2_distance), "{:7.1f}", names)
    for i, n in enumerate(names[:3]):
        print(f"  Khách giống {n} nhất: {names[nearest_customer(X, i)]}")

    A = [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0]]
    print("\n[1.2] Biến đổi A*x (chọn 2 đặc trưng đầu + trung bình của chúng):")
    for n, z in zip(names[:3], transform_all(A, X[:3])):
        print(f"  {n}: {[round(v, 2) for v in z]}")

    res = pca(X, 2)
    print("\n[1.3] PCA tự cài đặt:")
    print("  Trung bình cột:", [round(v, 3) for v in res["means"]])
    print("  Ma trận hiệp phương sai (4x4):")
    print_matrix(res["cov"], "{:9.3f}")
    for i, (lam, v) in enumerate(zip(res["eigenvalues"], res["components"]), 1):
        print(f"  PC{i}: trị riêng={lam:.3f} ({res['explained'][i-1]*100:.1f}% phương sai), "
              f"vector={[round(x, 3) for x in v]}")
    print(f"  Hai trục giữ lại {sum(res['explained'])*100:.1f}% tổng phương sai")
    print("  Dữ liệu sau khi nén 4 chiều -> 2 chiều:")
    for n, z in zip(names, res["Z"]):
        print(f"    {n:>6}: PC1={z[0]:8.3f}  PC2={z[1]:8.3f}")


if __name__ == "__main__":
    demo()
