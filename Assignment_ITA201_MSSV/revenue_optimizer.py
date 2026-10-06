"""Module 4: Quy hoạch tuyến tính (duyệt đỉnh), dạng chính tắc, đối ngẫu và Gradient Descent.

LƯU Ý: đề không ghi số cụ thể của ràng buộc ngân sách / nhân sự, nên các số dưới đây là GIẢ ĐỊNH
minh họa. Muốn đổi sang số của thầy, chỉ cần sửa PRIMAL_A, PRIMAL_B, PRIMAL_C ngay bên dưới.
"""
from itertools import combinations

from linalg_utils import solve_linear_system

# max f = 50*x1 + 40*x2   (x1: ngân sách Facebook Ads, x2: Google Ads, đơn vị: triệu đồng)
PRIMAL_C = [50, 40]
PRIMAL_A = [[3, 2],      # Ngân sách: 3*x1 + 2*x2 <= 120
            [1, 2]]      # Nhân sự  : 1*x1 + 2*x2 <= 60   (giờ làm việc)
PRIMAL_B = [120, 60]
CONSTRAINT_NAMES = ["Ngân sách", "Nhân sự"]


# ---------- 4.1 Duyệt đỉnh hình học ----------
def solve_lp_2d(c, constraints, maximize=True, tol=1e-9):
    """Giải LP 2 biến bằng duyệt đỉnh.

    constraints: danh sách (a1, a2, dấu, b) với dấu là "<=" hoặc ">=". Đã gồm cả x1>=0, x2>=0.
    Mỗi đỉnh là giao điểm của 2 đường biên; giữ đỉnh thỏa mọi ràng buộc, chọn đỉnh tốt nhất.
    """
    vertices = []
    for (a1, a2, _, b1), (c1, c2, _, b2) in combinations(constraints, 2):
        if abs(a1 * c2 - a2 * c1) < tol:      # hai đường song song
            continue
        x = solve_linear_system([[a1, a2], [c1, c2]], [b1, b2])
        if all((p * x[0] + q * x[1] <= b + tol) if s == "<=" else (p * x[0] + q * x[1] >= b - tol)
               for p, q, s, b in constraints):
            if not any(abs(x[0] - v[0]) < tol and abs(x[1] - v[1]) < tol for v in vertices):
                vertices.append(x)
    if not vertices:
        return None, None, []
    key = lambda v: c[0] * v[0] + c[1] * v[1]
    best = max(vertices, key=key) if maximize else min(vertices, key=key)
    return best, key(best), vertices


def primal_constraints(A, b):
    cons = [(row[0], row[1], "<=", bi) for row, bi in zip(A, b)]
    return cons + [(1, 0, ">=", 0), (0, 1, ">=", 0)]


# ---------- 4.2 Dạng chính tắc & đối ngẫu ----------
def standard_form(A, b):
    """Thêm biến bù s_i để đổi bất đẳng thức <= thành đẳng thức. Trả về các hàng [x1, x2, s1.., | b]."""
    m = len(A)
    rows = []
    for i in range(m):
        slack = [1 if j == i else 0 for j in range(m)]
        rows.append((list(A[i]) + slack, b[i]))
    return rows


def make_dual(c, A, b):
    """Primal: max c.x, A x <= b, x >= 0   ->   Dual: min b.y, A^T y >= c, y >= 0."""
    m, n = len(A), len(A[0])
    dual_c = list(b)
    dual_cons = [(A[0][j], A[1][j], ">=", c[j]) for j in range(n)] if m == 2 else None
    dual_cons += [(1, 0, ">=", 0), (0, 1, ">=", 0)]
    return dual_c, dual_cons


# ---------- 4.3 Gradient Descent 1D ----------
def loss(w):
    return w * w - 6 * w + 9


def grad(w):
    return 2 * w - 6


def gradient_descent(w0=0.0, lr=0.1, max_iter=1000, tol=1e-8):
    """w <- w - lr * L'(w). Trả về (w, lịch sử [(bước, w, L(w), L'(w))])."""
    w, history = w0, []
    for t in range(max_iter):
        g = grad(w)
        history.append((t, w, loss(w), g))
        if abs(g) < tol:
            break
        w -= lr * g
    return w, history


def demo():
    print("=== MODULE 4: TỐI ƯU NGÂN SÁCH QUẢNG CÁO & GRADIENT DESCENT ===")
    print("\n[4.1] Quy hoạch tuyến tính - duyệt đỉnh (ràng buộc là số giả định, xem đầu file):")
    print(f"  max f = {PRIMAL_C[0]}*x1 + {PRIMAL_C[1]}*x2")
    for name, row, bi in zip(CONSTRAINT_NAMES, PRIMAL_A, PRIMAL_B):
        print(f"    {name}: {row[0]}*x1 + {row[1]}*x2 <= {bi}")
    print("    x1 >= 0, x2 >= 0")
    best, val, verts = solve_lp_2d(PRIMAL_C, primal_constraints(PRIMAL_A, PRIMAL_B))
    print("  Các đỉnh của miền khả thi:")
    for v in verts:
        print(f"    ({v[0]:7.2f}, {v[1]:7.2f})  ->  f = {PRIMAL_C[0]*v[0] + PRIMAL_C[1]*v[1]:8.2f}")
    print(f"  => Tối ưu tại x1={best[0]:.2f} (Facebook), x2={best[1]:.2f} (Google), f_max = {val:.2f}")

    print("\n[4.2] Dạng chính tắc (thêm biến bù):")
    for (row, bi), name in zip(standard_form(PRIMAL_A, PRIMAL_B), CONSTRAINT_NAMES):
        print(f"    {name}: {row[0]}*x1 + {row[1]}*x2 + {row[2]}*s1 + {row[3]}*s2 = {bi}")
    print("    x1, x2, s1, s2 >= 0")
    dual_c, dual_cons = make_dual(PRIMAL_C, PRIMAL_A, PRIMAL_B)
    print(f"  Bài toán đối ngẫu: min g = {dual_c[0]}*y1 + {dual_c[1]}*y2")
    for j in range(2):
        print(f"    {PRIMAL_A[0][j]}*y1 + {PRIMAL_A[1][j]}*y2 >= {PRIMAL_C[j]}")
    print("    y1, y2 >= 0")
    ybest, gval, _ = solve_lp_2d(dual_c, dual_cons, maximize=False)
    print(f"  Nghiệm đối ngẫu: y1={ybest[0]:.2f}, y2={ybest[1]:.2f}, g_min = {gval:.2f}")
    print(f"  Đối ngẫu mạnh: f_max = g_min ? {abs(val - gval) < 1e-6}"
          f"  (y là 'giá bóng': thêm 1 đơn vị ngân sách tăng ~{ybest[0]:.1f} lượt tiếp cận)")

    print("\n[4.3] Gradient Descent cho L(w) = w^2 - 6w + 9 (L'(w) = 2w - 6):")
    w, hist = gradient_descent(0.0, 0.1)
    print("  lr=0.1, w0=0. Vài bước đầu:")
    for t, wt, lt, gt in hist[:5]:
        print(f"    bước {t}: w={wt:.4f}, L={lt:.4f}, L'={gt:.4f}")
    print(f"  Hội tụ sau {len(hist)} bước: w* = {w:.6f}, L(w*) = {loss(w):.2e}  (nghiệm đúng: w=3, L=0)")
    for lr in (0.5, 1.1):
        w2, h2 = gradient_descent(0.0, lr, max_iter=50)
        status = "hội tụ" if abs(grad(w2)) < 1e-6 else "PHÂN KỲ"
        print(f"  lr={lr}: sau {len(h2)} bước w={w2:.4g} ({status})")


if __name__ == "__main__":
    demo()
