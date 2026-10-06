"""Module 3: Đồ thị bạn bè, chuỗi Markov hành vi và đồ thị hai phía (Python thuần)."""
from collections import deque

from linalg_utils import mat_mul, identity, vec_mat, solve_linear_system, transpose
from sample_data import (PEOPLE, FRIEND_EDGES, MARKOV_STATES, MARKOV_P,
                         WAREHOUSE_NODES, WAREHOUSE_EDGES, WAREHOUSE_EDGES_BAD)


# ---------- Đồ thị: danh sách kề ----------
def build_graph(nodes, edges):
    """Đồ thị vô hướng dạng danh sách kề."""
    g = {n: [] for n in nodes}
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)
    return g


# ---------- 3.1 BFS: đường đi ngắn nhất (theo số cạnh) ----------
def bfs_shortest_path(graph, start, goal):
    """Trả về danh sách đỉnh trên đường ngắn nhất start -> goal, hoặc None nếu không liên thông."""
    if start == goal:
        return [start]
    parent = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in parent:
                parent[v] = u
                if v == goal:
                    path = [v]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]
                queue.append(v)
    return None


# ---------- 3.2 DFS: đếm số cụm bạn bè (thành phần liên thông) ----------
def dfs_components(graph):
    """Duyệt DFS (dùng ngăn xếp tự quản lý để tránh tràn đệ quy), trả về danh sách các cụm."""
    visited, components = set(), []
    for s in graph:
        if s in visited:
            continue
        comp, stack = [], [s]
        visited.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)
    return components


# ---------- 3.3 Markov ----------
def check_stochastic(P, tol=1e-9):
    """Mỗi hàng của ma trận chuyển phải có tổng bằng 1 và không âm."""
    return all(abs(sum(row) - 1) < tol and min(row) >= 0 for row in P)


def matrix_power(P, k):
    """P^k bằng bình phương liên tiếp."""
    result = identity(len(P))
    base = [row[:] for row in P]
    while k > 0:
        if k & 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        k >>= 1
    return result


def state_after_k_days(P, start_dist, k):
    """Phân phối trạng thái sau k ngày: pi_k = pi_0 * P^k."""
    return vec_mat(start_dist, matrix_power(P, k))


def stationary_distribution(P):
    """Giải pi*P = pi, tổng pi = 1: thay một phương trình của (P^T - I) bằng phương trình tổng = 1."""
    n = len(P)
    A = [[a - (1.0 if i == j else 0.0) for j, a in enumerate(row)] for i, row in enumerate(transpose(P))]
    A[-1] = [1.0] * n
    b = [0.0] * (n - 1) + [1.0]
    return solve_linear_system(A, b)


# ---------- 3.4 Đồ thị hai phía: BFS tô 2 màu ----------
def is_bipartite(graph):
    """Trả về (True, màu của từng đỉnh) hoặc (False, cạnh gây mâu thuẫn)."""
    color = {}
    for s in graph:
        if s in color:
            continue
        color[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, (u, v)
    return True, color


def adjacency_matrix(nodes, edges):
    """Ma trận kề (0/1) của đồ thị vô hướng."""
    idx = {n: i for i, n in enumerate(nodes)}
    M = [[0] * len(nodes) for _ in nodes]
    for a, b in edges:
        M[idx[a]][idx[b]] = M[idx[b]][idx[a]] = 1
    return M


def demo():
    print("=== MODULE 3: MẠNG HÀNH VI & CHUỖI MARKOV ===")
    g = build_graph(PEOPLE, FRIEND_EDGES)
    print("\n[3.1] Biểu diễn đồ thị bạn bè")
    print("  Danh sách kề:")
    for n, nb in g.items():
        print(f"    {n:>6}: {nb}")
    print("  Ma trận kề:")
    print("          " + " ".join(f"{n[:3]:>3}" for n in PEOPLE))
    for n, row in zip(PEOPLE, adjacency_matrix(PEOPLE, FRIEND_EDGES)):
        print(f"    {n:>6} " + " ".join(f"{v:>3}" for v in row))
    print("  BFS - đường đi ngắn nhất:")
    for a, b in [("An", "Dung"), ("Binh", "Em"), ("An", "Ha")]:
        p = bfs_shortest_path(g, a, b)
        print(f"    {a} -> {b}: " + (" -> ".join(p) + f"  ({len(p)-1} bước)" if p else "không liên thông"))
    comps = dfs_components(g)
    print(f"  DFS - có {len(comps)} cụm cộng đồng liên thông:")
    for i, c in enumerate(comps, 1):
        print(f"    Cụm {i}: {sorted(c)}")

    print("\n[3.2] Chuỗi Markov hành vi (Xem hàng / Mua hàng / Rời đi)")
    print("  Ma trận chuyển P hợp lệ (mỗi hàng tổng = 1):", check_stochastic(MARKOV_P))
    pi0 = [1.0, 0.0, 0.0]
    print("  Bắt đầu từ trạng thái 'Xem hàng':")
    for k in (1, 3, 7, 30):
        d = state_after_k_days(MARKOV_P, pi0, k)
        print(f"    Sau {k:>2} ngày: " + ", ".join(f"{s}={v:.4f}" for s, v in zip(MARKOV_STATES, d)))
    st = stationary_distribution(MARKOV_P)
    print("  Trạng thái dừng pi*:", ", ".join(f"{s}={v:.4f}" for s, v in zip(MARKOV_STATES, st)))

    print("\n[3.3] Kiểm tra đồ thị hai phía (mạng kho hàng - cửa hàng), tô màu BFS")
    for name, edges in [("Mạng hợp lệ", WAREHOUSE_EDGES), ("Mạng có cạnh lỗi C1-C2", WAREHOUSE_EDGES_BAD)]:
        ok, info = is_bipartite(build_graph(WAREHOUSE_NODES, edges))
        if ok:
            print(f"  {name}: LÀ đồ thị hai phía. Kho={[n for n, c in info.items() if c == 0]}, "
                  f"Cửa hàng={[n for n, c in info.items() if c == 1]}")
        else:
            print(f"  {name}: KHÔNG là đồ thị hai phía, mâu thuẫn ở cạnh {info}")


if __name__ == "__main__":
    demo()
