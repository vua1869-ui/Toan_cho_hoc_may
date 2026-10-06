# LAB 2 - BÀI 5: Pipeline Tăng cường Dữ liệu Ảnh
import math


def matrix_multiply(A, B):
    """Nhân hai ma trận: A (m×k) nhân B (k×n) -> C (m×n)."""
    rows, cols = len(A), len(B[0])
    C = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            for t in range(len(B)):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def create_affine_matrix(sx, sy, angle_deg, tx, ty):
    """
    Tạo ma trận biến đổi Affine 3x3 gộp 3 phép biến đổi:
        Co giãn (Scaling) -> Xoay (Rotation) -> Tịnh tiến (Translation)
    theo Tọa độ đồng nhất (Homogeneous Coordinates).

    Điểm 2D (x, y) được biểu diễn thành (x, y, 1) nên mọi biến đổi
    đều trở thành phép NHÂN MA TRẬN:  p' = M · p

    M = T · R · S  (Co giãn trước -> Xoay -> Tịnh tiến sau)
    """
    rad = math.radians(angle_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    # Ma trận Co giãn 3x3
    S = [[sx,    0.0,   0.0],
         [0.0,   sy,    0.0],
         [0.0,   0.0,   1.0]]

    # Ma trận Xoay 3x3
    R = [[cos_r, -sin_r, 0.0],
         [sin_r,  cos_r, 0.0],
         [0.0,    0.0,   1.0]]

    # Ma trận Tịnh tiến 3x3
    T = [[1.0,   0.0,   tx],
         [0.0,   1.0,   ty],
         [0.0,   0.0,   1.0]]

    # Gộp thành một ma trận duy nhất: M = T · R · S
    return matrix_multiply(T, matrix_multiply(R, S))


def transform_bounding_box(bbox, affine_matrix):
    """
    Áp dụng ma trận Affine 3x3 lên toàn bộ đỉnh của Bounding Box.

    Với mỗi đỉnh [x, y]:
      1. Chuyển sang tọa độ đồng nhất:  [x, y, 1]
      2. Nhân ma trận:                  [x', y', w'] = M · [x, y, 1]
      3. Chuẩn hóa:                     (x'/w', y'/w')  (thường w' = 1)
    """
    new_bbox = []
    for (x, y) in bbox:
        # Bước 1: [x, y] -> [x, y, 1]
        p = [x, y, 1.0]

        # Bước 2: nhân ma trận 3x3
        out = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(3):
                out[i] += affine_matrix[i][j] * p[j]

        # Bước 3: chia cho w để trở về tọa độ 2D mới
        w = out[2] if abs(out[2]) > 1e-9 else 1.0
        new_bbox.append([round(out[0] / w, 2), round(out[1] / w, 2)])
    return new_bbox


# ---------- Ví dụ minh họa ----------
bbox = [[0, 0], [4, 0], [4, 3], [0, 3]]     # Bounding Box gồm 4 đỉnh

# Pipeline: Co giãn (2, 2) -> Xoay 30 độ -> Tịnh tiến (5, 3)
M = create_affine_matrix(sx=2, sy=2, angle_deg=30, tx=5, ty=3)

print("Ma trận Affine 3x3 (M = T·R·S):")
for row in M:
    print("   ", [round(v, 3) for v in row])

print("\nBounding Box gốc:        ", bbox)
print("Bounding Box sau biến đổi:", transform_bounding_box(bbox, M))


# ==========================================================================
# GIẢI THÍCH: Vì sao Tọa độ đồng nhất 3x3 giúp GPU xử lý song song nhanh hơn?
# ==========================================================================
# 1) Gộp mọi biến đổi về DUY NHẤT một phép nhân ma trận:
#    - Tịnh tiến (p' = p + t) KHÔNG biểu diễn được bằng ma trận 2x2
#      (nó là phép cộng, không phải phép nhân tuyến tính).
#    - Nhờ thêm chiều thứ 3 (w = 1), tịnh tiến trở thành phép nhân ma trận
#      3x3 -> cả 3 biến đổi (scale, rotate, translate) được NHÂN GỘP trước
#      thành duy nhất 1 ma trận M = T·R·S.
#
# 2) Hoàn hảo cho tính toán song song (SIMD/SIMT) của GPU:
#    - Sau khi gộp, biến đổi mỗi điểm chỉ là MỘT phép nhân ma trận-vector
#      (9 phép nhân + 6 phép cộng) và HOÀN TOÀN ĐỘC LẬP giữa các điểm
#      (không có phụ thuộc dữ liệu, không cần rẽ nhánh, không cần đồng bộ).
#    - GPU có hàng nghìn lõi xử lý, nên hàng nghìn điểm ảnh/đỉnh có thể
#      được biến đổi ĐỒNG THỜI trong cùng một lúc.
#
# 3) Giảm chi phí bộ nhớ và truyền dữ liệu:
#    - Chỉ cần nạp 1 ma trận 3x3 (9 số) vào shared memory/registers,
#      mọi thread dùng chung -> giảm băng thông truy cập bộ nhớ toàn cục.
#
# -> Đó là lý do TensorFlow, PyTorch, OpenCV (CUDA)... đều dùng tọa độ
#    đồng nhất cho pipeline Data Augmentation và biến đổi hình học trên GPU.
# ==========================================================================
