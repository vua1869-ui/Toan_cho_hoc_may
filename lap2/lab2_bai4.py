# LAB 2 - BÀI 4: Phép biến đổi Hình học
import math


def _round2(x):
    """Làm tròn đến 2 chữ số thập phân (+0.0 để loại -0.0 do sai số số thực)."""
    return round(x, 2) + 0.0


def scale_points(points, sx, sy):
    """
    Co giãn tập điểm 2D: hệ số sx theo trục x, sy theo trục y.
    Ma trận co giãn:  S = [[sx, 0 ],
                           [0,  sy]]
    Công thức: x' = sx*x,  y' = sy*y
    """
    S = [[sx, 0.0],
         [0.0, sy]]
    result = []
    for (x, y) in points:
        x_new = S[0][0] * x + S[0][1] * y
        y_new = S[1][0] * x + S[1][1] * y
        result.append([_round2(x_new), _round2(y_new)])
    return result


def rotate_points(points, angle_degrees):
    """
    Xoay tập điểm 2D quanh gốc tọa độ một góc angle_degrees (độ).
    Ma trận xoay:  R = [[cos(rad), -sin(rad)],
                        [sin(rad),  cos(rad)]]
    Công thức: x' = x*cos - y*sin,  y' = x*sin + y*cos
    """
    rad = math.radians(angle_degrees)
    R = [[math.cos(rad), -math.sin(rad)],
         [math.sin(rad),  math.cos(rad)]]
    result = []
    for (x, y) in points:
        x_new = R[0][0] * x + R[0][1] * y
        y_new = R[1][0] * x + R[1][1] * y
        result.append([_round2(x_new), _round2(y_new)])
    return result


# ---------- Dữ liệu mẫu ----------
points = [[1, 0], [0, 1], [2, 3]]

print("Tập điểm gốc:            ", points)
print("Co giãn (sx=2, sy=3):    ", scale_points(points, 2, 3))
print("Xoay 90 độ:              ", rotate_points(points, 90))
print("Xoay 30 độ:              ", rotate_points(points, 30))
