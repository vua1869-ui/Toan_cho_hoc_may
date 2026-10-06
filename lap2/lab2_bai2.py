# ============================================
# LAB 2 - BÀI 2: Kiểm tra Độc lập tuyến tính
# trong không gian 2D
# ============================================

def is_linearly_dependent_2d(v1, v2):
    """
    Kiểm tra tính độc lập / phụ thuộc tuyến tính của 2 vector 2D.

    Nguyên lý: 2 vector phụ thuộc tuyến tính (cùng phương)
               <=> det([v1 v2]) = 0, với det = x1*y2 - x2*y1

    Trả về:
        True  : phụ thuộc tuyến tính (có thể loại bỏ 1 thuộc tính)
        False : độc lập tuyến tính
    """
    # Bước 1: Tính định thức
    det = v1[0] * v2[1] - v1[1] * v2[0]

    # Bước 2: So sánh với 0 (dùng sai số 1e-9 để tránh lỗi làm tròn số thực)
    if abs(det) < 1e-9:
        return True     # phụ thuộc tuyến tính
    return False        # độc lập tuyến tính


# ---------- Kiểm tra ----------
print(is_linearly_dependent_2d([2, 4], [4, 8]))   # True  (det = 16 - 16 = 0)
print(is_linearly_dependent_2d([2, 4], [1, 5]))   # False (det = 10 - 4 = 6)