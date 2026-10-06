"""Dữ liệu mẫu nhỏ lấy bối cảnh cửa hàng thời trang online (AURA Studio / POLY-MART)."""

# --- Module 1: ma trận khách hàng X (M x N) ---
FEATURE_NAMES = ["so_don_thang", "chi_tieu_trieu", "luot_xem_tuan", "ti_le_voucher_%"]
CUSTOMERS = {
    "An":    [2, 0.8, 40, 80],
    "Binh":  [3, 1.1, 35, 70],
    "Chi":   [2, 0.9, 45, 90],
    "Dung":  [8, 6.5, 20, 30],
    "Em":    [7, 5.8, 25, 20],
    "Giang": [9, 7.2, 18, 25],
    "Ha":    [1, 0.3, 10, 10],
    "Khoa":  [1, 0.4, 12, 5],
}

# --- Module 2: đánh giá spam / thường ---
REVIEWS = [
    ("áo đẹp vải mềm mặc rất thoải mái", "thuong"),
    ("giao hàng nhanh đóng gói cẩn thận", "thuong"),
    ("size vừa vặn màu đúng như hình", "thuong"),
    ("chất vải mát mặc đi làm rất ổn", "thuong"),
    ("đầm đẹp form chuẩn shop tư vấn nhiệt tình", "thuong"),
    ("quần jeans hơi dài nhưng chất vải tốt", "thuong"),
    ("sản phẩm giống mô tả sẽ ủng hộ shop lần sau", "thuong"),
    ("giao hơi chậm nhưng áo đẹp", "thuong"),
    ("click link nhận quà miễn phí ngay", "spam"),
    ("kiếm tiền online tại nhà liên hệ zalo ngay", "spam"),
    ("mua 1 tặng 1 nhấn vào link giảm giá sốc", "spam"),
    ("vay tiền nhanh lãi suất thấp liên hệ ngay", "spam"),
    ("khuyến mãi sốc click link để nhận quà", "spam"),
    ("tuyển cộng tác viên thu nhập cao liên hệ zalo", "spam"),
]
NEW_REVIEWS = [
    "áo mặc rất thoải mái giao hàng nhanh",
    "click link nhận quà liên hệ zalo ngay",
]

# --- Module 2: dữ liệu tính Entropy / Information Gain (nhãn: có mua hay không) ---
BUY_FEATURES = ["tuoi", "thu_nhap", "voucher"]
BUY_ROWS = [
    {"tuoi": "<25",   "thu_nhap": "thap", "voucher": "co",    "nhan": "mua"},
    {"tuoi": "<25",   "thu_nhap": "thap", "voucher": "khong", "nhan": "khong"},
    {"tuoi": "<25",   "thu_nhap": "tb",   "voucher": "co",    "nhan": "mua"},
    {"tuoi": "<25",   "thu_nhap": "tb",   "voucher": "khong", "nhan": "khong"},
    {"tuoi": "25-35", "thu_nhap": "tb",   "voucher": "co",    "nhan": "mua"},
    {"tuoi": "25-35", "thu_nhap": "cao",  "voucher": "khong", "nhan": "mua"},
    {"tuoi": "25-35", "thu_nhap": "cao",  "voucher": "co",    "nhan": "mua"},
    {"tuoi": "25-35", "thu_nhap": "tb",   "voucher": "khong", "nhan": "mua"},
    {"tuoi": "25-35", "thu_nhap": "thap", "voucher": "khong", "nhan": "khong"},
    {"tuoi": ">35",   "thu_nhap": "cao",  "voucher": "khong", "nhan": "mua"},
    {"tuoi": ">35",   "thu_nhap": "cao",  "voucher": "co",    "nhan": "mua"},
    {"tuoi": ">35",   "thu_nhap": "tb",   "voucher": "khong", "nhan": "khong"},
    {"tuoi": ">35",   "thu_nhap": "thap", "voucher": "co",    "nhan": "khong"},
    {"tuoi": ">35",   "thu_nhap": "thap", "voucher": "khong", "nhan": "khong"},
]

# --- Module 3: mạng bạn bè, Markov, đồ thị kho - cửa hàng ---
PEOPLE = ["An", "Binh", "Chi", "Dung", "Em", "Giang", "Ha", "Khoa", "Lan"]
FRIEND_EDGES = [("An", "Binh"), ("Binh", "Chi"), ("Chi", "Dung"), ("An", "Em"),
                ("Em", "Dung"), ("Giang", "Ha"), ("Ha", "Khoa")]

MARKOV_STATES = ["Xem hang", "Mua hang", "Roi di"]
MARKOV_P = [
    [0.50, 0.20, 0.30],   # từ Xem hàng
    [0.30, 0.40, 0.30],   # từ Mua hàng (mua xong có thể xem tiếp / mua tiếp / rời đi)
    [0.20, 0.10, 0.70],   # từ Rời đi (có thể quay lại)
]

WAREHOUSE_NODES = ["K1", "K2", "C1", "C2", "C3", "C4"]
WAREHOUSE_EDGES = [("K1", "C1"), ("K1", "C2"), ("K2", "C2"), ("K2", "C3"), ("K2", "C4")]
WAREHOUSE_EDGES_BAD = WAREHOUSE_EDGES + [("C1", "C2")]  # tạo chu trình lẻ K1-C1-C2
