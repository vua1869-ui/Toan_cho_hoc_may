"""Chương trình chính: menu console chạy demo toàn bộ hệ thống POLY-MART / AURA Studio."""
import customer_analytics
import review_classifier
import behavior_network
import revenue_optimizer

MENU = """
--- SMART DATA PIPELINE - TOÁN CHO HỌC MÁY (ITA201) ---
1. Demo Module 1: Chuẩn hóa khoảng cách & Nén chiều PCA
2. Demo Module 2: Sinh tập con, Naive Bayes & Tính Entropy/IG
3. Demo Module 3: Duyệt đồ thị BFS/DFS & Dự báo Markov
4. Demo Module 4: Phân bổ ngân sách LP & Gradient Descent
5. Chạy toàn bộ Pipeline tổng hợp
0. Thoát chương trình"""

ACTIONS = {
    "1": customer_analytics.demo,
    "2": review_classifier.demo,
    "3": behavior_network.demo,
    "4": revenue_optimizer.demo,
}


def run_all():
    for key in ("1", "2", "3", "4"):
        ACTIONS[key]()
        print()


def main():
    while True:
        print(MENU)
        try:
            choice = input("Lựa chọn của bạn: ").strip()
        except EOFError:
            print("\nHết dữ liệu nhập, thoát chương trình.")
            return
        if choice == "0":
            print("Tạm biệt!")
            return
        if choice == "5":
            run_all()
        elif choice in ACTIONS:
            print()
            ACTIONS[choice]()
        else:
            print("Lựa chọn không hợp lệ, vui lòng nhập 0-5.")
        print()


if __name__ == "__main__":
    main()
