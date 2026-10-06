"""Module 2: Lọc đánh giá spam, Entropy & Information Gain (Bài 4, 5)."""
import math
from collections import Counter, defaultdict

from sample_data import REVIEWS, NEW_REVIEWS, BUY_ROWS, BUY_FEATURES


# ---------- Chức năng 2.1: sinh tổ hợp đặc trưng bằng quay lui (Bài 4) ----------
def generate_subsets(items):
    """Sinh toàn bộ 2^n tập con bằng backtracking."""
    result = []

    def backtrack(start, current):
        result.append(tuple(current))
        for i in range(start, len(items)):
            current.append(items[i])
            backtrack(i + 1, current)
            current.pop()          # quay lui

    backtrack(0, [])
    return result


# ---------- Chức năng 2.3: Entropy & Information Gain (Bài 5) ----------
def entropy(labels):
    """H(S) = -sum p_i * log2(p_i)."""
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(labels).values())


def information_gain(rows, feature, label_key="nhan"):
    """IG(S, A) = H(S) - sum |S_v|/|S| * H(S_v)."""
    return joint_information_gain(rows, [feature], label_key)


def joint_information_gain(rows, features, label_key="nhan"):
    """Độ tăng thông tin khi chia dữ liệu theo đồng thời nhiều thuộc tính."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] for f in features)].append(r[label_key])
    n = len(rows)
    remainder = sum(len(g) / n * entropy(g) for g in groups.values())
    return entropy([r[label_key] for r in rows]) - remainder


def best_feature_subset(rows, features, label_key="nhan", penalty=0.1, max_size=3):
    """Duyệt mọi tập con (quay lui), chấm điểm = IG - penalty*kích thước. Trả về danh sách xếp hạng."""
    ranking = []
    for subset in generate_subsets(features):
        if 0 < len(subset) <= max_size:
            gain = joint_information_gain(rows, list(subset), label_key)
            ranking.append((gain - penalty * len(subset), gain, subset))
    ranking.sort(reverse=True)
    return ranking


# ---------- Chức năng 2.2: Naive Bayes + Laplace (Bài 5) ----------
def tokenize(text):
    """Tách từ thủ công: chữ thường, ký tự không phải chữ/số thành khoảng trắng."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return cleaned.split()


class NaiveBayesSpam:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # hệ số làm mượt Laplace

    def fit(self, texts, labels):
        self.classes = sorted(set(labels))
        self.doc_count = Counter(labels)
        self.word_count = {c: Counter() for c in self.classes}
        self.vocab = set()
        for text, label in zip(texts, labels):
            for w in tokenize(text):
                self.word_count[label][w] += 1
                self.vocab.add(w)
        self.total_words = {c: sum(self.word_count[c].values()) for c in self.classes}
        return self

    def log_scores(self, text):
        n_docs = sum(self.doc_count.values())
        V = len(self.vocab)
        scores = {}
        for c in self.classes:
            score = math.log(self.doc_count[c] / n_docs)
            for w in tokenize(text):
                # P(w|c) = (count + alpha) / (tổng từ của lớp + alpha * |V|)
                score += math.log((self.word_count[c][w] + self.alpha) /
                                  (self.total_words[c] + self.alpha * V))
            scores[c] = score
        return scores

    def predict_proba(self, text):
        scores = self.log_scores(text)
        m = max(scores.values())
        exp = {c: math.exp(s - m) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}

    def predict(self, text):
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)


def demo():
    print("=== MODULE 2: SINH TẬP CON, NAIVE BAYES & ENTROPY/IG ===")
    subsets = generate_subsets(BUY_FEATURES)
    print(f"\n[2.1] Quay lui sinh {len(subsets)} tập con của {BUY_FEATURES}:")
    for s in subsets:
        print("  ", s)

    texts = [t for t, _ in REVIEWS]
    labs = [l for _, l in REVIEWS]
    nb = NaiveBayesSpam(alpha=1.0).fit(texts, labs)
    print(f"\n[2.2] Naive Bayes (Laplace alpha=1), từ vựng {len(nb.vocab)} từ:")
    for t in NEW_REVIEWS:
        probs = nb.predict_proba(t)
        print(f"  '{t}' -> {nb.predict(t)}  "
              + ", ".join(f"P({c})={p:.3f}" for c, p in probs.items()))

    labels = [r["nhan"] for r in BUY_ROWS]
    print(f"\n[2.3] Entropy của tập dữ liệu: H(S) = {entropy(labels):.4f} bit")
    for f in ("tuoi", "thu_nhap", "voucher"):
        print(f"  IG theo '{f}': {information_gain(BUY_ROWS, f):.4f}")
    print("  Top 3 tập thuộc tính tốt nhất (điểm = IG - 0.1*số thuộc tính):")
    for score, gain, subset in best_feature_subset(BUY_ROWS, BUY_FEATURES)[:3]:
        print(f"    {subset}: IG={gain:.4f}, điểm={score:.4f}")


if __name__ == "__main__":
    demo()
