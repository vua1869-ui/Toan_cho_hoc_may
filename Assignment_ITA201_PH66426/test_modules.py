"""Kiểm thử các module. Chạy: python test_modules.py  (hoặc python -m unittest -v)

numpy chỉ dùng TRONG TEST để đối chiếu kết quả; code lõi không import numpy.
"""
import math
import unittest

import customer_analytics as ca
import review_classifier as rc
import behavior_network as bn
import revenue_optimizer as ro
from sample_data import *


class TestModule1(unittest.TestCase):
    def test_distances(self):
        self.assertEqual(ca.l1_distance([1, 2, 3], [4, 0, 3]), 5)
        self.assertAlmostEqual(ca.l2_distance([0, 0], [3, 4]), 5.0)

    def test_matrix_transform(self):
        self.assertEqual(ca.transform_all([[1, 2], [3, 4]], [[1, 1]]), [[3, 7]])

    def test_pca_matches_numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("không có numpy để đối chiếu")
        X = [CUSTOMERS[n] for n in CUSTOMERS]
        res = ca.pca(X, 2)
        C = np.cov(np.array(X, dtype=float).T)             # chuẩn hóa 1/(M-1)
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        for i in range(2):
            self.assertAlmostEqual(res["eigenvalues"][i], vals[order[i]], places=4)
            ref = vecs[:, order[i]]
            mine = np.array(res["components"][i])
            self.assertAlmostEqual(abs(float(ref @ mine)), 1.0, places=5)  # cùng phương (bỏ qua dấu)

    def test_pca_given_components(self):
        X = [CUSTOMERS[n] for n in CUSTOMERS]
        auto = ca.pca(X, 2)
        given = ca.pca(X, 2, components=auto["components"])
        self.assertEqual(len(given["Z"]), len(X))
        self.assertAlmostEqual(given["Z"][0][0], auto["Z"][0][0])


class TestModule2(unittest.TestCase):
    def test_subsets_count(self):
        self.assertEqual(len(rc.generate_subsets([1, 2, 3, 4])), 16)

    def test_entropy(self):
        self.assertAlmostEqual(rc.entropy(["a", "a", "b", "b"]), 1.0)
        self.assertAlmostEqual(rc.entropy(["a"] * 5), 0.0)

    def test_information_gain_perfect_split(self):
        rows = [{"x": 0, "nhan": "a"}, {"x": 0, "nhan": "a"}, {"x": 1, "nhan": "b"}, {"x": 1, "nhan": "b"}]
        self.assertAlmostEqual(rc.information_gain(rows, "x"), 1.0)

    def test_naive_bayes(self):
        nb = rc.NaiveBayesSpam().fit([t for t, _ in REVIEWS], [l for _, l in REVIEWS])
        self.assertEqual(nb.predict(NEW_REVIEWS[0]), "thuong")
        self.assertEqual(nb.predict(NEW_REVIEWS[1]), "spam")
        self.assertAlmostEqual(sum(nb.predict_proba("áo đẹp").values()), 1.0)

    def test_laplace_unseen_word(self):
        nb = rc.NaiveBayesSpam().fit([t for t, _ in REVIEWS], [l for _, l in REVIEWS])
        p = nb.predict_proba("từlạchưatừngthấy")   # từ chưa gặp không được làm xác suất = 0
        self.assertTrue(all(v > 0 for v in p.values()))


class TestModule3(unittest.TestCase):
    def setUp(self):
        self.g = bn.build_graph(PEOPLE, FRIEND_EDGES)

    def test_bfs(self):
        self.assertEqual(len(bn.bfs_shortest_path(self.g, "An", "Dung")) - 1, 2)
        self.assertIsNone(bn.bfs_shortest_path(self.g, "An", "Ha"))
        self.assertEqual(bn.bfs_shortest_path(self.g, "An", "An"), ["An"])

    def test_dfs_components(self):
        self.assertEqual(len(bn.dfs_components(self.g)), 3)

    def test_markov(self):
        self.assertTrue(bn.check_stochastic(MARKOV_P))
        d = bn.state_after_k_days(MARKOV_P, [1, 0, 0], 1)
        self.assertEqual([round(x, 6) for x in d], [0.5, 0.2, 0.3])
        pi = bn.stationary_distribution(MARKOV_P)
        self.assertAlmostEqual(sum(pi), 1.0)
        again = bn.vec_mat(pi, MARKOV_P)
        for a, b in zip(pi, again):
            self.assertAlmostEqual(a, b)                      # pi * P = pi
        far = bn.state_after_k_days(MARKOV_P, [1, 0, 0], 200)
        for a, b in zip(pi, far):
            self.assertAlmostEqual(a, b, places=6)           # P^k hội tụ về pi*

    def test_bipartite(self):
        ok, _ = bn.is_bipartite(bn.build_graph(WAREHOUSE_NODES, WAREHOUSE_EDGES))
        bad, _ = bn.is_bipartite(bn.build_graph(WAREHOUSE_NODES, WAREHOUSE_EDGES_BAD))
        self.assertTrue(ok)
        self.assertFalse(bad)


class TestModule4(unittest.TestCase):
    def test_lp(self):
        best, val, _ = ro.solve_lp_2d(ro.PRIMAL_C, ro.primal_constraints(ro.PRIMAL_A, ro.PRIMAL_B))
        self.assertAlmostEqual(val, 2100.0)
        self.assertAlmostEqual(best[0], 30.0)
        self.assertAlmostEqual(best[1], 15.0)

    def test_strong_duality(self):
        _, val, _ = ro.solve_lp_2d(ro.PRIMAL_C, ro.primal_constraints(ro.PRIMAL_A, ro.PRIMAL_B))
        dual_c, dual_cons = ro.make_dual(ro.PRIMAL_C, ro.PRIMAL_A, ro.PRIMAL_B)
        _, gval, _ = ro.solve_lp_2d(dual_c, dual_cons, maximize=False)
        self.assertAlmostEqual(val, gval)

    def test_lp_against_scipy(self):
        try:
            from scipy.optimize import linprog
        except ImportError:
            self.skipTest("không có scipy để đối chiếu")
        r = linprog([-50, -40], A_ub=ro.PRIMAL_A, b_ub=ro.PRIMAL_B, bounds=[(0, None)] * 2)
        self.assertAlmostEqual(-r.fun, 2100.0, places=4)

    def test_gradient_descent(self):
        w, _ = ro.gradient_descent(0.0, 0.1)
        self.assertAlmostEqual(w, 3.0, places=5)
        # lr=1.1 -> sai số nhân (1 - 2*lr) = -1.2 mỗi bước: |w - 3| phải tăng dần (phân kỳ)
        _, h = ro.gradient_descent(0.0, 1.1, max_iter=30)
        errs = [abs(step[1] - 3.0) for step in h]
        self.assertTrue(all(b > a for a, b in zip(errs, errs[1:])))
        self.assertAlmostEqual(errs[-1] / errs[-2], 1.2, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
