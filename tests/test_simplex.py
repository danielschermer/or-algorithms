import unittest
import numpy as np
from simplex.model import Model


def assignment_problem():
    c = np.array([-4, -3, -3, -1, -6, -7, -5, -2, -4, -8, -4, -2, -5, -4, -2, -3])  # Z = 11
    # c = np.array([-4, -3, -3, -1, -6, -4, -5, -1, -4, -8, -4, -2, -5, -2, -2, -3]) # Z = 10

    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
    A = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [-1, -1, -1, -1, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            [-0, -0, -0, -0, -1, -1, -1, -1, -0, -0, -0, -0, -0, -0, -0, -0],
            [-0, -0, -0, -0, -0, -0, -0, -0, -1, -1, -1, -1, -0, -0, -0, -0],
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -1, -1, -1, -1],
            [-1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0],
            [-0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0],
            [-0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0],
            [-0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1],
        ]
    )

    return A, b, c


class Test(unittest.TestCase):

    def test_transportation_problem(self):
        A = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [-1, -1, -1, -1, -0, -0, -0, -0, -0, -0, -0, -0],
                [-0, -0, -0, -0, -1, -1, -1, -1, -0, -0, -0, -0],
                [-0, -0, -0, -0, -0, -0, -0, -0, -1, -1, -1, -1],
                [-1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0],
                [-0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0, -0],
                [-0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1, -0],
                [-0, -0, -0, -1, -0, -0, -0, -1, -0, -0, -0, -1],
            ]
        )
        b = np.array([10, 8, 7, 6, 5, 8, 6, -10, -8, -7, -6, -5, -8, -6])  # , 7])
        c = np.array([-7, -7, -4, -7, -9, -5, -3, -3, -7, -2, -6, -4])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == -100 and m.cnt_primal == 0 and m.cnt_dual == 6

    def test_assignment(self):
        A, b, c = assignment_problem()
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == -11 and m.cnt_primal == 0 and m.cnt_dual == 5

    def test_simple(self):
        A = np.array([[1, 0, 0, 2, 1], [0, 1, 0, -3, 1], [0, 0, 1, 1, -3]])
        b = np.array([1, 2, 3])
        c = np.array([1, 1, 1, 1, 1])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == 8.0 and m.cnt_primal == 5 and m.cnt_dual == 0

    def test_dual(self):
        A = np.array([[-1, -0, -1, -1, -2], [-0, -1, -2, -1, -1]])
        b = np.array([-21, -12])
        c = np.array([-20, -20, -31, -11, -12])

        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == -141.0

        m.dualize()
        m.optimize()
        assert m.get_objective() == 141.0 and m.cnt_primal == 2 and m.cnt_dual == 0

    def test_primal_degenerate(self):
        A = np.array([[-1, 1], [1, -2], [1, 2]])
        b = np.array([3, 2, 6])
        c = np.array([2, 3])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == 11.0 and m.cnt_primal == 3 and m.cnt_dual == 0

    def test_dual_degenerate(self):
        A = np.array([[-1, 1], [1, -2], [1, 1]])
        b = np.array([3, 2, 7])
        c = np.array([1, 1])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == 7.0 and m.cnt_primal == 2 and m.cnt_dual == 0

    def test_phase_one(self):
        A = np.array([[-1, 1], [1, -2], [1, 1], [-1, -0.25], [-1 / 3, -1]])
        b = np.array([3, 2, 7, -1, -1])
        c = np.array([2, 3])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.get_objective() == 19.0 and m.cnt_primal == 1 and m.cnt_dual == 2

    def test_infeasible(self):
        A = np.array([[1], [-1]])
        b = np.array([5, -6])
        c = np.array([1])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert m.status == "INFEASIBLE"

    def test_unbounded(self):
        A = np.array([[-1, 1], [1, -2]])
        b = np.array([3, 2])
        c = np.array([2, 3])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert (
            m.get_objective() == 9.0
            and m.cnt_primal == 1
            and m.cnt_dual == 0
            and m.status == "UNBOUNDED"
        )

        m = Model()
        m.init_from_values(A, b, c)
        m.dualize()
        m.optimize()
        assert m.cnt_primal == 0 and m.cnt_dual == 2 and m.status == "INFEASIBLE"

    def test_knapsack(self):
        A = np.array(
            [
                [9, 13, 11, 15, 2],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        b = np.array([30, 1, 1, 1, 1, 1])
        c = np.array([22, 46, 34, 42, 5])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize()
        assert (
            abs(m.get_objective() - 96.80) < 0.001
            and m.cnt_primal == 4
            and m.cnt_dual == 0
            and m.status == "OPTIMAL"
        )

    def test_gomory(self):
        A = np.array([[2, 6], [6, 4]])  # , [3, 4]])
        b = np.array([15, 21])  # , 7])
        c = np.array([1, 2])
        m = Model()
        m.init_from_values(A, b, c)
        m.optimize_gomory()
        assert (
            abs(m.get_objective() - 5) < 0.001
            and m.cnt_primal == 2
            and m.cnt_dual == 8
            and m.status == "INTEGER-OPTIMAL"
        )


if __name__ == "__main__":
    unittest.main()
