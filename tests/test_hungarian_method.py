import unittest

from hungarian_method import HungarianMethod


class Test_Hungarian_Method(unittest.TestCase):

    def test_lecture(self):
        M = float("inf")
        cost = [
            [M, 12, 4, 10, 10, 19],
            [12, M, 14, 8, 22, 14],
            [4, 14, M, 6, 8, 15],
            [10, 8, 6, M, 14, 9],
            [10, 22, 8, 14, M, 20],
            [19, 14, 15, 9, 20, M],
        ]
        hm = HungarianMethod(cost)
        assignment, objective = hm.optimize()
        assert (
            assignment == set(((2, 4), (4, 0), (5, 1), (0, 2), (1, 3), (3, 5))) and objective == 53
        )

    def test_exercise(self):
        cost = [[4, 3, 3, 1], [6, 7, 5, 2], [4, 8, 4, 2], [5, 4, 2, 3]]
        hm = HungarianMethod(cost)
        assignment, objective = hm.optimize()
        assert assignment == set(((1, 3), (2, 0), (3, 2), (0, 1))) and objective == 11

        cost = [[4, 3, 3, 1], [6, 4, 5, 1], [4, 8, 4, 2], [5, 2, 2, 3]]
        hm = HungarianMethod(cost)
        assignment, objective = hm.optimize()
        assert assignment == set(((2, 0), (0, 1), (1, 3), (3, 2))) and objective == 10

        M = float("inf")
        cost = [
            [M, 11, 20, 9, 16],
            [11, M, 10, 19, 17],
            [20, 10, M, 13, 21],
            [9, 19, 13, M, 9],
            [16, 17, 21, 9, M],
        ]
        hm = HungarianMethod(cost)
        assignment, objective = hm.optimize()
        assert assignment == set(((1, 2), (4, 0), (2, 1), (3, 4), (0, 3))) and objective == 54
