import unittest
from transportation_problem import TransportationProblem


class TestTransportationProblem(unittest.TestCase):

    def test_unbalanced_supply(self):
        demand, supply, cost = [1], [1, 1], [[1], [1]]
        tp = TransportationProblem(supply, demand, cost, verbose=False)
        tp.north_west_corner()
        tp.iterative_improvement()
        assert tp.get_objective() == 1
        assert tp.iterations == 1

    def test_unbalanced_demand(self):
        demand, supply, cost = [1, 1], [1], [[1, 1]]
        tp = TransportationProblem(supply, demand, cost, verbose=False)
        tp.north_west_corner()
        tp.iterative_improvement()
        assert tp.get_objective() == 1
        assert tp.iterations == 1

    def test_simple(self):
        demand = [2, 5, 3]
        supply = [5, 5]
        cost = [[5, 4, 5], [2, 4, 5]]
        tp = TransportationProblem(supply, demand, cost, verbose=False)
        tp.north_west_corner()
        assert tp.get_objective() == 45
        tp.iterative_improvement()
        assert tp.get_objective() == 39

        demand = [70, 60, 70, 130]
        supply = [160, 80, 90]
        cost = [[70, 200, 160, 240], [170, 210, 60, 125], [220, 85, 155, 140]]

        tp = TransportationProblem(supply, demand, cost, verbose=False)
        tp.north_west_corner()
        assert tp.get_objective() == 41700
        tp.iterative_improvement()
        assert tp.iterations == 3
        assert tp.get_objective() == 40200

        supply = [10, 8, 7]
        demand = [6, 5, 8, 6]
        cost = [[7, 7, 4, 7], [9, 5, 3, 3], [7, 2, 6, 4]]

        tp = TransportationProblem(supply, demand, cost, verbose=False)
        tp.north_west_corner()
        assert tp.get_objective() == 126
        tp.iterative_improvement()
        assert tp.iterations == 4
        assert tp.get_objective() == 100

    def test_tutorial(self):
        demand = [110, 40, 30, 50]
        supply = [90, 60, 80]
        cost = [[28, 26, 23, 31], [14, 18, 16, 19], [24, 29, 22, 25]]

        tp = TransportationProblem(supply, demand, cost, verbose=False)
        tp.north_west_corner()
        assert tp.get_objective() == 5430
        tp.iterative_improvement()
        assert tp.iterations == 4
        assert tp.get_objective() == 5100


if __name__ == "__main__":
    unittest.main()
