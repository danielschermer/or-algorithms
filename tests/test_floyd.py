import unittest
from floyd import Floyd

from misc.graph import Graph, Vertex


class Test_Floyd(unittest.TestCase):

    def test_lecture(self):
        # Slide 2.37
        vertices = {}
        for i in range(1, 6):
            vertices[i] = Vertex(i)

        vertices[1].add_neighbors([[2, 11], [3, 1], [4, 6], [5, 2]])
        vertices[2].add_neighbors([[1, 11], [3, 9], [4, 4], [5, 7]])
        vertices[3].add_neighbors([[1, 1], [2, 9], [4, 4]])
        vertices[4].add_neighbors([[1, 6], [2, 4], [3, 4], [5, 2]])
        vertices[5].add_neighbors([[1, 2], [2, 7], [4, 2]])

        g = Graph(vertices.values())

        f = Floyd(g)
        f.calc_shortest_paths()
        path, cost = f.get_shortest_path(1, 2)
        assert path == [1, 5, 4, 2] and cost == 8

        path, cost = f.get_shortest_path(2, 5)
        assert path == [2, 4, 5] and cost == 6

    def test_exercise(self):
        vertices = {}
        for i in range(ord("K") - ord("A") + 1):
            name = chr(ord("A") + i)
            vertices[name] = Vertex(name)

        vertices["A"].add_neighbors([("C", 1), ("D", 8), ("K", 2)])
        vertices["B"].add_neighbors([("C", 4), ("E", 3)])
        vertices["C"].add_neighbors([("A", 1), ("B", 4), ("D", 6)])
        vertices["D"].add_neighbors([("A", 8), ("C", 6), ("E", 3), ("F", 6)])
        vertices["E"].add_neighbors([("B", 3), ("D", 3), ("I", 5), ("J", 4)])
        vertices["F"].add_neighbors([("D", 6), ("J", 5), ("K", 3)])
        vertices["G"].add_neighbors([("H", 2), ("I", 1), ("J", 7)])
        vertices["H"].add_neighbors([("G", 2), ("J", 8), ("K", 9)])
        vertices["I"].add_neighbors([("E", 5), ("G", 1)])
        vertices["J"].add_neighbors([("E", 4), ("F", 5), ("G", 7), ("H", 8)])
        vertices["K"].add_neighbors([("A", 2), ("F", 3), ("H", 9)])

        g = Graph(vertices.values())
        f = Floyd(g)

        f.calc_shortest_paths()
        path, cost = f.get_shortest_path("A", "E")
        assert path == ["A", "C", "B", "E"] and cost == 8
        path, cost = f.get_shortest_path("E", "A")
        assert path == ["E", "B", "C", "A"] and cost == 8
        path, cost = f.get_shortest_path("B", "H")
        assert path == ["B", "E", "I", "G", "H"] and cost == 11
        path, cost = f.get_shortest_path("H", "B")
        assert path == ["H", "G", "I", "E", "B"] and cost == 11
        path, cost = f.get_shortest_path("A", "J")
        assert path == ["A", "K", "F", "J"] and cost == 10
        path, cost = f.get_shortest_path("K", "E")
        assert path == ["K", "A", "C", "B", "E"] and cost == 10
