import unittest
from dijkstra import Dijkstra
from misc.graph import Graph, Vertex


class Test_Dijkstra(unittest.TestCase):

    def test_lecture(self):
        # Slide 2.18
        vertices = {}
        for i in range(1, 6):
            vertices[i] = Vertex(i)

        vertices[1].add_neighbors([[2, 11], [3, 1], [4, 6], [5, 2]])
        vertices[2].add_neighbors([[1, 11], [3, 9], [4, 4], [5, 7]])
        vertices[3].add_neighbors([[1, 1], [2, 9], [4, 4]])
        vertices[4].add_neighbors([[1, 6], [2, 4], [3, 4], [5, 2]])
        vertices[5].add_neighbors([[1, 2], [2, 7], [4, 2]])

        g = Graph(vertices.values())
        d = Dijkstra(g)
        d.shortest_path(1, 2)
        assert d.get_path() == [1, 5, 4, 2] and d.get_cost() == 8

        # Slide 2.24
        vertices = {}
        for i in range(1, 7):
            vertices[i] = Vertex(i)

        vertices[1].add_neighbors([[2, 7], [3, 9], [5, 14]])
        vertices[2].add_neighbors([[1, 7], [3, 10], [4, 15]])
        vertices[3].add_neighbors([[1, 9], [2, 10], [4, 11], [5, 2]])
        vertices[4].add_neighbors([[2, 15], [3, 11], [6, 6]])
        vertices[5].add_neighbors([[1, 14], [3, 2], [6, 9]])
        vertices[6].add_neighbors([[4, 6], [5, 9]])

        g = Graph(vertices.values())
        d = Dijkstra(g)
        d.shortest_path(1, 6)
        assert d.get_path() == [1, 3, 5, 6] and d.get_cost() == 20

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

        d = Dijkstra(g)
        d.shortest_path("A", "E")
        assert d.get_path() == ["A", "C", "B", "E"] and d.get_cost() == 8
        d.shortest_path("E", "A")
        assert d.get_path() == ["A", "C", "B", "E"][::-1] and d.get_cost() == 8

        d.shortest_path("B", "H")
        assert d.get_path() == ["B", "E", "I", "G", "H"] and d.get_cost() == 11
        d.shortest_path("H", "B")
        assert d.get_path() == ["B", "E", "I", "G", "H"][::-1] and d.get_cost() == 11


if __name__ == "__main__":
    unittest.main()
