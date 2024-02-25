import unittest
from ford_fulkerson import FordFulkerson
from misc.graph import Graph, Vertex


class TestFordFulkerson(unittest.TestCase):

    def test_exercise(self):

        vertices = {}
        for i in range(6):
            name = i
            vertices[name] = Vertex(name)

        vertices[0].add_neighbors([(1, 5), (2, 2), (3, 3)])
        vertices[1].add_neighbors([(0, 2), (4, 6)])
        vertices[2].add_neighbors([(0, 5), (1, 3)])
        vertices[3].add_neighbors([(0, 3), (1, 4), (4, 3), (5, 5)])
        vertices[4].add_neighbors([(2, 6), (3, 3), (5, 3)])
        vertices[5].add_neighbors([(3, 5), (4, 3)])

        g = Graph(vertices.values())
        ff = FordFulkerson()
        ff.calc_maximum_flow(g, 0, 5)

        assert (
            ff.get_max_flow() == 8
            and len(ff.get_cut_set().symmetric_difference(set(((3, 5), (4, 5))))) == 0
        )

    def test_simple(self):

        vertices = {}
        for i in range(ord("L") - ord("A") + 1):
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

        ff = FordFulkerson()
        ff.calc_maximum_flow(g, "K", "E")
        assert (
            ff.get_max_flow() == 11
            and len(
                ff.get_cut_set().symmetric_difference(
                    set((("J", "E"), ("G", "I"), ("D", "E"), ("B", "E")))
                )
            )
            == 0
        )

        ff.calc_maximum_flow(g, "F", "J")
        assert (
            ff.get_max_flow() == 14
            and len(
                ff.get_cut_set().symmetric_difference(set((("F", "K"), ("F", "J"), ("F", "D"))))
            )
            == 0
        )

        ff.calc_maximum_flow(g, "H", "D")
        assert (
            ff.get_max_flow() == 13
            and len(
                ff.get_cut_set().symmetric_difference(
                    set((("G", "I"), ("J", "E"), ("K", "A"), ("F", "D")))
                )
            )
            == 0
        )


if __name__ == "__main__":
    unittest.main()
