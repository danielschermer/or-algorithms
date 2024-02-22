import unittest

from typing import TypeAlias


class Vertex:
    def __init__(self, id):
        self.id = id
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight):
        if neighbor not in self.neighbors:
            self.neighbors[neighbor] = weight

    def add_neighbors(self, neighbors):
        for n in neighbors:
            self.add_neighbor(n[0], n[1])


class Graph:
    Vertices = list[Vertex]

    def __init__(self, vertices=None):
        self.vertices = {}
        if vertices:
            for v in vertices:
                self.vertices[v.id] = v

    def get_nodes(self):
        return list(self.vertices.keys())

    def get_vertex(self, id) -> Vertex:
        return self.vertices.get(id, None)

    def init_from_distance_matrix(self, A):
        rows, cols = len(A), len(A[0])
        symmetric = A == [list(i) for i in zip(*A)]

        def index_to_name(index):
            offset = ord("A")
            return chr(offset + index)

        if not rows == cols or not symmetric:
            raise Exception

        for i in range(rows):
            v = Vertex(index_to_name(i))
            for j in range(cols):
                if i == j:
                    if not A[i][i] == 0:
                        return False
                    continue
                if not A[i][j] == A[j][i]:
                    return False

                if A[i][j] > 0:
                    v.add_neighbor(index_to_name(j), A[i][j])
            self.vertices[index_to_name(i)] = v

    def add_vertex(self, vertex):
        if vertex.id not in self.vertices:
            self.vertices[vertex.id] = vertex
            return True
        else:
            return False

    def add_edge(self, a: Vertex, b: Vertex, weight: float):
        a.add_neighbor(b, weight)
        b.add_neighbor(a, weight)


if __name__ == "__main__":
    distances = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 4], [3, 3, 4, 0]]

    assert distances == [list(i) for i in zip(*distances)]

    g = Graph()
    g.init_from_distance_matrix(distances)

    vertices = []

    pass
