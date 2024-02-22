from misc.graph import Graph, Vertex


class Floyd:
    """Implements Floyd's algorithm for calculating all shortest paths in a graph."""

    def __init__(self, graph: Graph, verbose=False) -> None:
        """Initialize the algorithm with a Graph.

        Args:
            graph (Graph): A Graph object.
            verbose (bool, optional): Whether or not to print step-by-step solutions. Defaults to False.
        """

        self.nodes = graph.get_nodes()
        self.cost = {}
        self.pred = {}
        self.verbose = verbose

        for i in self.nodes:
            self.cost[i] = {}
            self.pred[i] = {}
            for j in self.nodes:
                # A node has itself as predecessors with a cost of 0
                if i == j:
                    self.cost[i][i] = 0
                    self.pred[i][i] = i
                # Check neighbors
                elif j in graph.get_vertex(i).neighbors:
                    self.pred[i][j] = i
                    self.cost[i][j] = graph.get_vertex(i).neighbors[j]
                # Initially, no path exists
                else:
                    self.pred[i][j] = None
                    self.cost[i][j] = float("inf")

    def calc_shortest_paths(self) -> None:
        """
        Calculate all shortest paths through Dynamic Programming.
        """
        for detour in self.nodes:
            for start in self.nodes:
                if start == detour:
                    continue
                for end in self.nodes:
                    if end in (start, detour):
                        continue

                    if self.cost[start][end] > self.cost[start][detour] + self.cost[detour][end]:
                        self.cost[start][end] = self.cost[start][detour] + self.cost[detour][end]
                        self.pred[start][end] = self.pred[detour][end]

    def get_shortest_path(self, start, target) -> tuple[list[int], float]:
        """Once calc_shortest_paths has been called, use this method to query the shortest path from start to target.

        Args:
            start (key): Key associated with the start location in the graph.
            target (key): Key associated with the target location in the graph.

        Returns:
            _type_: [Path from start to target as a list, cost]
        """
        if self.cost[start][target] == float("inf"):
            print("There exists no path from %s to %s." % (start, target))
            return ([], -1)

        curr = target
        stack = [curr]
        while curr != start:
            stack.append(self.pred[start][curr])
            curr = self.pred[start][curr]

        if self.verbose:
            print(
                f"The shortest path from {start} to {target} is {stack[::-1]} with length of {self.cost[start][target]:.2f}"
            )

        return stack[::-1], self.cost[start][target]


if __name__ == "__main__":

    # Initialize a dictionary of vertices
    vertices = {}
    for char in range(ord("L") - ord("A") + 1):
        name = chr(ord("A") + char)
        vertices[name] = Vertex(name)

    # Add neighbors (neighbor, distance) to each vertex
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

    # Construct a graph from the vertices
    g = Graph(vertices.values())

    # Apply Floyd's Algorithm
    f = Floyd(g, verbose=True)
    f.calc_shortest_paths()

    # Get some (arbitrary) shortest paths
    f.get_shortest_path("A", "E")
    f.get_shortest_path("E", "A")
    f.get_shortest_path("B", "H")
    f.get_shortest_path("A", "L")
