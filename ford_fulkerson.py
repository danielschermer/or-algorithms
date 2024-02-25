from misc.graph import Graph, Vertex


class FordFulkerson:
    """Implements the Ford-Fulkerson algorithm for finding the maximum flow in a weighted graph."""

    def __init__(self, graph: Graph, verbose=False):
        """Initialize the algorithm with a Graph

        Args:
            graph (Graph): A Graph object.
            verbose (bool, optional): Whether or not to print step-by-step solutions. Defaults to False.
        """
        # Nodes and neighbors copied from the Graph g
        self.nodes = set()
        self.neighbors = {}
        self.initialize_residuals(graph)

        # Predecessors and residual capacity; continuously updated by calls to self.bfs()
        self.pred = {}
        self.residual = {}

        self.verbose = verbose

        # Stores the result
        self.cut_set = set()
        self.max_flow = 0

    def initialize_residuals(self, graph):
        """
        Initializes the residual capacity to the weight of each edge.
        """
        self.nodes = set(graph.get_nodes())

        for v in graph.get_nodes():
            self.residual[v] = {}
            self.neighbors[v] = graph.get_vertex(v).neighbors
            for n in self.neighbors[v]:
                self.residual[v][n] = self.neighbors[v][n]

    def bfs(self, s, t) -> bool:
        """Performs a Breadth-First Search (BFS) to find any augmenting path from node s to node t."""

        # Mark all nodes as unvisited
        unvisited = list(self.nodes)
        queue = [s]
        unvisited.remove(s)

        # Perform BFS
        while queue:
            curr = queue.pop(0)
            neighbors = self.neighbors[curr]

            # For each neighbor
            for n in neighbors:
                # Only consider unvisited nodes with a positive residual capacity
                if n in unvisited and self.residual[curr][n] > 0:
                    queue.append(n)
                    unvisited.remove(n)
                    self.pred[n] = curr

        # If t is not in unvisited, then a residual (s, t) path exists
        return t not in unvisited

    def calc_maximum_flow(self, s, t):
        """Determines the maximum flow (if existent) by means of the Ford-Fulkerson algorithm.

        Args:
            start (key): Key associated with the start location in the graph.
            target (key): Key associated with the target location in the graph.
        """

        # As long as there exists an augmenting (s,t)-path (found by means of BFS).
        while self.bfs(s, t):

            # Capacity of the augmenting path
            path_flow = float("inf")

            # Traverse the path in reverse to determine the path_flow
            curr = t
            path = [curr]
            while curr != s:
                # Find the bottleneck
                prev = self.pred[curr]
                path_flow = min(path_flow, self.residual[prev][curr])

                curr = self.pred[curr]
                path.append(curr)

            if self.verbose:
                print(f"Found path {path[::-1]} with capacity {path_flow} from {s} to {t}.")

            # Update the maximum flow based on the current path
            self.max_flow += path_flow

            # Traverse the path in reverse and adjust its capacity based on path_flow
            curr = t
            prev = self.pred.get(curr, None)
            while prev is not None:
                self.residual[prev][curr] = self.residual[prev].get(curr, 0) - path_flow
                self.residual[curr][prev] = self.residual[curr].get(prev, 0) + path_flow

                prev = self.pred.get(prev, None)
                curr = self.pred[curr]

        self.calc_cut_set(s, t)
        if self.verbose:
            print(
                f"No further flow is possible from {s} to {t}. The maximum flow is {self.max_flow}."
            )
            print(
                f"The corresponding min-cut has a weight of {self.max_flow} and corresponds to the cut-set: {self.cut_set}"
            )

    def calc_cut_set(self, s, t):
        """
        Determines the cut-set corresponding to the minimum-cut (maximum flow):
        1. Performs a Breadth-First Search to identify reachable nodes in the residual network.
        2. An edge (i,j) belongs to the cut-set if i is reachable from s but t is not and (i,j) has a residual capacity of 0.
        """
        cut_set = set()
        unvisited = list(self.nodes)
        visited = set()
        queue = [s]
        unvisited.remove(s)

        # Perform a Breadth-First Search
        while queue:
            curr = queue.pop(0)
            visited.add(curr)
            neighbors = self.neighbors[curr]

            # For each neighbor
            for n in neighbors:
                if n in unvisited and self.residual[curr][n] > 0:
                    queue.append(n)
                    unvisited.remove(n)
                    self.pred[n] = curr

        # Identify edges that belong to the cut-set.
        for i in self.nodes:
            for j in self.neighbors[i]:
                # If i is reachable from s but j is not, then (i, j) is a candidate
                if i in visited and j in unvisited:
                    # (i,j) must have no residual capacity but must be an edge in the original graph
                    if self.residual[i][j] == 0 and self.neighbors[i][j] != 0:
                        cut_set.add((i, j))

        self.cut_set = cut_set

        return cut_set

    def get_max_flow(self):
        return self.max_flow

    def get_cut_set(self):
        return self.cut_set


if __name__ == "__main__":

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

    ff = FordFulkerson(g, verbose=True)
    ff.calc_maximum_flow("A", "E")

    ff = FordFulkerson(g, verbose=True)
    ff.calc_maximum_flow("K", "E")

    ff = FordFulkerson(g, verbose=True)
    ff.calc_maximum_flow("F", "J")

    ff = FordFulkerson(g, verbose=True)
    ff.calc_maximum_flow("H", "D")
