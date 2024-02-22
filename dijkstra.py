from misc.graph import Graph, Vertex


class Dijkstra:
    """Implements Dijkstra's algorithm for finding a shortest path in a weighted graph."""

    def __init__(self, g: Graph, verbose=False):
        """Initialize the algorithm with a Graph

        Args:
            graph (Graph): A Graph object.
            verbose (bool, optional): Whether or not to print step-by-step solutions. Defaults to False.
        """
        self.g = g
        self.unvisited = g.get_nodes()
        self.cost = {}
        self.pred = {}
        self.iteration = 1
        self.verbose = verbose
        self.start = None
        self.target = None

    def get_cost(self):
        """
        Returns the cost associated with the last call to shortest_path()
        """
        return self.cost.get(self.target, None)

    def get_path(self):
        """
        Returns the path associated with the last call to shortest_path()
        """
        stack = []
        curr = self.target
        while curr is not None:
            stack.append(curr)
            curr = self.pred.get(curr, None)

        return stack[::-1]

    def initialize_labels(self, start, target):
        """
        Initializes the labels to 0 for the start node and infinity for all other nodes.
        """
        self.unvisited = self.g.get_nodes()
        self.cost = {}
        self.pred = {}
        self.iteration = 1
        self.start = start
        self.target = target

        for vertex in self.g.get_nodes():
            if vertex == self.start:
                self.cost[self.start] = 0
            else:
                self.cost[vertex] = float("inf")

    def get_next_base_node(self):
        """
        Returns the vertex v with the currently smallest label
        """
        curr = None
        for vertex in self.unvisited:
            if curr is None:
                curr = vertex
            elif self.cost[vertex] < self.cost[curr]:
                curr = vertex

        # Remove the current base node from the list of unvisited nodes
        self.unvisited.remove(curr)

        return curr

    def update_neighbors(self, curr):
        """
        Given node curr, update the labels of neighbors.
        """
        neighbors = self.g.get_vertex(curr).neighbors
        for n in neighbors:
            if self.cost[curr] + neighbors[n] < self.cost[n]:
                self.cost[n] = self.cost[curr] + neighbors[n]
                self.pred[n] = curr
                if self.verbose:
                    print(f"New shortest path to {n} via {curr} with length {self.cost[n]:.2f}")

    def shortest_path(self, start, target) -> bool:
        """Determines the shortest path (if existent) by means of Dijkstra's algorithm.

        Args:
            start (key): Key associated with the start location in the graph.
            target (key): Key associated with the target location in the graph.
        """

        # Initialize the labels
        self.initialize_labels(start, target)

        # As long as each node has not been base node
        while self.unvisited:

            base = self.get_next_base_node()
            if self.verbose:
                print(f"\nIteration {self.iteration}: {base} is the current base node.")
                self.iteration += 1

            # Terminate the algorithm when the target node is the base node
            if base == target:
                if self.verbose:
                    path = self.get_path()
                    print(
                        f"The shortest path from {self.start} to {self.target} is {path} with length of {self.cost[self.target]:.2f}"
                    )
                return True

            # Update the neighbors of the current base node
            self.update_neighbors(base)

        return False


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

    d = Dijkstra(g, verbose=True)
    d.shortest_path("A", "E")
    d.shortest_path("B", "H")
    d.shortest_path("A", "L")
