# Operations Research Algorithms in Python
This repository contains Python implementations of Operations Research Algorithms taught in the undergraduate course *Management Science* (*Operations Research*) at the [University of Kaiserslautern-Landau](https://en.wikipedia.org/wiki/University_of_Kaiserslautern-Landau).
Included algorithms are:
* [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) for calculating the shortest path in graphs.
* [Floyd's algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm) for calculating all shortest paths in graphs.
* The [Ford-Fulkerson algorithm](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm) for computing the maximum flow (minimum cut) in graphs.
* A simplistic implementation of the [revised Simplex method](https://en.wikipedia.org/wiki/Revised_simplex_method) for:
  * solving [Linear Programming Problems](https://en.wikipedia.org/wiki/Linear_programming) ($\max c^\intercal x$ subject to $Ax \leq b, x\geq 0$),
  * reformulating and solving such problems as [Dual Programs](https://en.wikipedia.org/wiki/Dual_linear_program) ($\min b^\intercal w$ subject to $A^\intercal w \geq c, w \geq 0$),
  * solving Integer Linear Programs using [Gomory's Cutting-Plane](https://en.wikipedia.org/wiki/Cutting-plane_method) method,
  * TODO: solving Integer Linear Programs by means of the [Branch and Bound](https://en.wikipedia.org/wiki/Branch_and_bound) method.
* The [modified distribution (MODI) method](https://de.wikipedia.org/wiki/Transportproblem) for solving linear transportation problems.
* The primal-dual [Hungarian method](https://en.wikipedia.org/wiki/Hungarian_algorithm) for solving linear assignment problems.


The chosen programming language implies that this repository is not aiming to be a high-performance implementation.
Its purpose is solely to generate reference solutions for small examples and display the ***step-by-step*** solution process.
As such, it caters to curious students, seeking comprehensible solutions to standard textbook problems.


# Getting started
To get started, clone the repository, initialize a virtual Python3 environment, and install the required packages (`pip install -r requirements.txt`).
Afterwards, you can play around with the existing examples or create your own by following the `main` functions.

The implementation relies on [`numpy`](https://numpy.org/) for core data structures as well as algorithms and [`tabulate`](https://pypi.org/project/tabulate/) for pretty-printing.

# Disclaimer
*If you try to break the algorithms*, it is **guaranteed that they will break**!
This concerns, e.g., applying Dijkstra's algorithm in a Graph with negative weights or formulating a textbook adversarial linear programming problem that would require a lexicographic pivoting rule to escape primal-degenerate cycling.
