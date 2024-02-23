# A collection of Operations Research algorithms

This repository contains a straightforward implementation of several fundamental Operations Research Algorithms in Python.
The implementations are intended to be consistent with the algorithms taught in the undergraduate course *Management Science* (*Operations Research*) at the [University of Kaiserslautern-Landau](https://en.wikipedia.org/wiki/University_of_Kaiserslautern-Landau).
The algorithms include:
* [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) for calculating a shortest path in graphs.
* [Floyd's algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm) for calculating all shortest paths in graphs.
* The [Ford-Fulkerson algorithm](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm) for computing maximum flow (minimum cuts) in graphs.
* A simplistic implementation of the [revised Simplex method](https://en.wikipedia.org/wiki/Revised_simplex_method) that can be used for:
  * solving [Linear Programming Problems](https://en.wikipedia.org/wiki/Linear_programming) of the form $\max c^\intercal x$ subject to $Ax \leq b, x\geq 0$,
  * reformulating and solving such problems as [Dual Programs](https://en.wikipedia.org/wiki/Dual_linear_program) $\min b^\intercal w$ subject to $A^\intercal w \geq c, w \geq 0$,
  * solving Integer Linear Programs by means [Gomory's Cutting-Plane](https://en.wikipedia.org/wiki/Cutting-plane_method) method,
  * TODO: solving Integer Linear Programs by means of the [Branch and Bound](https://en.wikipedia.org/wiki/Branch_and_bound) method.
* The [modified distribution (MODI) method](https://de.wikipedia.org/wiki/Transportproblem) for solving linear transportation problems.
* The primal-dual [Hungarian method](https://en.wikipedia.org/wiki/Hungarian_algorithm) for solving linear assignment problems.

The choice of the programming language should immediately suggest to anyone that these implementations are by no means meant to be high-performance reference implementations that attempt to compete with the state-of-the-art in any way.
Instead, this repository only exists for generating reference solutions to small-sized examples and printing the *step-by-step solution process* to the standard output.
As such, this repository should satisfy the needs of curious students who are looking for comprehensible solutions to standard textbook problems.

# Getting started

To get started, clone the repository, initialize a virtual Python3 environment, and install the required packages (`pip install -r requirements.txt`).
Afterwards, you can play around with the existing examples or create your own by following the `main` functions.

The implementation relies on [`numpy`](https://numpy.org/) for core data structures as well as algorithms and [`tabulate`](https://pypi.org/project/tabulate/) for pretty-printing.

# Disclaimer
*If you try to break the algorithms*, it is **guaranteed that they will break**!
This concerns, e.g., applying Dijkstra's algorithm in a Graph with negative weights or formulating a textbook adversarial linear programming problem that would require a lexicographic pivoting rule to escape primal-degenerate cycling.
