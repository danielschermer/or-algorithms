import numpy as np
from tabulate import tabulate


class TransportationProblem:
    """Class for solving Transportation Problems."""

    EPS = 1e-6

    def __init__(self, supply, demand, cost, verbose=False):
        """Initializes a transportation problem.

        Args:
            supply List: List of length m that stores the available supply.
            demand List: List of length n that stores the given demand.
            cost : mxn matrix that stores the cost[i][j] of serving j from i.
            verbose (bool, optional): Whether or not to print step-by-step solutions. Defaults to False.
        """
        self.supply = supply
        self.demand = demand
        self.m, self.n = len(supply), len(demand)
        self.cost = np.array(cost)
        self.objective = 0
        self.basic = []
        self.iterations = 0
        self.verbose = verbose

        assert self.m == len(cost) and self.n == len(cost[0])

        # Make sure that the problem is balanced
        if sum(self.supply) > sum(self.demand):
            # Add a dummy demand
            if self.verbose:
                print("Problem is unbalanced: adding dummy demand.")
            self.demand.append(sum(self.supply) - sum(self.demand))
            self.n += 1
            self.cost = np.hstack((self.cost, np.zeros((self.m, 1))))
        elif sum(self.supply) < sum(self.demand):
            # Add a dummy supply
            if self.verbose:
                print("Problem is unbalanced: adding dummy supply.")
            self.supply.append(sum(self.demand) - sum(self.supply))
            self.m += 1
            self.cost = np.vstack((self.cost, np.zeros((1, self.n))))

        # Initialize primal and dual variables
        self.x = np.zeros((self.m, self.n))
        self.u = [None] * self.m
        self.v = [None] * self.n
        self.reduced_cost = None

    def get_objective(self):
        """
        Return the objective value based on the primal solution.
        """

        self.objective = np.multiply(self.x, self.cost).sum()
        return self.objective

    def fix_solution(self, x):
        """
        # Debug: Fix a primal solution.
        """
        self.x = x
        return

    def north_west_corner(self):
        """
        Generate an initial solution based on the North-West-Corner method.
        """
        residual_supply = self.supply.copy()
        residual_demand = self.demand.copy()

        i, j = 0, 0
        while i < self.m and j < self.n:
            amount = min(residual_supply[i], residual_demand[j])
            self.x[i][j] = amount
            residual_supply[i] -= amount
            residual_demand[j] -= amount
            if residual_supply[i] == 0:
                i += 1
            else:
                j += 1

        # Make the solution basic, if necessary
        self.make_basic()

        if self.verbose:
            print(
                "Initial solution with cost %.2f by means of the North-West-Corner method:"
                % self.get_objective()
            )
            self.print_solution()

    def make_basic(self):
        """
        Verify that we have the required m+n-1 basic variables.
        If not fulfilled, include arbitrary basic variables (based on minimal cost).
        """

        # Basic variables based on transportation quantities.
        self.basic = []
        for i in range(self.m):
            for j in range(self.n):
                if self.x[i][j] > 0:
                    self.basic.append((i, j))

        delta = self.n + self.m - 1 - len(self.basic)

        # If we have a basic solution
        if delta == 0:
            return

        while delta > 0:
            # Find the current non-basic field with minimal cost by brute-force
            curr = (None, None)
            best = float("inf")
            for i in range(self.m):
                for j in range(self.n):
                    if (i, j) not in self.basic:
                        if self.cost[i][j] < best:
                            best = self.cost[i][j]
                            curr = (i, j)

            # Make it basic
            self.basic.append(curr)
            delta -= 1

    def update_dual_variables(self):
        """
        Calculate the dual variables corresponding to the current basic solution.
        """

        # Reset the dual variables
        self.u = [0] * self.m
        self.v = [0] * self.n

        # Arbitrarily fix a single dual variable to guarantee a unique solution.
        # This holds for solving the linear system that follows!
        self.u[0] = 0

        # Build the linear system
        A = []
        b = []
        for i, j in self.basic:
            eq = [0] * (self.n + self.m)
            eq[i] = 1
            eq[self.m + j] = 1
            # u_0 is fixed, so only eq[1:]!
            A.append(eq[1:])
            b.append(self.cost[i][j])

        # Solve the linear system and update the dual variables.
        solution = np.linalg.solve(A, b)
        self.u = [0] + solution[0 : self.m - 1].tolist()
        self.v = solution[self.m - 1 :].tolist()

    def update_reduced_costs(self):
        """
        Calculate the reduced costs, given the current values of dual variables.
        """
        self.reduced_cost = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                if self.x[i][j] == 0:
                    self.reduced_cost[i][j] = self.cost[i][j] - self.u[i] - self.v[j]

    def modify_distribution(self):
        """
        This method identifies what quantity to exchange along which path.
        1.  The most-negative reduced cost defines the entering basic variable.
        2.  The next step is to perform a Depth-First-Search to find a cycle along
            which we can exchange transportation quantities.
        3.  The delivered supply on this cycle determines the maximum quantity that can
            be shifted (no delivery can be negative).
        """
        # The most-negative reduced costs define the entering variable
        s, t = np.unravel_index(self.reduced_cost.argmin(), self.reduced_cost.shape)
        entering = (s, t)

        def dfs():
            """
            Depth-First Search to find a cycle in the graph implied by basic variables.
            """
            stack = [[entering, [entering], True]]
            while stack:
                start, path, searching_for_supplier = stack.pop()
                s, d = start[0], start[1]

                # We can break the loop as soon as we have found a cycle
                if len(path) > 2 and path[0][0] == path[-1][0]:
                    break

                # Find all supply locations that match the current demand location
                if searching_for_supplier:
                    for i in range(self.m):
                        if i == start[0]:
                            continue
                        if (i, d) in self.basic:
                            stack.append([(i, d), path + [(i, d)], not searching_for_supplier])

                # Find all demand locations that match the current supply location
                else:
                    for j in range(self.n):
                        if j == start[1]:
                            continue
                        if (s, j) in self.basic:
                            stack.append([(s, j), path + [(s, j)], not searching_for_supplier])

            # Sanity check: There must always be an even number of supply and demand locations
            assert len(path) % 2 == 0

            return path

        cycle = dfs()

        def delta_quantity():
            """
            Determine what quantity can be shifted in the current cycle and which variable must leave the basis.
            """
            quantity = float("inf")
            leaving = (None, None)

            for index in range(1, len(cycle), 2):
                (i, j) = cycle[index]
                if self.x[i][j] < quantity:
                    quantity = self.x[i][j]
                    leaving = (i, j)

            if self.verbose:
                print(
                    "Entering non-basic variable is field %s and leaving basic-variable is field %s."
                    % (entering, leaving)
                )
                print("Exchange a delta-quantity of %.2f along path %s." % (quantity, cycle))
                d = {}
                for index, (i, j) in enumerate(cycle):
                    if index % 2 == 0:
                        d[(i, j)] = quantity
                    else:
                        d[(i, j)] = -quantity
                self.print_solution(delta=d)

            return quantity, leaving

        quantity, leaving = delta_quantity()

        # Adjust the supply in the cycle
        for index, (i, j) in enumerate(cycle):
            if index % 2 == 0:
                self.x[i][j] += quantity
            else:
                self.x[i][j] -= quantity

        # Update the basic variables
        self.basic.remove(leaving)
        self.basic.append(entering)

    def check_primal_degeneracy(self):
        """
        A solution is primal degenerate if a basic variable has value 0.
        """
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in self.basic:
                    continue
                if self.x[i][j] == 0:
                    print(
                        "Solution is primal degenerate: basic variable %s has transportation quantity of 0."
                        % ("(" + (str(i) + ", " + str(j) + ")"))
                    )

    def check_dual_degeneracy(self):
        """
        A solution is dual degenerate if a there exist reduced-costs with value 0.
        """
        if self.reduced_cost is None:
            return

        for i in range(self.m):
            for j in range(self.n):
                if (i, j) in self.basic:
                    continue
                if self.reduced_cost[i][j] == 0:
                    print(
                        "Solution is dual degenerate: non-basic variable %s has reduced costs of 0."
                        % ("(" + (str(i) + ", " + str(j) + ")"))
                    )
        return

    def iterative_improvement(self):
        """
        Main optimization loop of the modified distribution method.
        Given a feasible solution, iterate the following until no negative reduced costs remain:
        1.  Update the dual variables.
        2.  Determine the reduced costs.
        3.  Modify the distribution
        """
        if self.verbose:
            self.check_primal_degeneracy()
        # Given a feasible basic solution, update the dual variables
        self.update_dual_variables()
        # Calculate the reduced costs
        self.update_reduced_costs()

        self.iterations += 1
        # Continue exchanging basic variables based on reduced cost
        while np.min(self.reduced_cost) < 0:
            self.modify_distribution()
            if self.verbose:
                self.check_primal_degeneracy()
            self.update_dual_variables()
            self.update_reduced_costs()

            self.iterations += 1

        if self.verbose:
            print(
                f"No negative reduced costs remain. The following solution is optimal with cost {self.get_objective():.2f}."
            )
            self.print_solution()
            self.check_primal_degeneracy()
            self.check_dual_degeneracy()

    def print_solution(self, delta={}):
        """
        Prints the current solution as a transportation tableau.
        """

        table = []

        # Build the header
        head = [""]
        for j in range(self.n):
            head.append("%s" % j)
            head.append(" ")
        head.append("a_i")
        head.append("u_i")
        table.append(head)

        # Build the rows
        for i in range(self.m):
            for k in range(2):
                if k == 0:
                    row = ["%i" % i]
                else:
                    row = [""]
                for j in range(self.n + 1):
                    if j < self.n:
                        if k == 0:
                            row.append(self.cost[i][j])
                            row.append(self.x[i][j])
                        if k == 1:
                            if self.reduced_cost is None:
                                row.append("")
                            elif self.reduced_cost[i][j] == 0:
                                row.append("")
                            else:
                                row.append(self.reduced_cost[i][j])

                            if delta.get((i, j)) == None:
                                row.append("")
                            else:
                                row.append(delta.get((i, j)))
                    else:
                        if k == 0:
                            row.append(str(self.supply[i]))
                            row.append(str(self.u[i]))
                table.append(row)

        # Build the footer
        footer = ["b_j"]
        for j in range(self.n):
            footer.append(self.demand[j])
            footer.append("")
        footer.append("c_ij")
        footer.append("x_ij")
        table.append(footer)

        footer = ["v_j"]
        for j in range(self.n):
            footer.append(str(self.v[j]))
            footer.append("")
        footer.append("c_ij^*")
        footer.append("d_x_ij")
        table.append(footer)

        print(tabulate(table, tablefmt="rounded_grid"))


if __name__ == "__main__":

    demand = [2, 5, 3]
    supply = [5, 5]
    cost = [[5, 4, 5], [2, 4, 5]]
    tp = TransportationProblem(supply, demand, cost, verbose=True)
    tp.north_west_corner()
    assert tp.get_objective() == 45
    tp.iterative_improvement()
    assert tp.get_objective() == 39
