import numpy as np
from tabulate import tabulate


class HungarianMethod:
    """Implements Hungarian method by means of a matrix interpretation."""

    def __init__(self, c, verbose=False):
        """_summary_

        Args:
            c (_type_): A square matrix where c[i][j] determines the cost of assigning i to j.
            verbose (bool, optional): Whether or not to print step-by-step solutions. Defaults to False.
        """
        self.cost = np.array(c)
        self.m = len(c)
        self.n = len(c[0])
        assert self.m == self.n

        # Dual variables
        self.u = np.zeros(self.m)
        self.v = np.zeros(self.n)

        # Independent variables correspond to the assignment
        self.independent = set()
        self.dependent = set()

        self.iterations = 0
        self.verbose = verbose

    def initialize(self):
        """
        Initialize by a dual-feasible solution.
        """
        self.subtract_column_minima()
        self.subtract_row_minima()

    def subtract_column_minima(self):
        """
        Subtract the column minima.
        """
        self.v = np.min(self.cost, axis=0)
        self.cost = self.cost - self.v

    def subtract_row_minima(self):
        """
        Subtract the row minima.
        """
        self.u = np.min(self.cost, axis=1)
        self.cost = self.cost - self.u.reshape(len(self.u), 1)

    def get_independent_zeros(self):
        """
        Given the cost matrix, this method determines a maximal number of independent zeroes.
        """

        self.independent = set()
        self.dependent = set()

        # Determine all indices with fields associated with a cost of 0
        args = np.argwhere(self.cost == 0)

        # For each index determine the dependent zeros
        index_to_dependent = {}
        for [row, col] in args:
            index_to_dependent[(row, col)] = set()

            # Check if we have a zero in the same row
            for [i, j] in args[np.where(args[:, 0] == row)]:
                if i != row or j != col:
                    index_to_dependent[(row, col)].add((i, j))

            # Check if we have a zero in the same column
            for [i, j] in args[np.where(args[:, 1] == col)]:
                if i != row or j != col:
                    index_to_dependent[(row, col)].add((i, j))

        while len(index_to_dependent) > 0:
            # Take any zero that is minimally dependent on other zeros
            min_dependency = min(len(l) for l in index_to_dependent.values())
            row, col = [k for k, v in index_to_dependent.items() if len(v) == min_dependency][0]

            # Add it to the list of independent zeros
            self.independent.add((row, col))
            index_to_dependent.pop((row, col))

            # Remove all zeros in the same row or column
            queue_for_removal = []

            # A zero is dependent if it is in the same row or same column
            for i, j in index_to_dependent:
                if i == row or j == col:
                    queue_for_removal.append((i, j))

            for i, j in queue_for_removal:
                self.dependent.add((i, j))
                index_to_dependent.pop((i, j))

                # The now dependent element can be removed from all remaining elements
                for s, t in index_to_dependent:
                    if (i, j) in index_to_dependent[(s, t)]:
                        index_to_dependent[(s, t)].remove((i, j))

    def get_objective(self):
        """
        Returns the current objective based on the dual variables.
        """
        return sum(self.u) + sum(self.v)

    def update_dual_variables(self):
        """
        Updates the dual variables u and v based on the independent zeroes.
        """

        # TODO: This function could be made more efficient with a different dictionary structure.

        marked_rows = set()
        marked_cols = set()

        # Mark each row with no independent zero:
        for row in range(self.m):
            independent_zero_in_row = False
            for i, j in self.independent:
                if i == row:
                    independent_zero_in_row = True
                    break
            if not independent_zero_in_row:
                marked_rows.add(row)

        # Repeat the next two steps as long as there are new columns or rows to be marked
        while True:
            size = len(marked_rows) + len(marked_cols)

            # Mark each column with a dependent zero in a marked row
            for col in range(self.n):
                if col in marked_cols:
                    continue

                dependent_zero_in_marked_row = False
                for i, j in self.dependent:
                    if j == col and i in marked_rows:
                        dependent_zero_in_marked_row = True
                        break
                if dependent_zero_in_marked_row:
                    marked_cols.add(col)

            # Mark each row with an independent zero in a marked column
            for row in range(self.m):
                if row in marked_rows:
                    continue

                independent_zero_in_marked_col = False
                for i, j in self.independent:
                    if i == row and j in marked_cols:
                        independent_zero_in_marked_col = True
                        break
                if independent_zero_in_marked_col:
                    marked_rows.add(row)

            # Break if no new marked row or column was added
            if size == len(marked_rows) + len(marked_cols):
                break

        # Eta is the minimum of elements located in marked rows and unmarked columns
        eta = self.cost[list(marked_rows)][
            :, list(set.difference(set(range(self.n)), marked_cols))
        ].min()

        # Update the costs based on eta
        for i in range(self.m):
            for j in range(self.n):
                if i in marked_rows:
                    if j not in marked_cols:
                        self.cost[i][j] -= eta
                else:
                    if j in marked_cols:
                        self.cost[i][j] += eta

        # Update the dual variables
        for i in range(self.m):
            if i in marked_rows:
                self.u[i] += eta

        for j in range(self.n):
            if j in marked_cols:
                self.v[j] -= eta

    def print(self):
        table = []

        # Header
        header = [""]
        for i in range(self.m):
            header.append(str(i))
        header.append("u_i")
        table.append(header)

        # Rows
        for j in range(self.n):
            row = [str(j)]
            for i in range(self.m):
                row.append(str(self.cost[i][j]))
            row.append(self.u[j])
            table.append(row)

        # Footer
        footer = ["v_j"]
        for j in range(self.n):
            footer.append(str(self.v[j]))
        table.append(footer)

        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", floatfmt=".3f"))

        pass

    def optimize(self):
        """
        Main optimization loop:
        First determines a feasible dual solution; afterwards finds a maximal number of independent zeroes.
        Iteratively adjusts the dual variables until the number independent  zeroes matches the problem dimension.

        Returns:
            (set, float): Set of independent zeroes corresponding to the optimal assignment and objective value.
        """

        if self.verbose:
            print("We are given the following assignment problem.")
            self.print()
        self.initialize()
        if self.verbose:
            print(f"The following dual-feasible solution has a cost of {self.get_objective()}.")
            self.print()
        self.get_independent_zeros()
        if self.verbose:
            print(
                f"The maximal number of independent zeroes corresponds to the (rows, cols): {self.independent}"
            )

        # If we have n=m independent zeros, we have found an optimal solution.
        while len(self.independent) != self.m:
            self.iterations += 1
            self.update_dual_variables()
            if self.verbose:
                print(
                    f"Updating the dual variables based on these independent zeroes yields the following tableau."
                )
                self.print()
            self.get_independent_zeros()
            if self.verbose:
                print(
                    f"The maximal number of independent zeroes corresponds to the (rows, cols): {self.independent}."
                )

        if self.verbose:
            print(
                f"We have m=n={self.m} independent zeros. The optimal assignment is {self.independent} with a cost of {self.get_objective():.2f}."
            )

        return self.independent, self.get_objective()


if __name__ == "__main__":

    M = float("inf")
    cost = [
        [M, 12, 4, 10, 10, 19],
        [12, M, 14, 8, 22, 14],
        [4, 14, M, 6, 8, 15],
        [10, 8, 6, M, 14, 9],
        [10, 22, 8, 14, M, 20],
        [19, 14, 15, 9, 20, M],
    ]
    hm = HungarianMethod(cost, verbose=True)
    assignment, objective = hm.optimize()
    assert assignment == set(((2, 4), (4, 0), (5, 1), (0, 2), (1, 3), (3, 5))) and objective == 53
