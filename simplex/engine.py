import numpy as np

TOLERANCE = 1e-12


class Basis:
    """Helper class for storing indices of basic and non-basic variables."""

    def __init__(self, A):
        """Initializes a basis based on the slack variables."""
        self.cols = A.shape[1]
        self.rows = A.shape[0]
        # Initialization: Basic variables correspond to slack variables (associated with constraints)
        self.b = np.arange(self.cols, self.rows + self.cols + 1)
        # Initialization: Non-basic variables correspond to decision variables
        self.n = np.arange(0, self.cols)
        # The index associated with the objective must always be basic
        self.n = np.append(self.n, self.rows + self.cols + 1)

    def swap_basis(self, model, enter: int, leave: int) -> None:
        """Put value at index 'enter' from the non-basis into the basis and value at index 'leave' from the basis to the non-basis.

        Args:
            model (_type_): A model object.
            enter (int): Index of an entering variable.
            leave (int): Index of a leaving variable.
        """
        if model.verbose:
            print(
                f"Pivot: {model.index_to_name(self.n[enter])} enters the basis and {model.index_to_name(self.b[leave])} leaves the basis."
            )
        tmp = self.b[leave]
        self.b[leave] = self.n[enter]
        self.n[enter] = tmp

    def add_row(self, model):
        """Adds a new row to the basis. This is useful for generating Cutting-Planes.

        Args:
            model (_type_): A model object.
        """
        # Increment the index objective row
        model.engine.basis.b[-1] += 1
        # Increment the index of the RHS column
        model.engine.basis.n[-1] += 1
        # Add a new slack variable for the row (constraint) and make it a basic variable
        model.engine.basis.b = np.insert(model.engine.basis.b, -1, model.engine.basis.b[-1] - 1)
        # Increment the row counters
        self.rows += 1
        model.engine.rows += 1

    def get_slack_basis(self):
        return np.arange(self.cols, self.rows + self.cols + 1)

    def get_variable_basis(self):
        return np.append(np.arange(0, self.cols), self.rows + self.cols + 1)

    def get_basic_decision_variables(self):
        return [v for v in range(self.cols) if v in self.b]


class Engine:
    """This class does all the heavy lifting, in particular the matrix related operations."""

    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        self.A = A
        self.b = b
        self.c = c
        self.cols, self.rows = len(c), len(b)

        self.basis = Basis(A)
        # TODO: Replace by LU-Factorization
        self.A_Tilde: np.ndarray
        self.B_inv: np.ndarray

        self.init_A_tilde()
        self.LU_step(initialize=True)

    def LU_step(self, initialize=False):
        """Given a basis, updates the inverse of B.

        Args:
            initialize (bool, optional): If true initializes a diagonal slack basis. Defaults to False.
        """
        # TODO LU Factorization
        if initialize:
            self.B_inv = B = np.matrix(np.diag(np.full(self.rows + 1, 1)))
        else:
            B = np.matrix(self.A_tilde[:, self.basis.b])
            self.B_inv = B.I

    def get_current_matrix(self):
        """Return the state of the problem, given the current basis.

        Returns:
            np.ndarray: Has shape rows + 1 x cols + 2
        """
        return self.B_inv * self.A_tilde

    def get_A_tilde_col(self, col_index):
        """Calculate and return a single column of A_tilde.

        Args:
            col_index (int): The column index.

        Returns:
            np.ndarray: A vector corresponding to column[index] in A_tilde.
        """
        col = np.zeros(self.A_tilde.shape[0])
        for i in range(self.A_tilde.shape[0]):
            col[i] = self.B_inv[i, :] * self.A_tilde[:, col_index]

        return col

    def get_A_tilde_row(self, row_index):
        """Calculate and return a single row of A_tilde.

        Args:
            row_index (int): The row index.

        Returns:
            np.ndarray: A vector corresponding to row[index] in A_tilde.
        """
        row = np.zeros(self.A_tilde.shape[1])
        for i in range(self.A_tilde.shape[1]):
            row[i] = self.B_inv[row_index] * self.A_tilde[:, i]

        return row

    def get_rhs(self):
        """Returns the transpose of the current right-hand side (RHS) of the problem.

        Returns:
            np.ndarray: Has shape (1, rows).
        """
        return self.get_A_tilde_col(-1)[0:-1].flatten()

    def get_objective(self):
        """Returns the objective value.

        Returns:
            float: Objective value."""
        return float(self.B_inv[-1] * self.A_tilde[:, -1])

    def get_objective_row(self):
        """Returns the last row corresponding to the objective coefficients.

        Returns:
            np.ndarray: Has shape (1, cols+rows+1).
        """
        # Get the last row of the current inverse matrix
        return np.array(self.B_inv[-1] * self.A_tilde).flatten()

    def init_A_tilde(self):
        """Initializes the matrix A~ (A_tilde) which has the following shape:

        [A    0 | b]

        [-c^T 1 | 0]
        """

        # Add diagonal entries for slack variables
        A = np.concatenate((self.A, np.diag(np.full(self.rows, 1))), axis=1)
        # Add a zero objective coefficient for each row
        c = np.concatenate((self.c.reshape(1, self.cols), np.zeros((1, self.rows))), axis=1)
        # Append c to A
        A_tilde = np.concatenate((A, -c), axis=0)
        # Append the column vector (0, 0, ..., 1)
        A_tilde = np.concatenate((A_tilde, np.eye(1, self.rows + 1, self.rows).T), axis=1)
        # Append the column vector (b, 0)
        A_tilde = np.concatenate((A_tilde, np.append(self.b, 0).reshape(self.rows + 1, 1)), axis=1)

        self.A_tilde = np.matrix(A_tilde)

    def pivot_primal(self, model):
        """Primal Simplex"""

        enter, leave = None, None

        def find_entering(enter):
            # Get the last row of the current inverse matrix
            objective_row = self.get_objective_row()
            reduced_cost = float("inf")

            # For each non-basic variable
            for i in range(0, model.cols):
                idx = self.basis.n[i]
                red = np.round(objective_row[idx], 12)
                if red < 0 - TOLERANCE:
                    if red < reduced_cost:
                        enter = i
                        reduced_cost = red
            return enter

        enter = find_entering(enter)

        # If no entering solution exists, the solution must be optimal.
        if enter is None:
            model.status = "OPTIMAL"
            if model.verbose:
                print("Linear relaxation is optimal: no improving non-basic variable exists.")
            return

        def find_leaving(enter, leave):
            idx = self.basis.n[enter]
            pivot_column = self.get_A_tilde_col(idx)
            rhs = self.get_rhs()

            ratio = float("inf")

            for i, b in enumerate(rhs):
                if pivot_column[i] <= 0:
                    continue

                r = b / pivot_column[i]
                if r < ratio:
                    ratio = r
                    leave = i

            if leave is None:
                model.status = "UNBOUNDED"
                if model.verbose:
                    print(
                        f"Linear relaxation is unbounded: no valid leaving variable for entering {model.index_to_name(model.basis.n[enter])}."
                    )
                    return

            return leave

        leave = find_leaving(enter, leave)

        if leave is not None and enter is not None:
            self.basis.swap_basis(model, enter, leave)

    def pivot_dual(self, model):
        """Dual Simplex"""
        enter, leave = None, None

        # We want the basic variable with the most-negative RHS to leave the basis.
        def find_leaving():
            rhs = self.get_rhs()
            if np.min(rhs) >= 0:
                return None

            return np.argmin(rhs)

        leave = find_leaving()

        # We want to find an entering variable that minimizes the infeasibility of the current row (dual steepest-unit ascent).
        def find_entering(enter, leave):
            if leave is not None:
                z_row = self.get_A_tilde_row(-1)
                var_row = np.zeros(self.A_tilde.shape[1] - 1)
                var_row_inv = self.B_inv[leave]
                max_ratio = float("-inf")

                for i in self.basis.n[:-1]:
                    var_row[i] = var_row_inv * self.A_tilde[:, i]
                    if var_row[i] < 0 - TOLERANCE:
                        ratio = z_row[i] / var_row[i]
                        if ratio > max_ratio:
                            max_ratio = ratio
                            enter = int(np.argwhere(self.basis.n == i))

                if enter is None:
                    if model.verbose:
                        print(
                            f"Linear relaxation is infeasible: no valid column for leaving variable {model.index_to_name(model.basis.b[leave])}."
                        )
                    model.status = "INFEASIBLE"
                return enter

        enter = find_entering(enter, leave)

        if model.status != "INFEASIBLE":
            self.basis.swap_basis(model, enter, leave)
