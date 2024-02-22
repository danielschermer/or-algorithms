from enum import Enum
import numpy as np
from tabulate import tabulate
from simplex.engine import Engine

np.set_printoptions(formatter={"float": "{: 0.3f}".format})

TOLERANCE = 1e-12


class Sense(Enum):
    Optimal = 1


class Variable:
    def __init__(self, idx, obj_coeff=0.0, lb=0.0, ub=np.inf, vtype="CONTINUOUS", name=None):
        self.obj_coeff = obj_coeff
        self.ub = ub
        self.lb = lb
        self.type = vtype
        self.name = name
        self.idx = idx


class Model:
    def __init__(self, sense="MAXIMIZE", verbose=False):
        self.sense = sense
        if self.sense == "MAXIMIZE":
            self.incumbent = float("-inf")
        elif self.sense == "MINIMIZE":
            self.incumbent = float("inf")
        else:
            raise Exception("Sense must be either 'MAXIMIZE' or 'MINIMIZE'")

        self.obj_coeffs = []
        self.constraints = []
        self.eq_constraints = []
        self.lb = []
        self.ub = []

        # Dicts for associating indices to names
        self.var_name_to_index = {}
        self.index_to_var_name = {}
        self.constraint_name_to_index = {}
        self.index_to_constraint_name = {}

        # Main  matrix, RHS vector and objective vector
        self.A: np.array
        self.b: np.array
        self.c: np.array

        # Optimization engine
        self.engine: Engine

        # Various counters
        self.cols, self.rows = 0, 0
        self.cnt_primal, self.cnt_dual = 0, 0

        # Parameters
        self.verbose = verbose
        self.initialized = False
        self.dual_naming = False
        self.status = "LOADED"  # INFEASIBLE, UNBOUNDED, OPTIMAL, INTEGER-OPTIMAL

        # Removable
        self.variables = set()

    def add_variable(self, objCoefficient=0.0, lb=0.0, ub=np.inf, varType="CONTINUOUS", name=None):
        """Adds a variable to the model and returns the variable object.

        Args:
            objCoefficient (float, optional): _description_. Defaults to 0.0.
            lb (float, optional): _description_. Defaults to 0.0.
            ub (_type_, optional): _description_. Defaults to np.inf.
            varType (str, optional): _description_. Defaults to "CONTINUOUS".
            name (_type_, optional): _description_. Defaults to None.

        Raises:
            Exception: If a variable with the same name already exists.

        Returns:
            _type_: A new Variable object.
        """

        if name is None:
            name = f"x_{self.cols}"
        if self.var_name_to_index.get(name) is not None:
            raise Exception("A variable with the name '%s' has already been added." % name)

        var = Variable(self.cols, objCoefficient, lb, ub, varType, name)
        if self.sense == "MAXIMIZE":
            self.obj_coeffs.append(objCoefficient)
        elif self.sense == "MINIMIZE":
            self.obj_coeffs.append(-objCoefficient)

        # Store the bounds
        self.lb.append(lb)
        self.ub.append(ub)

        # Update the dictionary
        self.var_name_to_index[var.name] = self.cols
        self.index_to_var_name[self.cols] = var.name

        # Update the counters
        self.cols += 1
        self.variables.add(var)

        return var

    def add_constraint(self, LinearExpression, operator="<=", rhs=0.0, name=None):
        """Adds a new constraint to the model

        Args:
            LinearExpression (_type_): _description_
            operator (str, optional): _description_. Defaults to "<=".
            rhs (float, optional): _description_. Defaults to 0.0.
            name (_type_, optional): _description_. Defaults to None.

        Raises:
            Exception: A constraint with the same name already exists.
            Exception: Wrong RHS-operator.
        """

        if name is None:
            name = "s_%i" % (self.rows + 1)
        if self.constraint_name_to_index.get(name) is not None:
            raise Exception("A constraint with the name '%s' has already been added." % name)
        self.index_to_constraint_name[self.rows] = name
        self.constraint_name_to_index[name] = self.rows
        self.rows += 1

        constraint = np.zeros(self.cols + 1)

        for _, term in enumerate(LinearExpression):
            coeff, var = term
            constraint[var.idx] = coeff
        constraint[-1] = rhs

        if operator == "<=":
            self.constraints.append(constraint)
        elif operator == ">=":
            self.constraints.append(-constraint)
        elif operator == "==":
            # TODO: Currently not supported
            raise Exception(
                "Constraints of type '==' are currently not supported. Explicitly use 'ax <= b' and 'ax >= b'!"
            )
            # TODO: self.eq_constraints.append(constraint)
        else:
            raise Exception("Constraint does not use '<=', '>=' or '=='.")

        return constraint

    def init_from_objects(self):
        """
        Initializes the necessary datastructures based on added variable and constraint objects.
        """

        # Build the A matrix
        A = np.array(self.constraints)
        self.A = A[:, :-1]
        self.b = A[:, -1]
        self.c = np.array(self.obj_coeffs)

        self.cols, self.rows = self.A.shape[1], self.A.shape[0]

        # Initialize the optimization engine
        self.engine = Engine(self.A, self.b, self.c)

        # self.init_variable_dict(self.dual_naming)
        self.initialized = True

    def init_from_values(self, A, b, c, sense="MAXIMIZE", dual_naming=False):
        """
        Initializes the necessary datastructures by directly passing a model of the form: 'max c^Tx, s.t. Ax <= b'.
        """
        self.cols, self.rows = A.shape[1], A.shape[0]
        self.cols, self.rows = self.cols, self.rows
        self.A, self.b, self.c = A, b, c
        # TODO: Enum
        self.status = "LOADED"  # INFEASIBLE, UNBOUNDED, OPTIMAL, INTEGER-OPTIMAL
        self.sense = sense
        # TODO: Toggle for Sense?
        # if self.sense == "MINIMIZE":
        #    self.c = -self.c
        self.incumbent = float("-inf")
        self.cnt_primal, self.cnt_dual = 0, 0

        self.engine = Engine(self.A, self.b, self.c)

        for i in range(self.cols):
            if not dual_naming:
                name = f"x_{i}"
            else:
                name = f"w_{i}"
            self.var_name_to_index[name] = i
            self.index_to_var_name[i] = name

        for j in range(self.rows):
            name = f"s_{j}"
            self.constraint_name_to_index[name] = j
            self.index_to_constraint_name[j] = name

        # self.init_variable_dict(dual_naming)
        self.initialized = True

        return

    def dualize(self):
        """
        Reformulates the current model as a dual program.
        """
        # TODO: As we always have a maximization problem, the sign of the optimization will change but the absolute value remains identical.
        a, b, c = self.dual_coefficients()
        self.init_from_values(a, b, c, dual_naming=True)

    def dual_coefficients(self):
        """
        Returns the dual coefficients.
        """
        return -self.A.T, -self.c, -self.b

    def get_objective(self):
        """Returns the objective value."""

        return self.engine.get_objective()

    def test_degeneracy(self):
        # self.update_rhs()
        if min(self.engine.get_rhs()) <= 0 + TOLERANCE:
            if self.verbose:
                print(
                    "Solution is primal degenerate: basic variable %s has value 0"
                    % self.index_to_name(self.engine.basis.b[np.argmin(self.engine.get_rhs())])
                )

        row = self.engine.get_A_tilde_row(-1)
        if min(abs(row[self.engine.basis.n][0:-1])) <= 0 + TOLERANCE:
            if self.verbose:
                print(
                    "Solution is dual degenerate: non-basic variable %s has value 0"
                    % self.index_to_name(
                        self.engine.basis.n[np.argmin(row[self.engine.basis.n][0:-1])]
                    )
                )

    def minimize_infeasibility(self):
        """Main improvement loop during phase 1. Given an infeasible starting solution:
        1.  Get the most-negative (most-infeasible) row.
        2.  Perform a pivotation that minimizes this infeasibility.
        3.  The previous step determines if the problem is infeasible or feasible (but not necessarily optimal).
        """
        min_rhs = min(self.engine.get_rhs())
        while min_rhs < 0:
            if min_rhs < 0 - TOLERANCE:
                self.cnt_dual += 1
                self.engine.pivot_dual(self)
                if self.status == "INFEASIBLE":
                    break

                self.engine.LU_step()

                min_rhs = min(self.engine.get_rhs())
                if self.verbose:
                    self.print_table()
            else:
                break

    def maximize_objective(self):
        """Main improvement loop during phase 2. Given a feasible starting solution:
        1.  Check if the solution is currently primal- or dual-degenerate.
        2.  Search for a pivotation that improves the objective value.
        3.  The previous step determines if the problem is feasible (but not yet optimal), unbounded, or optimal.
        """
        while True:
            self.test_degeneracy()
            self.engine.pivot_primal(self)

            if self.status == "UNBOUNDED":
                break

            if self.status == "OPTIMAL":
                if self.verbose:
                    print(
                        f"Optimal after {self.cnt_primal + self.cnt_dual} iterations with objective value: {self.get_objective()}"
                    )
                break

            self.engine.LU_step()
            self.get_objective()

            if self.verbose:
                self.print_table()

            self.cnt_primal += 1

    def optimize(self):
        """Main optimization loop. Consists of two phases:
        1.  Determine a feasible solution, if one exists.
        2.  Once a feasible solution has been found, optimize the solution

        Args:
            verbose (bool, optional): Prints each pivotation to the standard output. Defaults to False.
        """

        if not self.initialized:
            self.init_from_objects()
        if self.verbose:
            self.print_table()
        # Phase 1: Minimize the infeasibility
        self.minimize_infeasibility()
        if self.status == "INFEASIBLE":
            return
        # Phase 2: Optimization
        self.maximize_objective()

    def add_row(self, row):
        """Given a list of coefficients, adds a new row to the model."""
        # Account for a new slack variable
        self.engine.basis.add_row(self)

        # Add the row to the original problem
        A = np.insert(self.engine.get_current_matrix(), -1, row, axis=0)
        # Include a new slack variable for the row
        A[-1, -2] = 0
        A = np.insert(A, -1, np.eye(1, self.rows + 2, self.rows + 1), axis=1)

        # Update the B matrices to account for the added basic variable
        B = A[:, self.engine.basis.get_slack_basis()]
        B_I = B.I

        # The new problem in terms of the original space is A_tilde = B.I*A
        # self.engine.A_tilde = B_I * A

        # TODO: Lazy update of A_tilde for LU
        # Alternative, a lazy update is possible for A_Tilde matrix
        A_tilde_row = np.zeros((1, A.shape[1]))
        for v in self.engine.basis.get_variable_basis():
            A_tilde_row[0, v] = B_I[-2] * A[:, v]
        # New slack variable
        A_tilde_row[0, -3] = 1
        self.engine.A_tilde = np.insert(
            self.engine.A_tilde, -2, np.eye(1, self.rows + 1, -1), axis=1
        )
        self.engine.A_tilde = np.insert(self.engine.A_tilde, -1, A_tilde_row, axis=0)

        # Give the slack variable a new name in the model
        self.constraint_name_to_index[f"s_{self.rows}"] = self.rows
        self.index_to_constraint_name[self.rows] = f"s_{self.rows}"
        self.rows += 1

        self.engine.LU_step()

    def optimize_gomory(self):
        """Implements Gomory's Cutting-Plane method for solving Integer Linear Programs.
        Starts by solving the linear relaxation. Then iterates the following procedure until an integer-feasible solution is found:
        1.  Selects the row with the most-fractional right-hand side.
        2.  Constructs a Gomory cutting plane based on the chosen row.
        3.  Adds the row to the model and solves the resulting linear relaxation.
            As everything happens in the primal problem, this first requires restoring feasibility.

        """

        # The standard technique works only for integer-valued A and b
        assert (np.int64(self.A) == self.A).all() and (np.int64(self.b) == self.b).all()

        self.optimize()
        while max(abs(np.rint(self.engine.get_rhs()) - self.engine.get_rhs())) > TOLERANCE:
            if self.verbose:
                print("Current solution is integer-infeasible.")

            # Get the row associated with the most-fractional RHS
            rhs = self.engine.get_rhs()
            index = np.argmax((abs(np.rint(rhs) - rhs)))
            row = self.engine.get_A_tilde_row(index)

            # Construct the cutting plane
            cutting_plane = np.eye(
                1, self.engine.cols + self.engine.rows + 2, self.engine.cols + self.engine.rows
            )
            for i in self.engine.basis.n:
                cutting_plane[0, i] = np.floor(row[i]) - row[i]

            self.add_row(cutting_plane)

            # Express the cutting plane in terms of the original problem
            if self.verbose:
                s = "Cutting Plane (Row %s): " % (self.index_to_name(self.engine.basis.b[index]))
                for i in range(self.cols):
                    s += f"{self.engine.A_tilde[-2, i]:2f} {self.index_to_name(i)} + "

                s += " <= %.2f " % (self.engine.A_tilde[-2, -1])
                print(s)

            self.optimize()

        self.status = "INTEGER-OPTIMAL"
        if self.verbose:
            print("Solution is integer-feasible and optimal.")

    def index_to_name(self, idx):
        """
        Helper function: given an integer index, returns the name of the corresponding variable.
        """
        if idx < self.cols:
            name = self.index_to_var_name[idx]
        elif idx < self.cols + self.rows:
            name = self.index_to_constraint_name[idx - self.cols]
        elif idx == self.cols + self.rows:
            name = "Z"
        elif idx == self.cols + self.rows + 1:
            name = "RHS"
        else:
            assert False
        return name

    def print_table(self, reduced_tableau=True):
        """Prints the active Simplex tableau

        Args:
            reduced_tableau (bool, optional): Do not print column vectors associated with basic-variables. Defaults to True.
        """
        table = []
        header = [""]

        if reduced_tableau:
            # The non-basic variables are in the header
            for i in range(self.cols + 1):
                header.append(self.index_to_name(self.engine.basis.n[i]))
        else:
            # Decision variables followed by slack  are in the header:
            for i in range(self.cols + self.rows):
                header.append(self.index_to_name(i))
            header.append("RHS")

        table.append(header)
        result = self.engine.get_current_matrix()

        for i in range(self.rows + 1):
            name = np.array(self.index_to_name(self.engine.basis.b[i])).reshape(1)
            if reduced_tableau:
                row = np.array(result[i])[0, self.engine.basis.n]
            else:
                row = np.array(result[i]).flatten()
                row = row[[i for i in range(len(row)) if i is not len(row) - 2]]

            table.append(np.concatenate((name, row), axis=0))

        # print(tabulate(table, headers='firstrow', tablefmt='latex_raw'))
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", floatfmt=".3f"))

    def check_sequence(self, sequence):

        def translate(name_or_index):
            if self.index_to_var_name.get(name_or_index, None) is not None:
                return name_or_index
            elif self.var_name_to_index.get(name_or_index, None) is not None:
                return self.var_name_to_index[name_or_index]
            elif self.index_to_constraint_name.get(name_or_index, None) is not None:
                return name_or_index
            elif self.constraint_name_to_index.get(name_or_index, None) is not None:
                return self.cols + self.constraint_name_to_index[name_or_index]

        self.print_table()
        for enter, leave in sequence:
            enter = translate(enter)
            leave = translate(leave)
            self.engine.basis.swap_basis(self, enter, leave)
            self.engine.LU_step()
            self.print_table()
