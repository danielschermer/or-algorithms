import numpy as np
from simplex.model import Model


if __name__ == "__main__":
    # There are two ways  to initialize a model

    # 1. By explicitly initializing variables and constraints
    m = Model(sense="MAXIMIZE", verbose=True)
    x = m.add_variable(objCoefficient=2.0, name="x")
    y = m.add_variable(objCoefficient=1.0, name="y")
    m.add_constraint([(1.0, x), (2.0, y)], "<=", 8.0, name="c_1")
    m.add_constraint([(1.0, x)], "<=", 4.0, name="c_2")
    m.optimize()

    # 2. By passing the coefficients of A, b, and c
    # In this case:
    # - We assume to have the standard form max c^T*x s.t. A*x <= b
    # - Decision variables are enumerated by (x_0, ..., x_n)
    # - Slack variables associated with constraints are enumerated by (c_0, ..., c_m)
    A = np.array([[1, 2], [1, 0]])
    b = np.array([8, 4])
    c = np.array([2, 1])
    m = Model(verbose=True)
    m.init_from_values(A, b, c)

    # The dual problem can be solved by calling
    m.dualize()
    m.optimize()
