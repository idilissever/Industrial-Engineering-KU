import cplex as cp
import numpy as np
import scipy.sparse as sp


def distance_between_ellipses(ellipses_file):
    ellipses = np.loadtxt(ellipses_file)
    prob = cp.Cplex()
    prob.variables.add(obj=[0, 0, 0, 0],
                       lb=[-cp.infinity, -cp.infinity, -
                           cp.infinity, -cp.infinity],
                       ub=[+cp.infinity, +cp.infinity, +
                           cp.infinity, +cp.infinity],
                       names=["x1", "y1", "x2", "y2"])

    prob.objective.set_sense(prob.objective.sense.minimize)

    Q = 2 * np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1],
                      [-1, 0, 1, 0],
                      [0, -1, 0, 1]])
    Q = sp.csr_matrix(Q)

    row_indices, col_indices = Q.nonzero()
    prob.objective.set_quadratic_coefficients(
        zip(row_indices.tolist(), col_indices.tolist(), Q.data.tolist()))

    Q1 = sp.csr_matrix(np.array([[ellipses[0, 0], ellipses[0, 1], 0, 0],
                                [0, ellipses[0, 2], 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]]))
    Q1_row_indices, Q1_col_indices = Q1.nonzero()

    Q2 = sp.csr_matrix(np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, ellipses[1, 0], ellipses[1, 1]],
                                 [0, 0, 0, ellipses[1, 2]]]))

    Q2_row_indices, Q2_col_indices = Q2.nonzero()

    prob.quadratic_constraints.add(lin_expr=cp.SparsePair([0, 1],
                                                          [ellipses[0, 3], ellipses[0, 4]]),
                                   quad_expr=cp.SparseTriple(Q1_row_indices.tolist(),
                                                             Q1_col_indices.tolist(),
                                                             Q1.data.tolist()),
                                   sense="L", rhs=-ellipses[0, -1])

    prob.quadratic_constraints.add(lin_expr=cp.SparsePair([2, 3],
                                                          [ellipses[1, 3], ellipses[1, 4]]),
                                   quad_expr=cp.SparseTriple(Q2_row_indices.tolist(),
                                                             Q2_col_indices.tolist(),
                                                             Q2.data.tolist()),
                                   sense="L", rhs=-ellipses[1, -1])

    prob.solve()

    x1_star, y1_star, x2_star, y2_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()
    distance_star = np.sqrt(obj_star)

    return (x1_star, y1_star, x2_star, y2_star, distance_star)


ellipses_file = "ellipses.txt"
print(distance_between_ellipses(ellipses_file))
