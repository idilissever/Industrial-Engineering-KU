import numpy as np
import cplex as cp
import scipy.sparse as sp


def linear_programming(direction, A, senses, b, c, l, u):

    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist())

    # define problem type
    if direction == "maximize":
        prob.objective.set_sense(prob.objective.sense.maximize)
    else:
        prob.objective.set_sense(prob.objective.sense.minimize)

    # add constraints to the problem including their directions and right-hand side values
    prob.linear_constraints.add(senses = senses.tolist(), rhs = b.tolist())

    # add coefficients for each constraint
    row_indices, col_indices = A.nonzero()
    prob.linear_constraints.set_coefficients(zip(row_indices.tolist(),
                                                 col_indices.tolist(),
                                                 A.data.tolist()))

    # solve the problem
    prob.solve()

    # check the solution status
    print(prob.solution.get_status())
    print(prob.solution.status[prob.solution.get_status()])

    # get the solution
    x_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()

    return(x_star, obj_star)

    
def shortest_path(prob):
    costs = np.loadtxt(prob)

    E = costs.shape[0]
    N = max(costs[:, 0:2].flatten().astype(int))

    senses = np.repeat("E", N)

    c = costs[: ,-1]
    u = np.repeat(cp.infinity, E)
    l = np.repeat(0, E)

    b = np.concatenate((np.array([1]), np.repeat(0, N-2), np.array([-1])))

    aij = np.repeat([1, -1], E)
    row = np.concatenate((costs[:,0].astype(int)-1, costs[:,1].astype(int)-1))
    col = np.concatenate((range(E),range(E)))

    A = sp.csr_matrix((aij, (row, col)), shape = (N,E))



    (x_star, obj_star) = linear_programming("minimize",A , senses, b, c, l, u)
    return x_star, obj_star


print(shortest_path("shortest_path_problem1.txt"))

