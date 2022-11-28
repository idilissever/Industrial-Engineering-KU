import numpy as np
import scipy.sparse as sp
import cplex as cp

def linear_programming(direction, A, senses, b, c, l, u, types):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist(), types = types.tolist())

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


def knapsack_problem(weights_file, values_file, capacity_file):

    weights = np.loadtxt(weights_file)
    print(weights)
    values = np.loadtxt(values_file)
    capacity = np.loadtxt(capacity_file)

    V = values.shape[0]

    b = np.array([capacity])
    print(b)
    c = values
    senses = np.array(["L"])
    types = np.repeat("B", V)
    l = np.repeat(0,V)
    u = np.repeat(1, V)

    A = sp.csr_matrix(weights)


    
    (x_star, obj) = linear_programming("maximize", A, senses, b , c, l, u, types)
    return (x_star, obj)

print(knapsack_problem("weights.txt","values.txt", "capacity.txt"))