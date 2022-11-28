import numpy as np
import scipy.sparse as sp
import cplex as cp

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


resources = np.loadtxt("resources1000.txt")
products = np.loadtxt("products2500.txt")

P = products.shape[0]
R = resources.shape[0]

c = np.concatenate((products[: , 1], products[: , 2]))

u = np.repeat(cp.infinity, 2*P)
l = np.repeat(0, 2*P)

senses = np.concatenate((np.repeat("G", P), np.repeat("L", R)))

b = np.concatenate((products[:, -1], resources[:, -1]))

aij = np.concatenate((np.repeat(1,2*P), resources[:,1:-1].flatten()))
row = np.concatenate((np.repeat(np.array(range(P)),2), np.repeat(np.array(range(R)),P) + P))

col = []

for i in range(P):
    col.append(i)
    col.append(i+P)

col = np.array(col)
col = np.concatenate((col, np.tile(range(P), R)))

A = sp.csr_matrix((aij, (row,col)), shape = (P+R, 2*P))
    
(XSTAR, OBJSTAR) = linear_programming("minimize",A, senses, b, c, l ,u)
print(XSTAR, OBJSTAR)