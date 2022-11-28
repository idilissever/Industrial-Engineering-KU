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
    
    print(prob.write_as_string())
    # get the solution
    x_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()

    return(x_star, obj_star)


arc_caps = np.loadtxt("maximal_flow_problem1.txt")
minNode = 1
maxNode = int(max(arc_caps[:,0:2].flatten().tolist()))
arc_caps = np.vstack((arc_caps, np.array([maxNode, minNode, cp.infinity])))

E = arc_caps.shape[0]
c = np.concatenate((np.repeat(0, E-1),np.array([1])))
N = maxNode
u = arc_caps[:,-1]
l = np.repeat(0,E)
b = np.repeat(0,N)
senses = np.repeat("E", N)

aij = np.repeat([1,-1], E)
row = np.concatenate((arc_caps[:,0].astype(int) - 1, arc_caps[:,1].astype(int) - 1))
col = np.concatenate((np.array(range(E)).astype(int),np.array(range(E)).astype(int)))

A = sp.csr_matrix((aij, (row,col)), shape = (N,E))
print(A)
# import matplotlib.pyplot as plt
# plt.figure(figsize = (6, 9))
# plt.spy(A, marker = "o")
# plt.show()

(xstar, objstar) = linear_programming("maximize", A, senses, b, c, l, u)
print(xstar, objstar)
