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

costs = np.loadtxt("costs25.txt")
capacities = np.loadtxt("capacities25.txt")
flows = np.loadtxt("flows25.txt")


K = max(costs[:,2].astype(int))
E = capacities.shape[0]
N = max(capacities[: , 0:2].flatten()).astype(int)

print(N)

print(K)

c = costs[: , -1]
b1 = flows[: , -1]
b2 = capacities[: , -1]
b = np.concatenate((b1,b2))

u = np.repeat(cp.infinity, E*K)
l = np.repeat(0, E*K)

senses = np.concatenate((np.repeat("E", N*K),np.repeat("L",E))) 

aij = np.repeat([1, -1], E)
print(aij)
row = np.concatenate((capacities[:, 0].astype(int)-1,capacities[:, 1].astype(int)-1))
print(row)
col = np.concatenate((np.array(range(E)),np.array(range(E))))
print(col)

A_sparse = sp.csr_matrix((aij,(row,col)), shape = (N,E))
A_array = A_sparse.toarray()

A1 = np.kron(np.eye(K,dtype=int),A_array)


A2 = np.zeros((E, E*K))


for i in range(E):
    A2[i][i] = 1
    for j in range(K):
        A2[i][i+j*E] = 1
    

A = sp.csr_matrix(np.vstack((A1, A2)))

print(linear_programming("minimize", A, senses, b, c, l, u))


