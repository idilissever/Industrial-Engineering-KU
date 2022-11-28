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


E = np.loadtxt("employees.txt").astype(int)
print(E)
D = np.loadtxt("days.txt").astype(int)
S = np.loadtxt("shifts.txt").astype(int)

N = E*D*S
print(N)

c = np.repeat(1, N)
b = np.repeat(1, D*S+E*D+E)

l = np.repeat(0, N)
u = np.repeat(1, N)
types = np.repeat("B", N)

senses = np.concatenate((np.repeat("E", D*S), np.repeat("L", E*D), np.repeat("G", E)))

aij = np.repeat(1, N*3)
row = np.concatenate((np.repeat(range(D*S), E), np.repeat(range(E*D), S) + D*S, np.repeat(range(E), D*S) + D*S + E*D))


col = []

for i in range(D*S):
    for j in range(E):
        col.append(i+j*D*S)


col = np.concatenate((np.array(col), np.array(range(N)), np.array(range(N))))

A = sp.csr_matrix((aij, (row,col)), shape = (D*S+E*D+E, N))

# import matplotlib.pyplot as plt
# plt.figure(figsize = (6.4, 7.0))
# plt.spy(A, marker = "o", markersize = 6)
# plt.show()

print(linear_programming("maximize", A, senses, b, c, l, u, types))