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

S = 3
D = 4


names = np.array(["x_{}_{}".format(i + 1, j + 1) 
                      for i in range(S)
                        for j in range(D)])

c = np.array([8, 6, 10, 9, 9, 12, 13, 7, 14, 9, 16, 5])

senses = np.repeat("E", S+D)

b = np.array([35, 50, 40, 45, 20, 30, 30])

l = np.repeat(0, S*D)

u = np.repeat(cp.infinity, S*D)


A1 = np.zeros((S , S*D))
A2 = np.eye(D)
temp_A2 = np.eye(D) 

for i in range(1,S):
    A2 = np.hstack((A2, temp_A2))


for i in range(S):
    for j in range(D):
        A1[i][ D*i + j] = 1 


A = sp.csr_matrix(np.concatenate((A1,A2)))


print(linear_programming("minimize", A, senses, b, c, l, u))



