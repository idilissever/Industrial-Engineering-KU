import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import cplex as cp


def linear_programming(direction, A, senses, b, c, l, u, types):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their coefficients in objective and ranges
    prob.variables.add(obj=c.tolist(), lb=l.tolist(),
                       ub=u.tolist(), types=types.tolist())

    # define problem type
    if direction == "maximize":
        prob.objective.set_sense(prob.objective.sense.maximize)
    else:
        prob.objective.set_sense(prob.objective.sense.minimize)

    # add constraints to the problem including their directions and right-hand side values
    prob.linear_constraints.add(senses=senses.tolist(), rhs=b.tolist())

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


def create_constraint_matrix(M, N):
    A = []
    for i in range(M):
        for j in range(N):
            # check all attacking positions
            if i + 2 < M:
                if j + 1 < N:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i+2)*N + j+1] = 1
                    if row not in A:
                        A.append(row)

                if j - 1 >= 0:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i+2)*N + j-1] = 1
                    if row not in A:
                        A.append(row)
            if i - 2 >= 0:
                if j + 1 < N:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i-2)*N + j+1] = 1
                    if row not in A:
                        A.append(row)
                if j - 1 >= 0:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i-2)*N + j-1] = 1
                    if row not in A:
                        A.append(row)
            if j + 2 < N:
                if i + 1 < M:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i+1)*N + j+2] = 1
                    if row not in A:
                        A.append(row)
                if i - 1 >= 0:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i-1)*N + j+2] = 1
                    if row not in A:
                        A.append(row)
            if j - 2 >= 0:
                if i + 1 < M:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i+1)*N + j-2] = 1
                    if row not in A:
                        A.append(row)
                if i - 1 >= 0:
                    row = [0] * (M*N)
                    row[i*N + j] = 1
                    row[(i-1)*N + j-2] = 1
                    if row not in A:
                        A.append(row)

    return A


def nonattacking_knights_problem(M, N):
    A_array = np.array(create_constraint_matrix(M, N))

    V = M*N  # number of decision variables
    E = A_array.shape[0]  # number of constraints

    senses = np.repeat("L", E)
    b = np.repeat(1, E)
    c = np.repeat(1, V)
    l = np.repeat(0, V)
    u = np.repeat(1, V)
    types = np.repeat("B", V)

    A = sp.csr_matrix(A_array)

    X_star, obj_star = linear_programming(
        "maximize", A, senses, b, c, l, u, types)
    return X_star, obj_star


print(nonattacking_knights_problem(3, 3))
