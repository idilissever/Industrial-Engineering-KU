import numpy as np
import cplex as cp
import scipy.sparse as sp

def quadratic_programming(direction, A, senses, b, c, Q, l, u):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their linear coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist())
    
    # add quadratic coefficients in objective
    row_indices, col_indices = Q.nonzero()
    prob.objective.set_quadratic_coefficients(zip(row_indices.tolist(), col_indices.tolist(), Q.data.tolist()))

    # define problem type
    if direction == "maximize":
        prob.objective.set_sense(prob.objective.sense.maximize)
    else:
        prob.objective.set_sense(prob.objective.sense.minimize)

    # add constraints to the problem including their directions and right-hand side values
    prob.linear_constraints.add(senses = senses.tolist(), rhs = b.tolist())

    # add coefficients for each constraint
    row_indices, col_indices = A.nonzero()
    prob.linear_constraints.set_coefficients(zip(row_indices.tolist(), col_indices.tolist(), A.data.tolist()))

    print(prob.write_as_string())
    # solve the problem
    prob.solve()

    # check the solution status
    print(prob.solution.get_status())
    print(prob.solution.status[prob.solution.get_status()])

    # get the solution
    x_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()

    return(x_star, obj_star)

def distance_between_squares(squares_file):
    squares = np.loadtxt(squares_file)

    square1 = squares[0]
    square2 = squares[1]
    
    #square 1 values
    a1 = square1[0]
    b1 = square1[1]
    r1 = square1[2]

    #square 2 values
    a2 = square2[0]
    b2 = square2[1]
    r2 = square2[2]

    #bounds
    u = np.repeat(cp.infinity, 4)
    l = np.repeat(-cp.infinity, 4)

    #right hand side values
    b = np.array([a1 - 0.5*r1, a1 + 0.5*r1, b1 - 0.5*r1, b1 + 0.5*r1, 
                  a2 - 0.5*r2, a2 + 0.5*r2, b2 - 0.5*r2, b2 + 0.5*r2])
    
    #senses
    senses = np.tile(np.array(["G", "L"]), 4)

    #A matrix 
    row = np.array(range(8))
    col = np.repeat(range(4), 2)
    aij = np.repeat(1, 8)

    A = sp.csr_matrix((aij, (row, col)), shape = (8, 4))
    
    #coefficients for linear terms
    c = np.repeat(0, 4)

    #Q matrix for quadratic term coefficients
    Q = 2 * np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1],
                      [-1, 0, 1, 0],
                      [0, -1, 0, 1]])
    Q = sp.csr_matrix(Q)    
    

    x_star, obj_star = quadratic_programming("minimize", A, senses, b, c, Q, l, u)
    x1_star = x_star[0]
    y1_star = x_star[1]
    x2_star = x_star[2]
    y2_star = x_star[3]

    distance_star = np.sqrt(obj_star)
    
    return (x1_star, y1_star, x2_star, y2_star, distance_star)


print(distance_between_squares("squares.txt"))
