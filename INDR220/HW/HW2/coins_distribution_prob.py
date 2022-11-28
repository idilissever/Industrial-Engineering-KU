import numpy as np
import scipy.sparse as sp
import cplex as cp

def linear_programming(direction, A, senses, b, c, l, u, types):
    problem = cp.Cplex()
    
    #add decision variables with coefficients and bounds
    problem.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist(), types = types.tolist())
    
    #problem type
    if direction == "minimize":
        problem.objective.set_sense(problem.objective.sense.minimize)
    elif direction == "maximize":
        problem.objective.set_sense(problem.objective.sense.minimize)
        
    problem.linear_constraints.add(senses = senses.tolist(), rhs = b.tolist())
    
    row_indices, col_indices = A.nonzero()
    
    problem.linear_constraints.set_coefficients(zip(row_indices.tolist(), col_indices.tolist(), A.data.tolist()))
    
    problem.solve()
    
    print(problem.solution.get_status())
    print(problem.solution.status[problem.solution.get_status()])

    
    x_star = problem.solution.get_values()
    obj_star = problem.solution.get_objective_value()

    return(x_star, obj_star)
    


def coins_distribution_problem(coins_file, M):
    coins = np.loadtxt(coins_file).astype(int)
    C = coins.shape[0] #=4
    total_amount = 0
    V = C*M #=8

   
    for i in range(C):
        total_amount += coins[i]
    
    b = np.concatenate((np.repeat(total_amount/M ,M), np.repeat(1, C)))
    senses = np.repeat("E", M+C)
    c = np.repeat(1, V)
    l = np.repeat(0, V)
    u = np.repeat(1, V)
    types = np.repeat("B", V)

    aij = np.concatenate((np.tile(coins, M), np.repeat(1, V)))
    row = np.concatenate((np.repeat(range(M), C), np.repeat(range(C), M) + M))
    col1 = []

    for i in range(M):
        for j in range(C):
            col1.append(i + 2*j)

    col = np.concatenate((np.array(col1), np.array(range(V))))

    A = sp.csr_matrix((aij , (row,col)), shape = (C + M, V))

    x_star, obj_star = linear_programming("minimize", A, senses, b, c, l, u, types)

    X_star = []

    for i in range(M):
        X_star.append(x_star[i*C:(i+1)*C])
    
    X_star = np.array(X_star)


    return X_star





