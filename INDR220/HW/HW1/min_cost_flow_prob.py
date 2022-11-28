import numpy as np
import scipy.sparse as sp
import cplex as cp

def linear_programming(direction, A, senses, b, c, l, u):
    problem = cp.Cplex()
    
    #add decision variables with coefficients and bounds
    problem.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist())
    
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
    
    
def minimum_cost_flow_problem(costs_file, capacities_file, flows_file):
    
    costs0 = np.loadtxt(costs_file, dtype = float)
    capacities0 = np.loadtxt(capacities_file, dtype = float)
    flows0 = np.loadtxt(flows_file, dtype = float)
    
    
    listCosts = costs0.tolist()
    listCosts.sort()
    costs = np.array(listCosts)
    
    listCapacities = capacities0.tolist()
    listCapacities.sort()
    capacities = np.array(listCapacities)
    
    listFlows = flows0.tolist()
    listFlows.sort()
    flows = np.array(listFlows)
    
    
    V = np.max(costs[: , 0 : 2]).astype(int)
    E = costs.shape[0]
    c = costs[:,2]
    b = flows[:,1]
    l = np.repeat(0 , E)
    u = capacities[:,-1]
    senses = np.repeat("E", V)
    
    
    aij = np.repeat([+1.0, -1.0], E)
    
    row = np.concatenate((capacities[:,0].astype(int) - 1, capacities[:,1].astype(int) - 1))
    col = np.concatenate((range(E), range(E)))
    A = sp.csr_matrix((aij, (row, col)), shape = (V, E))    
    
    x_star, obj_star = linear_programming("minimize", A, senses, b, c, l, u) 
    return(x_star, obj_star)



x_star, obj_star = minimum_cost_flow_problem("costs.txt","capacities.txt", "flows.txt")

print(x_star)
print(obj_star)

