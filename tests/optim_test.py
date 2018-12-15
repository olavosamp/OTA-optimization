import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import cvxpy                as cp
import scipy.optimize       as spo

from libs.cost_function     import *
import libs.defines         as defs
import libs.dirs            as dirs


# Problem data.
M = defs.NUM_DIFFERENTIAL_PAIRS

## SCIPY
# bounds = [(-3.,3.),
#           (0,0.5),
#           (0,0.5),
#           (0,0.5),
# ]
bounds = [(-3.,3.)]
for i in range(1,M):
    bounds.append((0,0.5))

# deltaTest = [ 4.6067978,   4.72363884,  4.34032225, -1.10735863]
# # print
# print(cost_function(deltaTest))
# exit()

def call_func(xk, convergence=0):
    print("Delta: ", xk)
    print("f(x): {:.2e}\n".format(cost_function(xk)))
result = spo.differential_evolution(cost_function, bounds, disp=True, callback=call_func)
print(result.x)
print(result.nfev)

## CVXPY
# Construct the problem.
# delta = cp.Variable(M)
# expr = cp.expressions.expression.Expression(cost_function_exact(delta))
# print(expr.is_convex())
# # input()
# objective = cp.problems.objective.Minimize(cost_function_exact(delta))
# constraints = [(delta >= -5), # Operators are elementwise
#                (delta <= 5 )
#                ]
# prob = cp.Problem(objective, constraints)
#
# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print(prob.status)
# print(delta.value)
# # The optimal Lagrange multiplier for a constraint is stored in
# # `constraint.dual_value`.
# print(constraints[0].dual_value)
