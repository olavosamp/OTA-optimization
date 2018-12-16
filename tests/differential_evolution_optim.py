import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import cvxpy                as cp
import scipy.optimize       as spo

from libs.cost_function     import *
from libs.constraints       import convexConstraints, constraint_test
import libs.defines         as defs
import libs.dirs            as dirs


# Problem data.
M = defs.NUM_DIFFERENTIAL_PAIRS

## SCIPY
bounds = [(-1.,0.)]
for i in range(1,M):
    bounds.append((defs.MIN_DELTA_DIFF_VALUE, defs.MIN_DELTA_DIFF_VALUE+3e-3))

def call_func(xk, convergence=0):
    print("Delta: ", xk)
    print("")
    # print("f(x): {:.2e}\n".format(cost_function(xk)))
result = spo.differential_evolution(cost_function_alt, bounds, disp=True, callback=call_func)
print("Minimization finished. Results:")
print("x*:     ", result.x)
print("f(x*):  ", result.fun)
print("FEvals: ", result.nfev)
print("Constraint Test:\n", constraint_test(result.x, constraints))
input()
## Result DE
# x* = [0.09247358, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
# f(x*) = -6.29
# FEvals


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
