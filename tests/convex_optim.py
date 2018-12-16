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
# Inequality constraints in form
# f_i(x) >= 0
# INEQUALITIES TO IMPLEMENT
#  [v] deltaDiffs[0]  in [-3, 0]
#  [v] deltaDiffs[1:] in [defs.MIN_DELTA_DIFF_VALUE, defs.SIGNAL_SPAN]
#  [x] ripple         <= 0.5
#  [x] bandwidth      in [0.7, 0.9]
constraints = convexConstraints

# Initialize variables
# deltaDiff0 = np.zeros(M)
# maxVal = defs.MIN_DELTA_DIFF_VALUE +3e-3
# minVal = defs.MIN_DELTA_DIFF_VALUE
# while True:
#     deltaDiff0[0] = -1.
#     deltaDiff0[1:] = np.random.random(M-1)*(maxVal - minVal) + minVal
#     # print(delta)
#     print("Min: ", minVal)
#     print("Max: ", maxVal)
#     if constraint_test(deltaDiff0, constraints) == False:
#         delta = convert_delta(deltaDiff0)
#         print(deltaDiff0)
#         print(delta)
#         print(maxVal-minVal)
#         input()
#     else:
#         maxVal += 1e-5
#         # minVal += 1e-5
#
#
#
# exit()
# deltaDiff0 = [-1.]
# for i in range(1, M):
#     deltaDiff0.append(defs.MIN_DELTA_DIFF_VALUE +1e-9)
# deltaDiff0 = np.array(deltaDiff0)

deltaDiff0 = np.zeros(M)
maxVal = defs.MIN_DELTA_DIFF_VALUE +3e-3
minVal = defs.MIN_DELTA_DIFF_VALUE
deltaDiff0[0] = -1.
deltaDiff0[1:] = np.random.random(M-1)*(maxVal - minVal) + minVal

# resultFeasible = spo.minimize(lambda x: 1, np.zeros(), constraints=constraints, options=opts)

print("Constraint test:")
print("deltaDiff: ", deltaDiff0)
print("Initial Delta: ", convert_delta(deltaDiff0))
print(constraint_test(deltaDiff0, constraints))
# exit()

opts = {'disp':True}
def call_func(xk, convergence=0):
    print("Delta: ", xk)
    print("")
    # print("f(x): {:.2e}\n".format(cost_function(xk)))
result = spo.minimize(cost_function_alt, deltaDiff0, constraints=constraints, options=opts,
                        callback=call_func)
print("Minimization finished. Results:")
print("x*:     ", result.x)
print("f(x*):  ", result.fun)
print("FEvals: ", result.nfev)
print("Constraint Test:\n", constraint_test(result.x, constraints))
input()
# deltaDiffOpt = result.x
# x, y = get_xy(deltaDiffOpt)
# plt.plot(x, y)
# plt.show()

## Results
# x* = [0.05, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
# f(x*) = -5.114129518610489

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
