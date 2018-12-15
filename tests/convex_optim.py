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
# Inequality constraints in form
# f_i(x) >= 0
# INEQUALITIES TO IMPLEMENT
#  [v] deltaDiffs[0]  in [-3, 0]
#  [v] deltaDiffs[1:] in [defs.MIN_DELTA_DIFF_VALUE, defs.SIGNAL_SPAN]
#  [x] ripple         <= 0.05
#  [x] bandwidth      in [0.7, 0.9]
constraints = [
               {'type':'ineq', # deltaDiff[0] >= -3
                 'fun': lambda x: x[0] +3.
                 },
               {'type':'ineq', # deltaDiff[0] <= 0 || -deltaDiff[0] +0 >=0
                'fun': lambda x: -x[0]
                },
               {'type':'ineq', # deltaDiff[1:] >= defs.MIN_DELTA_DIFF_VALUE
                'fun': lambda x: x[1:] - defs.MIN_DELTA_DIFF_VALUE
                },
               {'type':'ineq', # deltaDiff[1:] <= defs.SIGNAL_SPAN || -deltaDiff[1:] + span >=0
                'fun': lambda x: -x[1:] + defs.SIGNAL_SPAN
                },
               {'type':'ineq', # ripple <= 0.05 || -ripple +0.05 => 0
                'fun':, lambda x: -get_ripple_percent(x) + 0.05
                },
               {'type':'ineq', # bandwidth <= 0.9 || -bandwidth + 0.9 >= 0
                'fun':, lambda x: -get_bandwidth(x) +0.9
                },
               {'type':'ineq', # bandwidth >= 0.7 || bandwidth - 0.7 >= 0
                'fun':, lambda x: get_bandwidth(x) -0.7
                },
]

# Initialize variables
deltaDiff0 = np.zeros(M)
deltaDiff0[0] = np.random.random()*(3 +3) - 3
deltaDiff0[1:] = np.random.random(M-1)*(defs.SIGNAL_SPAN - defs.MIN_DELTA_DIFF_VALUE) + defs.MIN_DELTA_DIFF_VALUE

opts = {'disp':True}
def call_func(xk, convergence=0):
    print("Delta: ", xk)
    print("")
    # print("f(x): {:.2e}\n".format(cost_function(xk)))
result = spo.minimize(cost_function_alt, deltaDiff0, constraints=constraints, options=opts,
                        callback=call_func)
print(result.x)
print(result.fun)
print(result.nfev)

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
