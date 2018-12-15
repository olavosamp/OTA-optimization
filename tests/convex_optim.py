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
#     deltaDiffs[0]  in [-3, 3]
#     deltaDiffs[1:] in [0, defs.SIGNAL_SPAN]
#     ripple         <= 0.05
#     bandwidth      in [0.7, 0.9]
constraints = [
               {'type':'ineq', # deltaDiff[0] >= -3
                 'fun': lambda x: x[0] +3.
                 },
               {'type':'ineq', # deltaDiff[0] <= 3
                'fun': lambda x: x[0] -3.
                },
               {'type':'ineq', # deltaDiff[1:] >= 0
                'fun': lambda x: x[1:]
                },
               {'type':'ineq', # deltaDiff[1:] <= defs.SIGNAL_SPAN
                'fun': lambda x: x[1:] - defs.SIGNAL_SPAN
                },
               # {'type':'ineq',
               #  'fun':, lambda x:
               #  },
]

# Initialize variables
deltaDiff0 = np.zeros(M)
deltaDiff0[0] = np.random.random()*(3 +3) - 3
deltaDiff0[1:] = np.random.random(M-1)*(defs.SIGNAL_SPAN - 0) + 0

opts = {'disp':True}
result = spo.minimize(cost_function, deltaDiff0, constraints=constraints, options=opts)
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
