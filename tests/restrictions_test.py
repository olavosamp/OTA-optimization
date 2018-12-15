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
               {'type':'ineq', # deltaDiff[0] <= 3 || -deltaDiff[0] +3 >=0
                'fun': lambda x: -x[0] +3.
                },
               {'type':'ineq', # deltaDiff[1:] >= 0
                'fun': lambda x: x[1:]
                },
               {'type':'ineq', # deltaDiff[1:] <= defs.SIGNAL_SPAN || -deltaDiff[1:] + span >=0
                'fun': lambda x: -x[1:] + defs.SIGNAL_SPAN
                },
               # {'type':'ineq',
               #  'fun':, lambda x:
               #  },
]

# Initialize variables
deltaDiff0 = np.zeros(M)
deltaDiff0[0] = np.random.random()*(3 +3) - 3
deltaDiff0[1:] = np.random.random(M-1)*(defs.SIGNAL_SPAN - 0) + 0

print(deltaDiff0)
for cons in constraints:
    func = cons['fun']
    print(func(deltaDiff0) >= 0)

# opts = {'disp':True}
# def call_func(xk, convergence=0):
#     print("Delta: ", xk)
#     print("")
#     # print("f(x): {:.2e}\n".format(cost_function(xk)))
# result = spo.minimize(cost_function, deltaDiff0, constraints=constraints, options=opts
#                         callback=call_func)
# print(result.x)
# print(result.f)
# print(result.nfev)