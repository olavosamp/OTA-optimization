import numpy                as np

from libs.cost_function     import *
import libs.defines         as defs
import libs.dirs            as dirs

## SCIPY
# Inequality constraints in form
# f_i(x) >= 0
# INEQUALITIES TO IMPLEMENT
#     deltaDiffs[0]  in [-1, 0]
#     deltaDiffs[1:] in [0, defs.SIGNAL_SPAN]
#     ripple         <= 0.05
#     bandwidth      in [0.7, 0.9]
convexConstraints = [
                   {'type':'ineq', # deltaDiff[0] >= -1
                     'fun': lambda x: x[0] +(1. -0)
                     },
                   {'type':'ineq', # deltaDiff[0] <= 0 || -deltaDiff[0] +0 >=0
                    'fun': lambda x: -x[0] - 1e-9
                    },
                   {'type':'ineq', # deltaDiff[1:] >= defs.MIN_DELTA_DIFF_VALUE
                    'fun': lambda x: x[1:] - (defs.MIN_DELTA_DIFF_VALUE - 1e-9)
                    },
                   {'type':'ineq', # deltaDiff[1:] <= defs.SIGNAL_SPAN || -deltaDiff[1:] + span >=0
                    'fun': lambda x: -x[1:] + (defs.SIGNAL_SPAN - 1e-9)
                    },
                   {'type':'ineq', # ripple <= 0.5 || -ripple +0.5 => 0
                    'fun': lambda x: -get_ripple_percent(x) + 0.5
                    },
                   {'type':'ineq', # bandwidth <= 0.9 || -bandwidth + 0.9 >= 0
                    'fun': lambda x: -get_bandwidth(x) +(defs.MAX_BW_VALUE - 1e-9)
                    },
                   {'type':'ineq', # bandwidth >= 0.7 || bandwidth - 0.7 >= 0
                    # 'fun': lambda x: get_bandwidth(x) -0.7
                    'fun': lambda x: get_bandwidth(x) - (0.7/defs.MAX_BW_VALUE - 1e-9)
                    },
]

def constraint_test(deltaDiff, constraints):
    passing = []
    for cons in constraints:
        func = cons['fun']
        passing.append(np.all(func(deltaDiff) >= 0))

    if np.all(passing) == False:
        print(passing)

    return np.all(passing)
