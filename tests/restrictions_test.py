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
#     deltaDiffs[0]  in [-3, 3]
#     deltaDiffs[1:] in [0, defs.SIGNAL_SPAN]
#     ripple         <= 0.05
#     bandwidth      in [0.7, 0.9]
constraints = convexConstraints

# Initialize variables
deltaDiff0 = np.zeros(M)
# maxVal = defs.MIN_DELTA_DIFF_VALUE/defs.MAX_BW_VALUE+3e-3
maxVal = defs.MIN_DELTA_DIFF_VALUE+1.2e-2
minVal = defs.MIN_DELTA_DIFF_VALUE+1e-3
# minVal = 0
while True:
    deltaDiff0[0] = -1.
    deltaDiff0[1:] = minVal
    # deltaDiff0[1:] = np.ones(M-1)*np.random.random(1)*(maxVal - minVal) + minVal
    print("Min: ", minVal)
    print("Max: ", maxVal)
    print("Ripple:    ", get_ripple_percent(deltaDiff0))
    print("Bandwidth: ", get_bandwidth(deltaDiff0))
    if constraint_test(deltaDiff0, constraints) == True:
        delta = convert_delta(deltaDiff0)
        # break
        print(deltaDiff0)
        print(delta)
        print(maxVal-minVal)
        input()
    else:
        maxVal += 1e-4
        # minVal += 1e-3

# deltaDiff0 = np.zeros(M)
# maxVal = defs.MIN_DELTA_DIFF_VALUE/defs.MAX_BW_VALUE +3e-3
# minVal = defs.MIN_DELTA_DIFF_VALUE
# deltaDiff0[0] = -1.
# deltaDiff0[1:] = np.random.random(M-1)*(maxVal - minVal) + minVal
