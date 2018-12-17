import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import cvxpy                as cp
import scipy.optimize       as spo

from libs.cost_function     import *
from libs.constraints       import convexConstraints, constraint_test
import libs.defines         as defs
import libs.dirs            as dirs
from libs.utils             import *

np.random.seed(17)

# Problem data.
M = defs.NUM_DIFFERENTIAL_PAIRS

## SCIPY
constraints = convexConstraints

# Initialize variables
deltaDiff0 = np.zeros(M)
maxVal = defs.MIN_DELTA_DIFF_VALUE+1.2e-2
minVal = defs.MIN_DELTA_DIFF_VALUE+1e-3
while True:
    deltaDiff0[0] = -1.
    deltaDiff0[1:] = np.random.random(M-1)*(maxVal - minVal) + minVal
    # deltaDiff0[1:] = np.ones(M-1)*np.random.random(1)*(maxVal - minVal) + minVal
    print("Min: ", minVal)
    print("Max: ", maxVal)
    print("Ripple:    ", get_ripple_percent(deltaDiff0))
    print("Bandwidth: ", get_bandwidth(deltaDiff0))
    if constraint_test(deltaDiff0, constraints) == True:
        break
        # delta = convert_delta(deltaDiff0)
        # print(deltaDiff0)
        # print(delta)
        # print(maxVal-minVal)

print("\nPreparing optimization")
print("MIN: ", defs.MIN_DELTA_DIFF_VALUE)
print("MAX: ", defs.MAX_DELTA_DIFF_VALUE)
print("Constraint test:")
print("deltaDiff: ", deltaDiff0)
print("Initial Delta: ", convert_delta(deltaDiff0))
print(constraint_test(deltaDiff0, constraints))
print("")

print("\nStarting optimization")
opts = {'disp':True}
def call_func(xk, convergence=0):
    print("Delta: ", xk)
    print("")
    # print("f(x): {:.2e}\n".format(cost_function(xk)))
result = spo.minimize(cost_function_alt, deltaDiff0,method='COBYLA', constraints=constraints, options=opts,
                        callback=call_func)
deltaDiffOpt = result.x
print("Minimization finished. Results:")
print("x*:     ", result.x)
print("f(x*):  ", result.fun)
print("FEvals: ", result.nfev)
print("Constraint Test:\n", constraint_test(result.x, constraints))
print("Ripple:    ", get_ripple_percent(deltaDiffOpt))
print("Bandwidth: ", get_bandwidth(deltaDiffOpt))
# input()

plot_results(deltaDiffOpt)
save_results(deltaDiffOpt, result.nfev, optimizer='COBYLA_'+str(defs.NUM_DIFFERENTIAL_PAIRS))

## Results
# x* = [0.05, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
# f(x*) = -5.114129518610489
