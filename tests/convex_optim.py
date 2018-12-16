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
constraints = convexConstraints

# Initialize variables
deltaDiff0 = np.zeros(M)
maxVal = defs.MIN_DELTA_DIFF_VALUE/defs.MAX_BW_VALUE+3e-3
minVal = defs.MIN_DELTA_DIFF_VALUE
while True:
    deltaDiff0[0] = -1.
    deltaDiff0[1:] = np.random.random(M-1)*(maxVal - minVal) + minVal
    # print(delta)
    print("Min: ", minVal)
    print("Max: ", maxVal)
    print("Ripple:    ", get_ripple_percent(deltaDiff0))
    print("Bandwidth: ", get_bandwidth(deltaDiff0))
    break
    # if constraint_test(deltaDiff0, constraints) == True:
    #     delta = convert_delta(deltaDiff0)
    #     break
        # print(deltaDiff0)
        # print(delta)
        # print(maxVal-minVal)
        # input()
#     else:
#         # maxVal += 1e-5
#         minVal += 1e-5
# exit()

# deltaDiff0 = np.zeros(M)
# maxVal = defs.MIN_DELTA_DIFF_VALUE/defs.MAX_BW_VALUE +3e-3
# minVal = defs.MIN_DELTA_DIFF_VALUE
# deltaDiff0[0] = -1.
# deltaDiff0[1:] = np.random.random(M-1)*(maxVal - minVal) + minVal

print("Preparing optimization")
print("MIN: ", defs.MIN_DELTA_DIFF_VALUE)
print("MAX: ", defs.MAX_DELTA_DIFF_VALUE)
print("Constraint test:")
print("deltaDiff: ", deltaDiff0)
print("Initial Delta: ", convert_delta(deltaDiff0))
print(constraint_test(deltaDiff0, constraints))

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
input()
deltaOpt = convert_delta(deltaDiffOpt)
x, y = get_xy(deltaOpt)
plt.plot(x, y)

plt.xlabel("Tensão (V)")
plt.ylabel("Corrente (A)")
plt.savefig(dirs.figures+"response_sum.png", orientation='portrait',
            bbox_inches='tight')
plt.show()


## Results
# x* = [0.05, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
# f(x*) = -5.114129518610489
