import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function         import *
import libs.defines             as defs
import libs.dirs                as dirs

M       = defs.NUM_DIFFERENTIAL_PAIRS   # Number of differential pairs
span    = 0.08                          # Non-zero response width
spacing = span/2
edge    = (M-1)/2*spacing
delta   = np.linspace(-edge,edge, num=M)

lowerBound = delta[0]  -0.5
upperBound = delta[-1] +0.5

x = np.linspace(lowerBound, upperBound, num=round(defs.PLOT_POINT_DENSITY*span))
y = sum_function(x, delta, M=M)

# Plot function with full span
fig = plt.figure(figsize=(20,10))
plt.plot(x,y)
plt.xlim(-edge-span*2, +edge+span*2)

# Plot component functions
for deltai in delta:
    plt.plot(x, differential_pair_response(x, deltai))

plt.xlabel("Tensão (V)")
plt.ylabel("Transcondutância (gm)")

# Compute dropoff as x where f(x) is 80% of max(y)

dropoff       = 0.8*np.max(y)
dropoffIndex  = np.squeeze(np.argwhere(np.isclose(y, dropoff , atol=1e-9)))
dropoffLeft   = dropoffIndex[0]
dropoffRight  = dropoffIndex[-1]

leftBound     = np.squeeze(x[dropoffIndex])[0]
rightBound    = np.squeeze(x[dropoffIndex])[-1]

# Compute dropoff points as first and last component signal peaks (does not necessarily
# correspond to signal sum peaks)
dropoffLeft2  = np.squeeze(np.argwhere(np.isclose(x, defs.DISTANCE_TO_MAX+ delta[0],
                          atol=1e-4)))
dropoffRight2 = np.squeeze(np.argwhere(np.isclose(x, defs.DISTANCE_TO_MAX+ delta[-1],
                          atol=1e-4)))
leftBound2    = x[dropoffLeft2]
rightBound2   = x[dropoffRight2]

# Compute bandwidth
bandwidth  =  rightBound - leftBound
bandwidth2 =  rightBound2 - leftBound2
assert bandwidth >= 0, "Negative bandwidth: Deltas must be in crescent order."

xBW = x[dropoffLeft:dropoffRight]
yBW = y[dropoffLeft:dropoffRight]

xBW2 = x[dropoffLeft2:dropoffRight2]
yBW2 = y[dropoffLeft2:dropoffRight2]

maxIndex = np.argmax(yBW)
minIndex = np.argmin(yBW)
plt.plot(xBW[maxIndex], yBW[maxIndex], 'rx', label="Pontos de máx e mín 80%")
plt.plot(xBW[minIndex], yBW[minIndex], 'rx')

maxIndex2 = np.argmax(yBW2)
minIndex2 = np.argmin(yBW2)
plt.plot(xBW2[maxIndex2], yBW2[maxIndex2], 'gx', label="Pontos de máx e mín Delta")
plt.plot(xBW2[minIndex2], yBW2[minIndex2], 'gx')

plt.axvline(x=leftBound,  color='r', label='Limites de Banda 80%')
plt.axvline(x=rightBound, color='r')
plt.axvline(x=leftBound2, color='k', label='Limites de Banda Delta')
plt.axvline(x=rightBound2, color='k')


# Compute ripple
ripple = np.max(yBW) - np.min(yBW)
ripple2 = np.max(yBW2) - np.min(yBW2)

print("Bandwidth 80%: {:2e}".format(bandwidth))
print("Bandwidth Delta: {:2e}".format(bandwidth))
print("Ripple 80%: ", ripple)
print("Approx error for bandwidth: {:2e}".format(bandwidth - bandwidth2))
print("Approx error for ripple: {:2e}".format(ripple - ripple2))


# print("\nFunction comparison")
# bandwidth  = cost_function_exact(delta)[2]
# bandwidth2 = cost_function(delta)[2]
#
# ripple    = cost_function_exact(delta)[1]
# ripple2   = cost_function(delta)[1]
# print("Approx error for bandwidth: {:2e}".format(bandwidth - bandwidth2))
# print("Approx error for ripple: {:2e}".format(ripple - ripple2))
# print("Approx error for Cost function: {:2e}".format(cost_function_exact(delta)[0] - cost_function(delta)[0]))

plt.legend()

plt.savefig(dirs.figures+"response_sum_comparison.png", orientation='portrait',
            bbox_inches='tight')
plt.show()
