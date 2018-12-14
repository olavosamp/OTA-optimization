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
pointDensity = 1e4/(0.4)

x = np.linspace(lowerBound, upperBound, num=round(pointDensity*span))
y = sum_function(x, delta, M=M)

print("x ", x.shape)
print("y ", y.shape)

fig = plt.figure(figsize=(30,16))
plt.plot(x,y)
plt.xlim(-edge-span*2, +edge+span*2)

plt.xlabel("Tensão (V)")
plt.ylabel("Transcondutância (gm)")

# Obtain Cost function parameters
maxVal    = np.max(y)
dropIndex = np.argwhere(y > 0.8*maxVal)
bandwidth = np.squeeze(x[dropIndex[-1]] - x[dropIndex[0]])
yBW = y[np.squeeze(dropIndex[0]):np.squeeze(dropIndex[-1])]
xBW = x[np.squeeze(dropIndex[0]):np.squeeze(dropIndex[-1])]

# # Plot bandwidth limited signal
# plt.plot(xBW,yBW)

for deltai in delta:
    plt.plot(x, differential_pair_response(x, deltai))

ripple = np.max(yBW) - np.min(yBW)

print("Limit Left: ", x[dropIndex[0]])
print("Limit Right: ", x[dropIndex[-1]])

print("Limit Left  Defs: ", defs.RIPPLE_DROPOFF_LEFT)
print("Limit Right Defs: ", defs.RIPPLE_DROPOFF_RIGHT)
print("")

print("Bandwidth: ", bandwidth)
print("Ripple: ", ripple)
print("f_0(delta) = ", ripple - bandwidth)

print("Encapsulated cost function")
print("Cost function: ", cost_function(delta))

plt.savefig(dirs.figures+"response_sum.png", orientation='portrait',
            bbox_inches='tight')
