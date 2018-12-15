import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function         import *
import libs.defines             as defs
import libs.dirs                as dirs

M       = defs.NUM_DIFFERENTIAL_PAIRS   # Number of differential pairs
span    = defs.SIGNAL_SPAN              # Non-zero response width
# spacing = 200*span/2
# edge    = (M-1)/2*spacing
# delta   = np.linspace(-edge,edge, num=M)

# deltaDiff = [0.6053946, 0.5, 0.5, 0.5]
deltaDiff = [-1.5]
for i in range(1, M):
    deltaDiff.append(0.5)

delta = convert_delta(deltaDiff)

lowerBound = delta[0]  -2*span
upperBound = delta[-1] +2*span
pointDensity = defs.PLOT_POINT_DENSITY
numPoints = round(pointDensity*(upperBound-lowerBound))

x = np.linspace(lowerBound, upperBound, num=int(1e6))
y = sum_function(x, delta, M=M)

print("x ", x.shape)
print("y ", y.shape)
print(lowerBound)
print(upperBound)

fig = plt.figure(figsize=(30,16))
plt.plot(x,y)
plt.xlim(lowerBound-span, upperBound+span)

plt.xlabel("Tensão (V)")
plt.ylabel("Transcondutância (gm)")
plt.show()

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
