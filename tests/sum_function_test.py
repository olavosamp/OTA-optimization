import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function         import sum_function
import libs.defines             as defs
import libs.dirs                as dirs

M     = 12               # Number of transistor responses
span  = 0.08             # Non-zero response width
edge  = (M-1)/2*span
delta = np.linspace(-edge,edge, num=M)

lowerBound = delta[0]  -0.5
upperBound = delta[-1] +0.5
pointDensity = 1e4/(0.4)


x = np.linspace(lowerBound, upperBound, num=round(pointDensity*span))

y = sum_function(x, delta, M=M)

print("x ", x.shape)
print("y ", y.shape)

fig = plt.figure(figsize=(20,20))
plt.plot(x,y)
plt.xlim(-edge-span*2, +edge+span*2)

plt.xlabel("Tensão (V)")
plt.ylabel("Transcondutância (gm)")
# plt.show()

plt.savefig(dirs.figures+"response_sum.png")


# Obtain parameters
maxVal    = np.max(y)
dropIndex = np.argwhere(y > 0.2*maxVal)

bandwidth = np.squeeze(x[dropIndex[-1]] - x[dropIndex[0]])
yBW = y[dropIndex]
ripple = np.max(yBW) - np.min(yBW)

print("Bandwidth: ", bandwidth)
print("Ripple: ", ripple)
