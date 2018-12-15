import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt
import scipy.integrate          as spi

from libs.cost_function         import *
import libs.defines             as defs
import libs.dirs                as dirs

M       = defs.NUM_DIFFERENTIAL_PAIRS   # Number of differential pairs
span    = defs.SIGNAL_SPAN              # Non-zero response width

# deltaDiff = [0.6053946, 0.5, 0.5, 0.5]
deltaDiff = [0.09247358]
for i in range(1, M):
    deltaDiff.append(0.4)
# deltaDiff[-2] = span/2

# result = cost_function_alt(deltaDiff)
result = cost_function(deltaDiff)
print(result)
exit()

delta = convert_delta(deltaDiff)

lowerBound = delta[0]  -2*span
upperBound = delta[-1] +2*span
pointDensity = defs.PLOT_POINT_DENSITY
numPoints = int(np.clip(pointDensity*(upperBound-lowerBound), 2**20+1, 2**25+1))

x, step   = np.linspace(lowerBound, upperBound, num=numPoints, retstep=True)
y = np.zeros(np.shape(x))

# Compute differential pair response centered on zero
yZero = differential_pair_response(x, 0)
print("\nNum Points: ", numPoints)
print("Base: {:.4e}".format(spi.romb(yZero, dx=step)))

# Rotate response back to center it on lowerBound
correctIndex = int(round((0+lowerBound)/step))
yZero = np.roll(yZero, correctIndex)

print(delta)
# Compute responses centered on each delta_i
indexDiff = np.zeros(M)
for i in range(M):
    if i == 0:
        indexDiff[i] = (delta[i] - lowerBound)
    else:
        indexDiff[i] = delta[i] - delta[i-1]
    indexDiff[i] = round(int((indexDiff[i])/step))

    yComp = np.roll(yZero, int(sum(indexDiff[:i+1])))
    y += yComp

    # print("{:.4e}".format(spi.romb(yComp, dx=step)))
    # plt.plot(x, yComp)
    # plt.show()

plt.plot(x, y)
plt.show()

print("x ", x.shape)
print("y ", y.shape)
print(lowerBound)
print(upperBound)

fig = plt.figure(figsize=(20,10))
plt.plot(x,y)
plt.xlim(lowerBound, upperBound)

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
