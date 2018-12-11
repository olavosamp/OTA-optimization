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
# print(delta.shape)
# exit()

# M=5
# start = -(M-1)/2*span
# ---|---|---0---|---|---
# M=6
# start = -(M-1)/2*span
# ---|---|---|-0-|---|---|---

lowerBound = delta[0]  -0.5
upperBound = delta[-1] +0.5
pointDensity = 1e4/(0.4)


x = np.linspace(lowerBound, upperBound, num=pointDensity*span)

y = sum_function(x, delta, M=M)

print(x.shape)
print(y.shape)
fig = plt.plot(x,y)
plt.xlim(-edge-span*2, +edge+span*2)

plt.xlabel("V")
plt.ylabel("gm")
# plt.show()

plt.savefig(dirs.figures+"response_sum.png")
