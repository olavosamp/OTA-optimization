import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function         import differential_pair_response
import libs.defines             as defs
import libs.dirs                as dirs

delta = -3
lowerEdge   = delta-2*defs.SIGNAL_SPAN
upperEdge   = delta+2*defs.SIGNAL_SPAN
pointDensity = defs.PLOT_POINT_DENSITY
numPoints    = int(np.clip(pointDensity*(upperEdge-lowerEdge), 2**20+1, 2**25+1))

x, step   = np.linspace(lowerEdge, upperEdge, num=numPoints, retstep=True)
y = differential_pair_response(x, delta)

plt.plot(x,y)
plt.show()
