import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt
import scipy.integrate          as spi

from libs.cost_function         import *
import libs.defines             as defs
import libs.dirs                as dirs

M       = defs.NUM_DIFFERENTIAL_PAIRS   # Number of differential pairs
span    = defs.SIGNAL_SPAN              # Non-zero response width

# deltaDiff = [-1.,   0.05099223,  0.04889312,  0.05028007,  0.05155574,  0.04952145,
#   0.04975524,  0.05064246 , 0.0493633,   0.05008275,  0.05013026,  0.04869392,
#   0.04921992,  0.05098836 , 0.05094915,  0.05021183,  0.04942693]
deltaDiff = [-1.]
for i in range(1, M):
    deltaDiff.append(defs.MIN_DELTA_DIFF_VALUE)
    # deltaDiff.append(0)

resultAlt = cost_function_alt(deltaDiff)
print("ripple F: ", get_ripple_percent(deltaDiff))
print("bandwidth F: ", get_bandwidth(deltaDiff))
print(resultAlt)

delta = convert_delta(deltaDiff)
x, y = get_xy(delta)
# plt.plot(x, y)
# plt.show()
# exit()

dropoffLeft, dropoffRight = get_dropoff_points(x, delta)


fig = plt.figure(figsize=(20,10))
plt.plot(x,y)
plt.xlim(x[dropoffLeft]-0.2, x[dropoffRight]+0.2)

plt.xlabel("Tensão (V)")
plt.ylabel("Corrente (A)")


# Plot bandwidth limits
plt.axvline(x=x[dropoffRight], color='k', label='Limites de Banda')
plt.axvline(x=x[dropoffLeft ], color='k')
plt.legend()

# for deltai in delta:
#     plt.plot(x, differential_pair_response(x, deltai))

plt.show()
exit()

plt.savefig(dirs.figures+"response_sum.png", orientation='portrait',
            bbox_inches='tight')
