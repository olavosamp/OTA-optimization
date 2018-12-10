import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from functions.cost_function    import transistor_response

x = np.linspace(-0.2, 0.2, num=1e4)
y = transistor_response(x, 0)

dataY = pd.DataFrame(y)
print(dataY.describe())

mean = dataY.mean(axis=0)
max = dataY.max()

print(mean)
print(max)
# exit()
#
# plt.plot(x, y)
plt.show()
