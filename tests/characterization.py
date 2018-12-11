import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function    import transistor_response

numPoints = 1e4
delta = 0
x = np.linspace(-0.2+delta, 0.2+delta, num=numPoints)
y = transistor_response(x, -delta)

data = {'x': x,
        'y': y}

dataDf = pd.DataFrame(data)
maxVal = dataDf.max()['y']

# Filter very small values
threshold = maxVal*0.001
newYIndex = np.argwhere(y >= threshold)
y = np.squeeze(y[newYIndex])
x = np.squeeze(x[newYIndex])

# Create a new DataFrame with filtered x and y
data = {'x': x,
        'y': y}

dataDf = pd.DataFrame(data)

xMaxValIndex  = np.argmax(y)
print(dataDf['y'].describe())

mean = dataDf.mean(axis=0)['y']
print(mean)
# input()
xMeanIndex = np.squeeze(np.argwhere(np.isclose(y, mean, atol=1e-10)))

span = np.abs(x[-1] - x[0])
# print(y)
# print(xMeanIndex)
# exit()
print(mean)
# print(maxVal)

print("Response Span: ", span)

# Plot response
plt.plot(x, y, 'b')

# Plot a vertical line through maxVal
plt.axvline(x=x[xMaxValIndex], color='r')

# Plot horizontal line on mean value
plt.axhline(y=mean, color='g')

plt.show()
