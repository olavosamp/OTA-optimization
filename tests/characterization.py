import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function    import transistor_response
import libs.defines             as defs
import libs.dirs                as dirs


numPoints = 1e4
delta = -0.025
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

print("\nResponse Span: {:5.2e} V".format( span))

# Plot response
fig = plt.figure(figsize=(20,12))
plt.plot(x, y, 'b', label='Sinal')
plt.title("Resposta do Par Diferencial com atraso {:.2e}".format(delta))

# Plot a vertical line through maxVal
plt.axvline(x=x[xMaxValIndex], color='r', label='Ponto Máximo')

# Plot horizontal line on mean value
plt.axhline(y=mean, color='g', label='Valor Médio')


# Filter Bandwidth
threshold = maxVal*0.2
bwIndex = np.argwhere(y >= threshold)

bandwidth = np.squeeze(x[bwIndex[-1]] - x[bwIndex[0]])

# Plot bandwidth limits
plt.axvline(x=x[bwIndex[0] ], color='k', label='Limites de Banda')
plt.axvline(x=x[bwIndex[-1]], color='k')
plt.legend()

# Annotations
plt.annotate('{:.2e}'.format(np.squeeze(y[xMaxValIndex])), xy=(x[xMaxValIndex], y[xMaxValIndex]),
              xytext=(2,4), textcoords='offset points')
plt.annotate('{:.2e}'.format(np.squeeze(y[xMeanIndex[0]])), xy=(x[xMeanIndex[0]], y[xMeanIndex[0]]),
              xytext=(100,4), textcoords='offset points')
plt.annotate('{:.2e}'.format(np.squeeze(y[bwIndex[0]])), xy=(x[bwIndex[0]], y[bwIndex[0]]),
              xytext=(-48,0), textcoords='offset points')
plt.annotate('{:.2e}'.format(np.squeeze(y[bwIndex[-1]])), xy=(x[bwIndex[-1]], y[bwIndex[-1]]),
              xytext=(2,0), textcoords='offset points')

print("Bandwidth: {:12.2e} V".format( bandwidth))

plt.savefig(dirs.figures+"response_characterization.png", orientation='portrait',
            bbox_inches='tight')
plt.show()
