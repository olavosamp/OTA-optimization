import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from libs.cost_function    import transistor_response
import libs.defines             as defs
import libs.dirs                as dirs


numPoints = 1e4
# delta = -0.025
delta = 0
x = np.linspace(-0.2+delta, 0.2+delta, num=numPoints)
y = transistor_response(x, -delta)

data = {'x': x,
        'y': y}

dataDf = pd.DataFrame(data)
maxVal = dataDf.max()['y']

# Filter very small values
thresholdBW = maxVal*0.001
newYIndex = np.argwhere(y >= thresholdBW)
y = np.squeeze(y[newYIndex])
x = np.squeeze(x[newYIndex])

# Create a new DataFrame with filtered x and y
data = {'x': x,
        'y': y}

dataDf = pd.DataFrame(data)

xMaxValIndex  = np.argmax(y)
print("\n", dataDf['y'].describe())

mean = dataDf.mean(axis=0)['y']

xMeanIndex = np.squeeze(np.argwhere(np.isclose(y, mean, atol=1e-10)))

# Filter Bandwidth
thresholdBW = 0.8*maxVal
bwIndex = np.argwhere(y >= thresholdBW)

span = np.abs(x[-1] - x[0])
print("\nResponse Span 0.1%: {:5.2e} V".format( span))

bandwidth = np.squeeze(x[bwIndex[-1]] - x[bwIndex[0]])
print("Bandwidth: {:12.2e} V".format( bandwidth))


# Plot response
fig = plt.figure(figsize=(20,12))
plt.plot(x, y, 'b', label='Sinal')
plt.title("Resposta do Par Diferencial com atraso {:.2e}".format(delta))
plt.xlabel("Tensão (V)")
plt.ylabel("Transcondutância (gm)")


# Compute ripple limit points
dropoff = thresholdBW
dropoffIndex  = np.argwhere(np.isclose(y, dropoff , atol=1e-11), )
dropoffLeft   = x[dropoffIndex[0]]
dropoffRight  = x[dropoffIndex[-1]]

print("Ripple dropoff Left: ", dropoffLeft[0])
print("Ripple dropoff Right: ", dropoffRight[0])

plt.plot(x[dropoffIndex], y[dropoffIndex], 'rx', label='Pontos de queda do ripple')

# Plot a vertical line through maxVal
plt.axvline(x=x[xMaxValIndex], color='r', label='Ponto Máximo')

# Plot horizontal line over mean value
plt.axhline(y=mean, color='g', label='Valor Médio')

# Plot bandwidth limits
plt.axvline(x=x[dropoffIndex[0] ], color='k', label='Limites de Banda')
plt.axvline(x=x[dropoffIndex[-1]], color='k')
plt.legend()

# Annotations
# Max value point
plt.annotate('{:.2e}'.format(np.squeeze(y[xMaxValIndex])), xy=(x[xMaxValIndex], y[xMaxValIndex]),
              xytext=(2,4), textcoords='offset points')
# Mean value line
plt.annotate('{:.2e}'.format(np.squeeze(y[xMeanIndex[0]])), xy=(x[xMeanIndex[0]], y[xMeanIndex[0]]),
              xytext=(100,4), textcoords='offset points')
# Left ripple point
plt.annotate('x = {:.2e}'.format(np.squeeze(x[bwIndex[0]])), xy=(x[bwIndex[0]], y[bwIndex[0]]),
              xytext=(-68,0), textcoords='offset points')
# Right ripple point
plt.annotate('x = {:.2e}'.format(np.squeeze(x[bwIndex[-1]])), xy=(x[bwIndex[-1]], y[bwIndex[-1]]),
              xytext=(4,0), textcoords='offset points')


plt.savefig(dirs.figures+"response_characterization.png", orientation='portrait',
            bbox_inches='tight')
plt.show()
