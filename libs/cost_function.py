import numpy as np
import libs.defines             as defs
import matplotlib.pyplot as plt

def transistor_response(x, delta):
    Ib = 1e-9
    n = 1.3
    phiT = 26*1e-3

    arg1 = np.exp((x + delta)/(n*phiT))
    arg2 = np.exp(((x + delta)/(n*phiT))**2)

    return (Ib/(n*phiT))*arg1/(1 + arg2)

def sum_function(x, delta, M=6):
    assert len(delta) == M, "Delta vector must be of same length as the number \
    of transistor responses"

    response = np.zeros((x.shape[0], M))
    for i in range(M):
        response[:,i] = transistor_response(x, delta[i])

    return np.sum(response, axis=1)

def cost_function(delta):
    M          = defs.NUM_DIFFERENTIAL_PAIRS
    leftBound  = delta[0]-0.5
    rightBound = delta[-1]+0.5
    span       = rightBound - leftBound


    x = np.linspace(leftBound, rightBound, num=round(defs.PLOT_POINT_DENSITY*span))
    y = sum_function(x, delta, M=M)

    dropoff       = 0.8*np.max(y)
    dropoffIndex  = np.squeeze(np.argwhere(np.isclose(y, dropoff , atol=1e-9)))
    leftBound     = np.squeeze(x[dropoffIndex])[0]
    rightBound    = np.squeeze(x[dropoffIndex])[-1]

    # Compute bandwidth
    bandwidth  =  rightBound - leftBound
    assert bandwidth >= 0, "Negative bandwidth: Deltas must be in crescent order."

    x = x[dropoffIndex[0]:dropoffIndex[-1]]
    y = y[dropoffIndex[0]:dropoffIndex[-1]]

    maxIndex = np.argmax(y)
    minIndex = np.argmin(y)
    plt.plot(x[maxIndex], y[maxIndex], 'rx')
    plt.plot(x[minIndex], y[minIndex], 'r*')
    # fig = plt.figure()
    plt.plot(x, y)
    print(y[maxIndex] == np.max(y))

    # plt.axvline(x=delta[0]  -defs.RIPPLE_DROPOFF_LEFT, color='k', label='Limites de Banda')
    # plt.axvline(x=delta[-1] +defs.RIPPLE_DROPOFF_RIGHT, color='k')
    plt.axvline(x=leftBound,  color='r', label='Limites de Banda')
    plt.axvline(x=rightBound, color='r')
    plt.show()

    # Compute ripple
    ripple = np.max(y) - np.min(y)
    print("Bandwidth: ", bandwidth)
    print("Ripple: ", ripple)

    return ripple - bandwidth
