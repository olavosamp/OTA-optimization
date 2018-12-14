import numpy as np
import libs.defines             as defs
import matplotlib.pyplot as plt


def differential_pair_response(x, delta):
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
        response[:,i] = differential_pair_response(x, delta[i])

    return np.sum(response, axis=1)

def cost_function_exact(delta):
    M         = defs.NUM_DIFFERENTIAL_PAIRS
    leftEdge  = delta[0]-0.5
    rightEdge = delta[-1]+0.5
    span      = rightEdge - leftEdge

    x = np.linspace(leftEdge, rightEdge, num=round(defs.PLOT_POINT_DENSITY*span))
    y = sum_function(x, delta, M=M)

    dropoff       = 0.8*np.max(y)
    dropoffIndex  = np.squeeze(np.argwhere(np.isclose(y, dropoff , atol=1e-9)))
    dropoffLeft   = dropoffIndex[0]
    dropoffRight  = dropoffIndex[-1]

    leftBound     = np.squeeze(x[dropoffIndex])[0]
    rightBound    = np.squeeze(x[dropoffIndex])[-1]

    x = x[dropoffLeft:dropoffRight]
    y = y[dropoffLeft:dropoffRight]

    # Compute bandwidth
    bandwidth  =  rightBound - leftBound
    assert bandwidth >= 0, "Negative bandwidth: Deltas must be in crescent order."

    # Compute ripple
    ripple = np.max(y) - np.min(y)

    return ripple - bandwidth, ripple, bandwidth


def cost_function(delta):
    M          = defs.NUM_DIFFERENTIAL_PAIRS
    leftEdge   = delta[0]-0.5
    rightEdge  = delta[-1]+0.5
    span       = rightEdge - leftEdge

    x = np.linspace(leftEdge, rightEdge, num=round(defs.PLOT_POINT_DENSITY*span))
    y = sum_function(x, delta, M=M)

    dropoffLeft  = np.squeeze(np.argwhere(np.isclose(x, defs.DISTANCE_TO_MAX+ delta[0],
                              atol=1e-5)))
    dropoffRight = np.squeeze(np.argwhere(np.isclose(x, defs.DISTANCE_TO_MAX+ delta[-1],
                              atol=1e-5)))
    leftBound   = x[dropoffLeft]
    rightBound  = x[dropoffRight]

    x = x[dropoffLeft:dropoffRight]
    y = y[dropoffLeft:dropoffRight]

    # maxIndex = np.argmax(y)
    # minIndex = np.argmin(y)
    # plt.plot(x[maxIndex], y[maxIndex], 'rx')
    # plt.plot(x[minIndex], y[minIndex], 'r*')
    # plt.plot(x, y)
    #
    # plt.axvline(x=leftBound, color='k', label='Limites de Banda Delta')
    # plt.axvline(x=rightBound, color='k')
    # plt.axvline(x=leftBound,  color='r', label='Limites de Banda 80%')
    # plt.axvline(x=rightBound, color='r')
    # plt.show()

    # Compute ripple
    ripple = np.max(y) - np.min(y)

    # Compute bandwidth
    bandwidth  =  rightBound - leftBound
    assert bandwidth >= 0, "Negative bandwidth: Deltas must be in crescent order."

    return ripple - bandwidth, ripple, bandwidth
