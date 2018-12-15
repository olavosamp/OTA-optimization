import numpy as np
import libs.defines             as defs
import matplotlib.pyplot as plt


def differential_pair_response(x, delta):
    Ib = 1e-9
    n = 1.3
    phiT = 26*1e-3

    response = np.zeros(np.shape(x)[0])
    index = (x + delta <= 10*defs.SIGNAL_SPAN)

    arg1 = np.exp((x[index] + delta)/(n*phiT))
    arg2 = np.exp(((x[index] + delta)/(n*phiT))**2)

    response[index] = (Ib/(n*phiT))*arg1/(1 + arg2)

    # arg1 = np.exp((x + delta)/(n*phiT))
    # arg2 = np.exp(((x + delta)/(n*phiT))**2)
    #
    # response = (Ib/(n*phiT))*arg1/(1 + arg2)

    # if np.isnan(response).any():
    #     print("Delta causing error: ", delta)
    #     print(x[np.isnan(response)] + delta)
    #     print(x[np.isnan(response)][10000] + delta)
    #     input()
    return response


def sum_function(x, delta, M=6):
    assert len(delta) == M, "Delta vector must be of same length as the number \
    of differential pair responses"

    response = np.ones((x.shape[0], M))
    for i in range(M):
        response[:,i] = differential_pair_response(x, delta[i])
        # if np.isnan(response[:, i]).any():
        # print("Pair number ", i)
        # print(response[:,i])
        # input()

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

    return ripple - bandwidth


def cost_function(deltaDiffs):
    delta = convert_delta(deltaDiffs)

    M          = defs.NUM_DIFFERENTIAL_PAIRS
    leftEdge   = delta[0]-0.5
    rightEdge  = delta[-1]+0.5
    span       = rightEdge - leftEdge
    # print(delta)
    # print(leftEdge)
    # print(rightEdge)
    # input()

    x = np.linspace(leftEdge, rightEdge, num=round(defs.PLOT_POINT_DENSITY*span))
    y = sum_function(x, delta, M=M)
    # print("X Span:")
    # print(x[0], x[-1])
    # print("Step: {:.2e}".format(x[0]-x[1]))
    # print("Y Sum: ",np.sum(y))
    # input()

    dropoffLeft  = []
    dropoffRight = []
    tol = 1e-11
    while np.shape(dropoffLeft)[0] == 0 or np.shape(dropoffRight)[0] == 0:
        dropoffLeft  = np.argwhere(np.isclose(x, defs.DISTANCE_TO_MAX+ delta[0],
                                  atol=tol))
        dropoffRight = np.argwhere(np.isclose(x, defs.DISTANCE_TO_MAX+ delta[-1],
                                  atol=tol))
        tol = tol*10
        if tol >= 1:
            return np.inf

    dropoffLeft  = np.squeeze(dropoffLeft )
    dropoffRight = np.squeeze(dropoffRight)

    if np.ndim(dropoffLeft) > 0:
        dropoffLeft   = np.squeeze(dropoffLeft)[0]
    else:
        dropoffLeft   = np.squeeze(dropoffLeft)

    if np.ndim(dropoffRight) > 0:
        dropoffRight  = np.squeeze(dropoffRight)[-1]
    else:
        dropoffRight  = np.squeeze(dropoffRight)

    leftBound  = x[dropoffLeft]
    rightBound = x[dropoffRight]

    # fig = plt.figure(figsize=(20,10))
    # plt.plot(x,y)
    # print("\nResponses")
    # for deltai in delta:
    #     resp = differential_pair_response(x, deltai)
    #     print(deltai)
    #     print(resp)
    #     print(np.sum(resp))
    #     plt.plot(x, resp)

    x = x[dropoffLeft:dropoffRight]
    y = y[dropoffLeft:dropoffRight]

    # maxIndex = np.argmax(y)
    # minIndex = np.argmin(y)
    # plt.plot(x[maxIndex], y[maxIndex], 'rx')
    # plt.plot(x[minIndex], y[minIndex], 'r*')
    # plt.plot(x, y)

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
    # print(x)
    # print(y)
    # print(bandwidth)
    # print(ripple)
    # input()
    return 1e8*ripple - bandwidth


def convert_delta(deltaDiffs):
    deltaLen = len(deltaDiffs)
    delta = np.zeros(deltaLen)
    delta[0] = deltaDiffs[0]

    for i in range(1, deltaLen):
        delta[i] = delta[i-1] + np.abs(deltaDiffs[i])
        # print()

    return delta
