import numpy as np
import libs.defines             as defs
import matplotlib.pyplot as plt


def differential_pair_response(x, delta):
    Ib   = 1e-9
    n    = 1.3
    phiT = 26*1e-3

    response = np.zeros(np.shape(x)[0])
    index = (x + delta <= 10*defs.SIGNAL_SPAN)

    # Hopefully numerically stable expression
    arg1 = 1
    w = (x[index] + delta)/(n*phiT)
    arg2 = np.exp(-w) + np.exp(w**2 -w)

    # Original expression
    # arg1 = np.exp((x[index] + delta)/(n*phiT))
    # arg2 = np.exp(((x[index] + delta)/(n*phiT))**2)

    response[index] = (Ib/(n*phiT))*arg1/(1 + arg2)

    # if np.isnan(response).any():
        # print("Delta causing error: ", delta)
        # print(x[np.isnan(response)] + delta)
        # print(x[np.isnan(response)][10000] + delta)
        # input()
        # return
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


def cost_function(deltaDiff):
    delta = convert_delta(deltaDiff)

    M          = defs.NUM_DIFFERENTIAL_PAIRS
    leftEdge   = delta[0]-0.5
    rightEdge  = delta[-1]+0.5
    span       = rightEdge - leftEdge

    x = np.linspace(leftEdge, rightEdge, num=round(defs.PLOT_POINT_DENSITY*span))
    y = sum_function(x, delta, M=M)

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

    # xBW = np.zeros(np.shape(x[dropoffLeft:dropoffRight]))
    # xBW = x[dropoffLeft:dropoffRight]
    yBW = np.zeros(np.shape(y[dropoffLeft:dropoffRight]))
    yBW = y[dropoffLeft:dropoffRight]

    fig = plt.figure(figsize=(20,10))
    plt.plot(x,y)
    print("\nResponses")
    for deltai in delta:
        resp = differential_pair_response(x, deltai)
        print(deltai)
        print(resp)
        print(np.sum(resp))
        plt.plot(x, resp)

    # maxIndex = np.argmax(y)
    # minIndex = np.argmin(y)
    # plt.plot(x[maxIndex], y[maxIndex], 'rx')
    # plt.plot(x[minIndex], y[minIndex], 'r*')
    # plt.plot(x, y)

    plt.axvline(x=leftBound, color='k', label='Limites de Banda Delta')
    plt.axvline(x=rightBound, color='k')
    # plt.axvline(x=leftBound,  color='r', label='Limites de Banda 80%')
    # plt.axvline(x=rightBound, color='r')
    # plt.show()

    # Compute ripple
    ripple = np.max(yBW) - np.min(yBW)

    # Compute bandwidth
    bandwidth  =  rightBound - leftBound
    assert bandwidth >= 0, "Negative bandwidth: Deltas must be in crescent order."
    # print(x)
    # print(y)
    print("\nRegular")
    print("bandwidth: ", bandwidth)
    print("ripple: ", ripple)
    # input()

    del x, y, yBW
    return 1e8*ripple - bandwidth


def get_dropoff_points(x, delta):
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

    return dropoffLeft, dropoffRight


def get_xy(delta):
    M       = np.shape(delta)[0]   # Number of differential pairs
    span    = defs.SIGNAL_SPAN              # Non-zero response width

    lowerEdge   = delta[0]  -2*span
    upperEdge   = delta[-1] +2*span
    pointDensity = defs.PLOT_POINT_DENSITY
    numPoints    = int(np.clip(pointDensity*(upperEdge-lowerEdge), 2**20+1, 2**25+1))

    x, step   = np.linspace(lowerEdge, upperEdge, num=numPoints, retstep=True)
    y = np.zeros(np.shape(x))

    # Compute differential pair response centered on zero
    yZero = differential_pair_response(x, 0)

    # Rotate response back to center it on lowerBound
    correctIndex = int(round((0+lowerEdge)/step))
    yZero = np.roll(yZero, correctIndex)

    # Compute responses centered on each delta_i
    indexDiff = np.zeros(M)
    for i in range(M):
        if i == 0:
            indexDiff[i] = (delta[i] - lowerEdge)
        else:
            indexDiff[i] = delta[i] - delta[i-1]
        indexDiff[i] = round(int((indexDiff[i])/step))

        yComp = np.roll(yZero, int(sum(indexDiff[:i+1])))
        y += yComp
    return x, y


def cost_function_alt(deltaDiff):
    delta = convert_delta(deltaDiff)

    # Get x, y values
    x, y = get_xy(delta)

    # Choose dropoff points as delta[0] and delta[-1]
    dropoffLeft, dropoffRight = get_dropoff_points(x, delta)

    leftBound  = x[dropoffLeft]
    rightBound = x[dropoffRight]

    # xBW = np.zeros(np.shape(x[dropoffLeft:dropoffRight]))
    # xBW = x[dropoffLeft:dropoffRight]
    yBW = np.zeros(np.shape(y[dropoffLeft:dropoffRight]))
    yBW = y[dropoffLeft:dropoffRight]

    bandwidth     = delta[-1] - delta[0]

    if np.max(yBW) == 0:
        return np.inf
    ripplePercent = (np.max(yBW) - np.min(yBW))/np.max(yBW)

    plt.plot(x, y)
    # print("\nAlt")
    print("bandwidth:  ", bandwidth)
    # print("bandwidth2: ", delta[-1] - delta[0])
    print("ripple: ", ripplePercent)

    del x, y, yBW, leftBound, rightBound, dropoffLeft, dropoffRight

    return ripplePercent - bandwidth


def get_ripple_percent(deltaDiff):
    delta = convert_delta(deltaDiff)

    # Get x, y values
    x, y = get_xy(delta)

    # Choose dropoff points as delta[0] and delta[-1]
    dropoffLeft, dropoffRight = get_dropoff_points(x, delta)

    yBW = np.zeros(np.shape(y[dropoffLeft:dropoffRight]))
    yBW = y[dropoffLeft:dropoffRight]

    if np.max(yBW) == 0:
        return np.inf

    ripplePercent = (np.max(yBW) - np.min(yBW))/np.max(yBW)
    return ripplePercent


def get_bandwidth(deltaDiff):
    delta = convert_delta(deltaDiff)
    bandwidth = delta[-1] - delta[0]
    assert bandwidth >= 0, "Negative bandwidth: Deltas must be in crescent order."

    return bandwidth


def convert_delta(deltaDiff):
    deltaDiff[1:] = np.clip(deltaDiff[1:], defs.MIN_DELTA_DIFF_VALUE, None)

    deltaLen = len(deltaDiff)
    delta = np.zeros(deltaLen)
    delta[0] = deltaDiff[0]

    for i in range(1, deltaLen):
        delta[i] = delta[i-1] + np.abs(deltaDiff[i])

    return delta
