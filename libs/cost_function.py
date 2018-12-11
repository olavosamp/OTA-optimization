import numpy as np

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
