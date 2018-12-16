import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt

from libs.cost_function     import *
import libs.defines         as defs
import libs.dirs            as dirs


def plot_results(deltaDiff):
    delta = convert_delta(deltaDiff)
    x, y = get_xy(delta)

    dropoffLeft, dropoffRight = get_dropoff_points(x, delta)

    xBW = x[dropoffLeft:dropoffRight]
    yBW = y[dropoffLeft:dropoffRight]
    rippleMean = np.mean(yBW)

    rippleIndex = np.argmax(np.abs(yBW - rippleMean))

    fig = plt.figure(figsize=(10,5))

    plt.plot(x, y)
    plt.xlabel("Tensão (V)")
    plt.ylabel("Corrente (A)")
    plt.xlim(x[dropoffLeft]-0.2, x[dropoffRight]+0.2)

    plt.plot(x[rippleIndex],y[rippleIndex], 'rx')
    plt.title("Resultado da Otimização")

    # Plot horizontal line over mean value
    plt.axhline(y=rippleMean, color='g', label='Valor Médio da Banda')

    # Plot bandwidth limits
    plt.axvline(x=x[dropoffRight], color='k', label='Limites de Banda')
    plt.axvline(x=x[dropoffLeft ], color='k')
    plt.legend()

    plt.savefig(dirs.figures+"result_"+defs.NUM_DIFFERENTIAL_PAIRS+".png", orientation='portrait',
                bbox_inches='tight')
    plt.savefig(dirs.figures+"result_"+defs.NUM_DIFFERENTIAL_PAIRS+".eps", orientation='portrait',
                bbox_inches='tight')
    plt.show()


def save_results(deltaDiff, fevals, optimizer='COBYLA'):
    M = defs.NUM_DIFFERENTIAL_PAIRS
    delta = convert_delta(deltaDiff)

    ripple = get_ripple_percent(deltaDiff)
    bandwith = get_bandwidth(deltaDiff)

    fOpt = cost_function_alt(deltaDiff)
    data = {'Delta': delta,
            'M': M,
            'Otimizador': optimizer,
            'Banda': bandwith*defs.MAX_BW_VALUE,
            'Ripple': ripple,
            'FEvals': fevals,
            'f(x)': fOpt,
            }

    df = pd.DataFrame(data)
    df.to_excel(dirs.results+"results_"+optimizer+".xlsx")
