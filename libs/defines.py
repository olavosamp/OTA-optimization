SIGNAL_SPAN              = 0.4                               # Response width in mV

RIPPLE_DROPOFF_LEFT      = 0.00794079                        # Dropoff point of 80% of peak,
RIPPLE_DROPOFF_RIGHT     = 0.04170417                        # used to bound ripple calculation

PLOT_POINT_DENSITY       = 1e4                               # Point density used to approximate
                                                             # signal response

NUM_DIFFERENTIAL_PAIRS   = 17                                # Not actually a constant. Number
                                                             # of differential pairs used

DISTANCE_TO_MAX          = 0.02618261826182619               # Distance from delta to max index,
                                                             # defined as d = max(x) - delta
MAX_BW_VALUE             = 0.9                               # Maximum absolute bandwidth value
MIN_BW_VALUE             = 0.7                               # Minimum absolute bandwidth value
MIN_DELTA_DIFF_VALUE     = MIN_BW_VALUE/(NUM_DIFFERENTIAL_PAIRS-1)   # Minimum value for deltaDiff[1:]
MAX_DELTA_DIFF_VALUE     = MAX_BW_VALUE/(NUM_DIFFERENTIAL_PAIRS-1)                           # Maximum value for deltaDiff[1:]
