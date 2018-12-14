SIGNAL_SPAN              = 0.4                  # Response width in mV

RIPPLE_DROPOFF_LEFT      = 0.00794079           # Dropoff point of 80% of peak,
RIPPLE_DROPOFF_RIGHT     = 0.04170417           # used to bound ripple calculation

PLOT_POINT_DENSITY       = 2.5e4                # Point density used to approximate
                                                # signal response

NUM_DIFFERENTIAL_PAIRS   = 14                   # Not actually a constant. Number
                                                # of differential pairs used

DISTANCE_TO_MAX          = 0.02618261826182619  # Distance from delta to max index,
                                                # defined as d = max(x) - delta
