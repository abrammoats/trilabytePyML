#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#

from statistics import mean
from statistics import stdev
import math
import numpy as np

def calcMAPE(x, y):
    pes = []
    
    for idx in range(len(x)):
        if not(math.isnan(x[idx])) and not(math.isnan(y[idx])) and y[idx] != 0.0:
            pe = math.fabs((x[idx] - y[idx]) / y[idx] * 100)
            pes.append(pe)

    return mean(pes)

def calcPredictionInterval(x):
    x = list(filter(lambda f: ~np.isnan(f), x)) 
    return 1.96 * stdev(x)