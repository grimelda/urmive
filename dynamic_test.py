import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import stockflow as sf
from scipy.optimize import curve_fit

#%%
startmodel = 1999
endmodel = 2051
x = np.linspace(startmodel, endmodel, (10*(endmodel-startmodel))+1)

#y = sf.LogisticSignal(x)
y = sf.FlatSignal(x, step=True)


AvgLs = 15*np.ones(len(x))

#%%

IOS, dt = sf.InOutStock(\
                        x,
                        y,
                        AvgLs,
                        scaleflow = 'year', # either 'year' or 'dt'
                        shape = 3, # shape for weibull distribution
#                        dm=40,
#                        lm=1.44, # 1.6 44 8
                        )

#%%

sf.PlotResponse(IOS, y, figs=True)
sf.PlotHistograms(IOS, y, figs=True)

#%%

#lscor = sf.LifespanCorrection(x, y, log=True)


