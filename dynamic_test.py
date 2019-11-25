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
#x = np.linspace(2000,2019, 19+1)
#y = np.array([413985,437778,460818,494250,516395,536931,552949,567911,585204,605604,623442,636199,646995,653245,653991,652336,652544,655991,661639,665880])

x = np.linspace(startmodel, endmodel, (1*(endmodel-startmodel))+1)

#y = sf.LogisticSignal(x)
y = sf.FlatSignal(x, step=True)


AvgLs = 21*np.ones(len(x))

#%%

IOS, dt = sf.InOutStock(\
                        x,
                        y,
                        AvgLs,
                        scaleflow = 'dt', # either 'year' or 'dt'
                        shape = 2, # shape for weibull distribution
#                        dm=40,
#                        lm=1.44, # 1.6 44 8
                        )

#%%

sf.PlotResponse(IOS, y, figs=True)
sf.PlotHistograms(IOS, x, y, figs=True, sel=[ .9])

#%%

#lscor = sf.LifespanCorrection(x, y, log=True)


