import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import stockflow as sf

#%%
startmodel = 1999
endmodel = 2051
#x = np.linspace(2000,2019, 19+1)
#y = np.array([413985,437778,460818,494250,516395,536931,552949,567911,585204,605604,623442,636199,646995,653245,653991,652336,652544,655991,661639,665880])

x = np.linspace(startmodel, endmodel, (1*(endmodel-startmodel))+1)

#y = sf.LogisticSignal(x)
y = sf.FlatSignal(x, step=-.2)


AvgLs = 15*np.ones(len(x))

#%%

IOS, dt = sf.InOutStock(\
                        x,
                        y,
                        AvgLs,
                        scaleflow = 'dt', # either 'year' or 'dt'
                        shape = 7, # shape for weibull distribution
#                        dm=1,
#                        lm=1,
                        dtype='jz'
                        )

#%%

sf.PlotResponse(IOS, y, figs=True)
sf.PlotHistograms(IOS, x, y, dt, figs=True, sel=[ 0, .1/5, 3.5/5])

#%%

#lscor = sf.LifespanCorrection(x, y, log=True)


