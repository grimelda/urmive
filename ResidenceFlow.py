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

y = sf.LogisticSignal(x)
y = sf.FlatSignal(x, step=True)


AvgLt = 13*np.ones(len(x))

#%%

IOS, dt = sf.InOutStock(\
                        x,
                        y,
                        AvgLt,
                        scaleflow = 'year', # either 'year' or 'dt'
                        shape = 5.5, # shape for weibull distribution
                        LtCorr = 1,#/0.923, # scales outflow to match expected outflow
                        )

#%%

sf.PlotResponse(IOS, y)
sf.PlotHistograms(IOS, y)

#%%

ltcor = sf.LifeTimeCorrection(x, y, log=False)


