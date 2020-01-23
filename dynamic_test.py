import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import stockflow as sf

#%%
startmodel = 1999
endmodel = 2051
#x = np.linspace(1990,2051, 19+1)
#y = np.array([413985,437778,460818,494250,516395,536931,552949,567911,585204,605604,623442,636199,646995,653245,653991,652336,652544,655991,661639,665880])
#y=np.array([50,83,101,131,152,250,296,324,363,410,447,485,672,905,1075,1244,1561,1749,2149,2222,2237,2316,2433,2713,2865,3391,4257,4202,4393,4700,5583,6466,7349,8232,9115,10000,11780,13560,15340,17120,18900,19725,20550,21375,22200,23025,23850,24675,25500,26325,27150,27975,28800,29625,30450,31275,32100,32925,33750,34575])
#y = np.array([3466,3643,3826,4014,4207,4406,4610,4820,5036,5257,5484,5716,5953,6197,6445,6699,6958,7221,7490,7763,8040,8322,8608,8898,9191,9489,9789,10094,10401,10713,11028,11347,11670,11997,12330,12668,13011,13361,13719,14084,14458,14842,15257,15678,16137,16612,17128,17674,18246,18828,19391,19917,20427,20934,21426,21934,22480,23070,23624,24232,24824,25419,25919,26398,26786,27147,27668,28338,29073,29881,30730,31598,32475,33454,34522,35688,36916,38182,39414,40715,42172,43672,45247,46879,48690,50616,52598,54703,56980,59444,62084,64361,66963,70124,72926,75137,77848,80864,83936,87094,89958,92397,94763,97196,99691,102104,105044,107794,110362,113184,115754,117986,120391,122551,124752,126603,128923,131303,134663,138341,140372,143020,145243,147090,148604,150681,152178,154001,156862,159212,161360,162082,163010,163876,164539,164515,164368,164099])

x = np.linspace(startmodel, endmodel, (1*(endmodel-startmodel))+1)

#y = sf.LogisticSignal(x)
y = sf.FlatSignal(x, step=-.15, restore=False)
#
#AvgLs = 10 * ((0.0164 * x) - 31.8)
AvgLs = 10*np.ones(len(x))

#%%

IOS, dt = sf.InOutStock(\
                        x,
                        y,
                        AvgLs,
                        scaleflow = 'dt', # either 'year' or 'dt'
                        shape = 5, # shape for weibull distribution
#                        dm=1,
#                        lm=1,
                        dtype='jz'
                        )

#%%

#dt = (max(x)-min(x))/len(x[:-1])
#
#IOS = pd.DataFrame(columns=['x', 'Infl_dt', 'Outf_dt', 'Stock', 'Hist', 'Control'])
#IOS['x'] = x
#
#### makes histogram for first timestep from weibull survival function at t0
#IOS.at[0, 'Hist'] = HistWeib_t0(x, dt, shape, AvgLs[0], y[0])
#IOS.at[0, 'Stock'] = sum(IOS.loc[0, 'Hist'])
#
#### calculates inflows, outflows, and stocks for each timestep
#for t in IOS.index[1:]:
#    IOS = InOutFlow_dt(x, y, t, dt, shape, IOS, AvgLs[t], dmulti, lsmulti, dtype)
#
#### builds control curve, use to diagnose errors in mass balance
#IOS = ControlCurve(IOS)
#
#### scales dataframe to fit yearly values or timestep values
#IOS = ScaleFlowsToYear(IOS, dt, scaleflow)
#
#### Notes the shape and lifespan values
#IOS['shape'] = shape
#
#
#
#
#
#%%

sf.PlotResponse(IOS, y, figs=True)
#sf.PlotHistograms(IOS, x, y, dt, figs=True, sel=[ 0, .1/5, 3.5/5])



#lscor = sf.LifespanCorrection(x, y, log=True)

#
