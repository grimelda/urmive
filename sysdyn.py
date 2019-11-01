import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.stats



def Dstock_dt(\
              y,
              t,
              hist,
              pdf,
              ):
    
    ### calculates stock change at begin of current time step (after age deaths)
    dstock = y[t] - sum(hist) # scalar
    
    ### scalar for number of births in timestep 
    if dstock > 0: # this represents 'births'
        hist = np.insert(hist, 0, dstock)[:len(hist)] # vector
    if dstock <= 0: # this represents non-age-related deaths
        deaths_ds = (hist*pdf)/sum(hist*pdf) * dstock # absolute negative vector
        hist = hist+deaths_ds # absolute positive vector
        
    return dstock, hist


def WeibDist(x, dt, AvgLt, shape=3, loc = 0):
    ### this makes an appropriate bin size- otherwise with high ages, 
    ### the bin would overflow
    binrange = range(0,int(len(x)*5))
    weib = dict()
    weib['pdf'] = scipy.stats.weibull_min.pdf(binrange, shape, loc, AvgLt)
    weib['cdf'] = scipy.stats.weibull_min.cdf(binrange, shape, loc, AvgLt)
    weib['sf'] = scipy.stats.weibull_min.sf(binrange, shape, loc, AvgLt)

    return weib


def Deaths_dt(hist, pdf):

    ### vector describing deaths per age bin, 7.05 is an optimised parameter    
    deaths_age = hist * pdf * 19#7.05
    ### create new histogram with 
    hist = hist-deaths_age
    deaths_age = sum(deaths_age)
            
    return deaths_age, hist


def HistWeib_t0(x, dt, AvgLt, y0):
    ### vector, based on input weibull survival function
    hist = WeibDist(x, dt, AvgLt, shape=5.5, loc = 0)['sf']
    # normalise
    hist = hist/sum(hist)
    # scale to y0
    hist = y0*hist
    
    return hist


def InOutFlow_dt(\
                 x,
                 y,
                 t,
                 dt,
                 IOS, #hist,
                 AvgLt,
                 ):
    
    ### formulates weilbull distribution
    pdf = WeibDist(x, dt, AvgLt*1.7, shape=5.5, loc = 0)['pdf'] # 1.33/2.7 || 1.5/4 || 2/7.05 || 1.9/6.0

    ### calculates age related deaths during timestep from previous ts histogram
    (deaths_age, 
     hist,
     )= Deaths_dt(IOS.loc[t-1,'Hist'], pdf)
        
    ### calculates stock change during timestep
    (dstock,
     hist,
     ) = Dstock_dt(y, t, hist, pdf)
    
    ### fill IOS dataframe    
    IOS.at[t, 'dstock'] = dstock
    IOS.at[t, 'Hist'] = hist
    IOS.at[t, 'Stock'] = sum(hist)
    if dstock > 0:
        IOS.at[t, 'Infl_dt'] = dstock
        IOS.at[t, 'Outf_dt'] = deaths_age #sum(deaths_age)
    if dstock <= 0:
        IOS.at[t, 'Infl_dt'] = 0
        IOS.at[t, 'Outf_dt'] = deaths_age - dstock # dstock is negative, so adds

    return IOS


def InOutStock(\
               x,
               y,
               AvgLt,
               scaleflow = 'year', # either 'year' or 'dt'
               shape = 3, # shape for weibull distribution
               ):
    
    ### find time step size
    dt = (max(x)-min(x))/len(x[:-1])
    
    ### scale lifetime to match timestep
    LtCorr = 1/0.925#6/5.557
    AvgLt= AvgLt/dt*LtCorr
    
    IOS = pd.DataFrame(columns=['x', 'Infl_dt', 'Outf_dt', 'Stock', 'Hist', 'Control'])
    IOS['x'] = x
    
    ### makes histogram for first timestep from weibull survival function at t0
    IOS.at[0, 'Hist'] = HistWeib_t0(x, dt, AvgLt[0], y[0])
    IOS.at[0, 'Stock'] = sum(IOS.loc[0, 'Hist'])
    
    ### calculates inflows, outflows, and stocks for each timestep
    for t in IOS.index[1:]:
        IOS = InOutFlow_dt(x, y, t, dt, IOS, AvgLt[t])
    
#    ### builds control curve, use to diagnose errors in mass balance
#    IOS = ControlCurve(IOS)
    
    ### scales dataframe to fit yearly values or timestep values
    IOS = ScaleFlowsToYear(IOS, dt, scaleflow)
    
    return IOS, dt


def AvgAge(hist, dt):
    age = []
    for i in range(len(hist)):
        age.append(hist[i]*i)
    AvgAge = sum(age)/sum(hist) * dt
    
    return AvgAge # in years


def ScaleFlowsToYear(IOS, dt, scaleflow):
    if scaleflow == 'dt':
        IOS = IOS.rename(columns={'Infl_dt':'Infl', 'Outf_dt':'Outf'})
        return IOS
    
    if scaleflow == 'year':        
        ## use this if you dont want to scale years later on 
        IOS = IOS.set_index('x', drop=True)
        ### list of int years
        years = list(IOS.index.astype('int').unique())
        for year in years[1:-1]: # first and last years do not count
            IOS.at[year,'Infl'] =  IOS.loc[((year<=IOS.index) & (IOS.index<year+1)),'Infl_dt'].sum()
            IOS.at[year,'Outf'] =  IOS.loc[((year<=IOS.index) & (IOS.index<year+1)),'Outf_dt'].sum()
        IOS = IOS.dropna(subset=['Infl'])
        IOS = IOS.reset_index(drop=False)

#        ## alternative, dirty way            
#        IOS['Infl'] = IOS['Infl_dt'].multiply(1/dt)
#        IOS['Outf'] = IOS['Outf_dt'].multiply(1/dt)

        return IOS
    
    
def ControlCurve(IOS):
    ### builds control curve by adding inflows and subtracting outflows from stock
    IOS.at[0, 'Control'] = IOS.loc[0, 'Stock']
    for i in IOS.index[1:]:
        IOS.at[i,'Control'] = IOS.loc[i-1, 'Control'] + IOS.loc[i, 'Infl_dt'] - IOS.loc[i, 'Outf_dt']

    ### useful histogram with rounded up numbers
    IOS['Hist1'] = [np.around(x) for x in IOS['Hist']]

    return IOS


def LogistiCurve(x, start=0, end=1, steepness=1, midpoint=2030):
    if end < start:
        y = np.ones(len(x))*start - np.ones(len(x))*(start-end) / (1+np.exp(-steepness*(x-midpoint)))
    elif end >= start: 
        y = np.ones(len(x))*start + np.ones(len(x))*(end-start) / (1+np.exp(-steepness*(x-midpoint)))
    return y

def WeibFit(path='cars.csv'):
    data = pd.Series(list(pd.read_csv(str('data/cdf/'+path),header=None).loc[0,::-1]))
#    prm = scipy.stats.weibull_min.fit(data)
    sf=scipy.stats.weibull_min.sf(range(len(data)), 5.5, 0, 19)
    factor=8.53e6/19
    ssf = sf*factor
    plt.plot(range(len(data)), data, range(len(data)), ssf)
    return data
    


