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


def WeibDist(x, dt, AvgLt, shape=5.5, loc = 0):
    ### sets larger bin size- otherwise with high lifetimes, bin would overflow
    binrange = range(0,int(len(x)*3))
    weib = dict()
    weib['pdf'] = scipy.stats.weibull_min.pdf(binrange, shape, loc, AvgLt)
    weib['cdf'] = scipy.stats.weibull_min.cdf(binrange, shape, loc, AvgLt)
    weib['sf'] = scipy.stats.weibull_min.sf(binrange, shape, loc, AvgLt)

    return weib


def Deaths_dt(hist, pdf, shape, dmulti):

    ### vector describing deaths per age bin, 7.05 is an optimised parameter    
    deaths_age = hist * pdf * dmulti#7.05
    ### create new histogram with 
    hist = hist-deaths_age
    deaths_age = sum(deaths_age)
            
    return deaths_age, hist


def HistWeib_t0(x, dt, shape, AvgLt, y0):
    ### vector, based on input weibull survival function
    hist = WeibDist(x, dt, AvgLt, shape, loc = 0)['sf']
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
                 shape,
                 IOS, #hist,
                 AvgLt,
                 dmulti,
                 ltmulti,
                 ):
    
    ### formulates weilbull distribution
    pdf = WeibDist(x, dt, AvgLt*ltmulti, shape, loc = 0)['pdf'] 
    # 1.33/2.7 || 1.5/4 || 2/7.05 || 1.9/6.0 || GC: 1.7//19(s5.5)

    ### calculates age related deaths during timestep from previous ts histogram
    (deaths_age, 
     hist,
     )= Deaths_dt(IOS.loc[t-1,'Hist'], pdf, shape, dmulti)
        
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
               scaleflow = 'dt', # either 'year' or 'dt'
               shape = 5.5, # shape for weibull distribution
               LtCorr = 1/0.925, # corr factor, scales outflow to match expected
               ):
    
    ### find time step size
    dt = (max(x)-min(x))/len(x[:-1])
    
    ### calibration factors
    dmulti = 7.99 * shape**0.623#3.27 * shape + 5.31
    C =  5.8543132044497312E+00
    T =  1.5839720757283211E+00
    K =  1.4650451557974826E+13
    Offset =  1.3522744683235266E+00
    ltmulti = C * shape**-T * np.exp(-shape/K) + Offset#6.2 * shape**-0.763
    '''
    p(k) = C * k(-T) * exp(-k/K) + Offset
    Fitting target of lowest sum of squared absolute error = 6.3588683778794508E-02
    '''
    
    ### outflow correction factors
    LtCorr = 0.942 + 0.181*shape - 0.0469*shape**2 + 3.39e-3*shape**3#1.11 + 0.0311*shape - 6.57e-3*shape**2

    
    ### scale lifetime to match timestep
    AvgLt= AvgLt/dt*LtCorr
    
    IOS = pd.DataFrame(columns=['x', 'Infl_dt', 'Outf_dt', 'Stock', 'Hist', 'Control'])
    IOS['x'] = x
    
    ### makes histogram for first timestep from weibull survival function at t0
    IOS.at[0, 'Hist'] = HistWeib_t0(x, dt, shape, AvgLt[0], y[0])
    IOS.at[0, 'Stock'] = sum(IOS.loc[0, 'Hist'])
    
    ### calculates inflows, outflows, and stocks for each timestep
    for t in IOS.index[1:]:
        IOS = InOutFlow_dt(x, y, t, dt, shape, IOS, AvgLt[t], dmulti, ltmulti)
    
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

#        ## alternative, dirty way (watch out mass balance looks wrong but isnt)    
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


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

    
def WeibFit(path='ships.csv',
            typ='raw', # either raw or hist
            lt=18,
            stock=1250,
            shape=2,
            ):
    data = pd.Series(list(pd.read_csv(str('data/cdf/'+path),header=None).loc[0,::-1]))
    if typ=='raw':
        data = np.histogram(data, bins=100)[0]

    sf=scipy.stats.weibull_min.sf(range(len(data)), shape, 0, lt)
    factor=stock/lt
    ssf = sf*factor
    
    plt.plot(range(len(data)), data)
    plt.plot(range(len(data)), ssf)
    plt.show()

    print('rsquared: ', round(rsquared(data,ssf), 4))
    print('pearsonr: ', round(scipy.stats.pearsonr(data,ssf)[0], 4))

    return ssf, data
    

def LogisticSignal(x):
    y = LogistiCurve(x, start=2e3, end=10e3, steepness=0.2, midpoint=2015)\
         - LogistiCurve(x, start=0, end=5e3, steepness=0.5, midpoint=2030)\
         + LogistiCurve(x, start=0, end=4e3, steepness=0.3, midpoint=2050)
    return y 
        

def FlatSignal(x, step=False):
    up = LogistiCurve(x, start=0, end=1e3, steepness=0.9, midpoint=2018)
    down = LogistiCurve(x, start=1e3, end=0, steepness=0.9, midpoint=2022)
    y = 2e3*np.ones(len(x))
    if step is True:
        y = y + up + down
    return y

def PlotResponse(IOS, y):
    plt.figure(figsize=(10,4))
    #plt.plot(x, y, label='Input')
    plt.plot(IOS['x'], IOS['Stock'], label='Stock')
    plt.plot(IOS['x'], IOS['Infl'], label='Inflow')
    plt.plot(IOS['x'], IOS['Outf'], label='Outflow')
    #plt.plot(IOS['x'], IOS['Control'], label='Control')
    plt.legend(loc='upper left')
    plt.ylim(-.0*max(y),1.2*max(y))
    plt.show()


def PlotHistograms(IOS, y):
    plt.figure(figsize=(10,4))
    plt.plot(IOS['Hist'][2])
    plt.plot(IOS['Hist'][29])
    plt.plot(IOS['Hist'][35])
    plt.plot(IOS['Hist'][45])
    plt.ylim(-1,1.2*max(IOS['Hist'][40]))
    plt.show()
    
def LifeTimeCorrection(x, y, log):
    
    i=0
    ltcor = pd.DataFrame()
    
    for lt in [2, 4, 7, 10, 20, 30, 50]:
        AvgLt = lt*np.ones(len(x))
        
        for shape in [1.2, 1.4, 1.8, 2, 3, 4, 5.5, 6]:
            IOS, dt = InOutStock(\
                                 x,
                                 y,
                                 AvgLt,
                                 scaleflow = 'dt', # either 'year' or 'dt'
                                 shape = shape, # shape for weibull distribution
                                 LtCorr = 1,#/0.923, # scales outflow to match expected outflow
                                 )
            ltcor.at[i, 'lifetime'] = lt
            ltcor.at[i, 'shape'] = shape
            ltcor.at[i, 'exp. lifetime'] = round(lt, 2)
            ltcor.at[i, 'real lifetime'] = round(IOS['Stock'].mean()/IOS['Outf'].mean()*dt,2)
            ltcor.at[i, 'lt factor'] = ltcor['real lifetime'][i]/ltcor['exp. lifetime'][i]
            ltcor.at[i, 'exp. outflow'] = round(IOS['Stock'].mean()/lt, 2)
            ltcor.at[i, 'real outflow'] = round(IOS['Outf'].mean()/dt,2)
            ltcor.at[i, 'OF factor'] = ltcor['real outflow'][i]/ltcor['exp. outflow'][i]
            i+=1
            if log is True:
                print(i, shape, lt)
    return ltcor
