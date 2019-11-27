import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import scipy.stats

#%%

def InOutStock(\
               x,
               y,
               AvgLs,
               scaleflow = 'dt', # either 'year' or 'dt'
               shape = 5, # shape for weibull distribution
               dm=False, # allows user to override deaths multiplier
               lm=False, # allows user to override lifespan multiplier
               dtype='jz',
               ):
    
    
    ### find time step size
    dt = (max(x)-min(x))/len(x[:-1])
    
    dmulti, lsmulti, LsCorr = WeibCalib(shape, dtype, lm, dm)
    
    ### scale lifespan to match timestep
    AvgLs= AvgLs/dt*LsCorr
    
    IOS = pd.DataFrame(columns=['x', 'Infl_dt', 'Outf_dt', 'Stock', 'Hist', 'Control'])
    IOS['x'] = x
    
    ### makes histogram for first timestep from weibull survival function at t0
    IOS.at[0, 'Hist'] = HistWeib_t0(x, dt, shape, AvgLs[0], y[0])
    IOS.at[0, 'Stock'] = sum(IOS.loc[0, 'Hist'])
    
    ### calculates inflows, outflows, and stocks for each timestep
    for t in IOS.index[1:]:
        IOS = InOutFlow_dt(x, y, t, dt, shape, IOS, AvgLs[t], dmulti, lsmulti, dtype)
    
    ### builds control curve, use to diagnose errors in mass balance
    IOS = ControlCurve(IOS)
    
    ### scales dataframe to fit yearly values or timestep values
    IOS = ScaleFlowsToYear(IOS, dt, scaleflow)
    
    ### Notes the shape and lifespan values
    IOS['shape'] = shape
    IOS['lifespan'] = np.mean(AvgLs)
    
    return IOS, dt

def WeibCalib(shape, dtype, lm, dm):

    if dtype=='jz':
        dm=False
        lm=False
    
    if dtype=='jz':
        ### calibration for deaths multiplier and lifespan multiplier
        dmulti = 7.99 * shape**0.623#
        ''' calibration factor for scale multiplier
        p(k) = C * k(-T) * exp(-k/K) + Offset
        Fitting target of lowest sum of squared absolute error = 6.6676704109128210E-02
        '''
        C =  5.8770581668489790E+00
        T =  1.5621696972883667E+00
        K =  3.5534833809179234E+13
        Offset =  1.3218793763933294E+00
        lsmulti = C * shape**-T * np.exp(-shape/K) + Offset
        ### outflow correction factors
        LsCorr = 0.728+ 0.375*shape - 0.108*shape**2 + 0.012*shape**3 -4.67e-4*shape**4


    if dtype=='sf':
        '''
        y = a * (x-b)c + Offset
    
        Fitting target of lowest sum of squared absolute error = 6.5036114209651166E-04
        '''
        a =  2.8898197136614545E+00
        b =  6.9387346583097098E-01
        c = -1.5430109179510860E+00
        Offset =  9.4214863283546935E-01
        lsmulti = a * (shape-b)**c + Offset
        dmulti = 1
        LsCorr = 1
    
    ### this allows user to override calibration multipliers using dm and lm variables
    if dtype=='cdf':
        dmulti = dm
        lsmulti = lm
        LsCorr = 1
    
    return dmulti, lsmulti, LsCorr


def Dstock_dt(\
              y,
              t,
              hist,
              ):
    
    ### calculates stock change at begin of current time step (after age deaths)
    dstock = y[t] - sum(hist) # scalar
    
    ### scalar for number of births in timestep 
    if dstock >= 0: # this represents 'births'
        hist[0] = dstock
    if dstock < 0: # this represents non-age-related deaths
        deaths_ds = hist - (sum(hist)-dstock)/sum(hist)*hist # absolute negative vector
        hist = hist+deaths_ds # absolute positive vector
#        hist = (sum(hist)-dstock)/sum(hist) * hist
        
    return dstock, hist


def Deaths_dt(hist, dt, pdf, cdf, sf, shape, AvgLs, dmulti, dtype='jz'):
    
    ###   C D F   methode
    if dtype=='cdf':
        stock = sum(hist)
        ### vector describing deaths per age cohort
        deaths_age = stock/AvgLs * (np.append(cdf,1)[1:] - cdf)    
        ### subtract deaths from histogram
        hist = hist - deaths_age    
        ### shift histogram to next time step
        hist = np.append(0,hist)[:-1]
    
    ###   S F   method
    if dtype=='sf':
        hist0 = hist
        hist = hist * sf #np.append(sf,0)[1:]
        deaths_age = hist0-hist
        hist = np.append(0,hist)[:-1]
    
#    ###   P D F   methode
    if dtype=='jz':
        ### vector describing deaths per age bin
        deaths_age = hist * pdf * dmulti
        ### create new histogram with subtracted deaths
        hist = hist-deaths_age
        ### shift histogram for next timestep
        hist = np.append(0,hist)[:-1]
            
    return sum(deaths_age), hist


def InOutFlow_dt(\
                 x, 
                 y,
                 t,
                 dt,
                 shape,
                 IOS,
                 AvgLs,
                 dmulti,
                 lsmulti,
                 dtype,
                 ):
    
    ### formulates weibull distribution
    pdf = WeibDist(x, dt, AvgLs*lsmulti, shape, loc = 0)['pdf'] 
    cdf = WeibDist(x, dt, AvgLs*lsmulti, shape, loc = 0)['cdf']
    sf = WeibDist(x, dt, AvgLs*lsmulti, shape, loc = 0)['sf']


    ### calculates age related deaths during timestep from previous ts histogram
    (deaths_age, 
     hist,
     )= Deaths_dt(IOS.loc[t-1,'Hist'], dt, pdf, cdf, sf, shape, AvgLs, dmulti, dtype=dtype)
        
    ### calculates stock change during timestep
    (dstock,
     hist,
     ) = Dstock_dt(y, t, hist)
    
    ### fill IOS dataframe    
    IOS.at[t, 'dstock'] = dstock
    IOS.at[t, 'Hist'] = hist
    IOS.at[t, 'Stock'] = sum(hist)
    if dstock > 0:
        IOS.at[t, 'Infl_dt'] = dstock
        IOS.at[t, 'Outf_dt'] = deaths_age
    if dstock <= 0:
        IOS.at[t, 'Infl_dt'] = 0
        IOS.at[t, 'Outf_dt'] = deaths_age - dstock # dstock is negative, so adds

    return IOS # results dataframe


def WeibDist(x, dt, AvgLs, shape=5, loc = 0):
    ### sets larger bin size- otherwise with high lifespans, bin would overflow
    binrange = range(0, int(len(x)*5))
    ### calculates weibull distros over selected range
    weib = dict()
    weib['pdf'] = scipy.stats.weibull_min.pdf(binrange, shape, loc, AvgLs)
    weib['cdf'] = scipy.stats.weibull_min.cdf(binrange, shape, loc, AvgLs)
    weib['sf'] = scipy.stats.weibull_min.sf(binrange, shape, loc, AvgLs)

    return weib


def AvgAge(hist, dt):
    age = []
    for i in range(len(hist)):
        age.append(hist[i]*i)
    AvgAge = sum(age)/sum(hist) * dt
    
    return AvgAge # in years


def HistWeib_t0(x, dt, shape, AvgLs, y0):
    ### vector, based on input weibull survival function
    hist = WeibDist(x, dt, AvgLs, shape, loc = 0)['sf']
    # normalise
    hist = hist/sum(hist)
    # scale to y0
    hist = y0*hist
    
    return hist


def ScaleFlowsToYear(IOS, dt, scaleflow):
    if scaleflow == 'dt':
        IOS = IOS.rename(columns={'Infl_dt':'Infl', 'Outf_dt':'Outf'})
        return IOS
    
    if scaleflow == 'year':        
#        ## use this if you dont want to scale years later on 
#        IOS = IOS.set_index('x', drop=True)
#        ### list of int years
#        years = list(IOS.index.astype('int').unique())
#        for year in years[1:-1]: # first and last years do not count
#            IOS.at[year,'Infl'] =  IOS.loc[((year<=IOS.index) & (IOS.index<year+1)),'Infl_dt'].sum()
#            IOS.at[year,'Outf'] =  IOS.loc[((year<=IOS.index) & (IOS.index<year+1)),'Outf_dt'].sum()
#        IOS = IOS.dropna(subset=['Infl'])
#        IOS = IOS.reset_index(drop=False)

        ## alternative, dirty way (watch out mass balance looks wrong but isnt)    
        IOS['Infl'] = IOS['Infl_dt'].multiply(1/dt)
        IOS['Outf'] = IOS['Outf_dt'].multiply(1/dt)

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

def WeibFit(\
            v = 'cars',
            figs = False,
            custom = False, # allows user to input custom dictionary.
            onlydata = False, # gets rid of orange weibull fit curve
            residuals = False, # 
            ):
    
    d = dict()
    d[v] = custom
    if d[v] is False:
        ### dictionary describing characteristics of demographics for selected vehicles
        d = dict(ex = dict(shape=5, stock=5000, lifespan=50, path='cars.csv'),
                 bus = dict(shape=5, stock=9822, lifespan=18, path='cars.csv'),
                 cars = dict(shape=5, stock=8222974, lifespan=18, path='cars.csv'),
                 vans = dict(shape=2, stock=2.252e6, lifespan=21.5, path='vans.csv'),
                 jets = dict(shape=10, stock=240, lifespan=20, path='jets.csv'),
#                 jets = dict(shape=4, stock=240, lifespan=20, path='jets.csv'),
                 ships = dict(shape=2, stock=1250, lifespan=30, path='ships.csv'),
                 trucks = dict(shape=2.5, stock=58159, lifespan=17, path='trucks.csv'),
                 trains = dict(shape=8, stock=1069, lifespan=40, path='trains.csv'),
                 mopeds = dict(shape=5, stock=1178300, lifespan=18, path='mopeds.csv'),
                 motorbikes = dict(shape=3.6, stock=655991, lifespan=38, path='motorbikes.csv'),
                 bestelwagen = dict(shape=4.5, stock=852622, lifespan=16, path='bestelwagen.csv'),
                 )
    
    df = pd.read_csv('data/cdf/'+d[v]['path'], header=0, index_col=0).drop(index=0)
    df = df.drop(index=df.index[-1]).dropna()

    factor = d[v]['stock'] / d[v]['lifespan']
    sf = scipy.stats.weibull_min.sf(range(200), d[v]['shape'], 0, d[v]['lifespan'])
    ssf = sf * factor
    for i in df.index:
        df.at[i, 'ssf'] = ssf[i]
        
    ###  descriptive statistics for fit:
    rsqu = round(rsquared(df.Count, df.ssf), 3)
    pear = round(scipy.stats.pearsonr(df.Count, df.ssf)[0], 3)
    
    plt.figure(figsize=(10,4))
    plt.bar(df.index, df.Count, color=(0.2, 0.4, 0.6, 0.6), label='data')
    if onlydata is False:
        plt.plot(range(0,200), ssf, color='orange', label='Weibull:'\
                                                              +'\nstock: ' + str(round(d[v]['stock']))\
                                                              +'\nlifespan: ' + str(d[v]['lifespan'])+'y'\
                                                              +'\nshape: ' + str(d[v]['shape'])\
                                                              +'\nrsquared: ' + str(rsqu) \
                                                              +'\npearsonr: ' + str(pear)\
                                                              )
    plt.legend(loc='upper right')
    plt.ylabel('Number of vehicles in age cohort')
    plt.xlabel('Age cohorts, by year')
    plt.savefig('figures/Weibull_' + v + '.pdf', dpi=300)        
    
    if figs is True: 
        plt.show()
    
    if residuals is True: 
        plt.figure(figsize=(10,4))
        plt.plot(df.index, 200*(df.Count-df.ssf)/max(df.ssf), 'r.', label='residual: relative difference between data and prediction')
        plt.plot(range(200), np.zeros(200), 'g-')
        plt.ylim(-100,100)
        plt.legend(loc='upper right')
#        fig.show()
        
    return df
        

def LogisticSignal(x):
    y = LogistiCurve(x, start=2e3, end=10e3, steepness=0.2, midpoint=2015)\
         - LogistiCurve(x, start=0, end=5e3, steepness=0.5, midpoint=2030)\
         + LogistiCurve(x, start=0, end=4e3, steepness=0.3, midpoint=2050)
    return y 
        

def FlatSignal(x, step=0):
    yavg = 5e3
    peak = yavg*step
    up = LogistiCurve(x, start=0, end=0, steepness=0.9, midpoint=2018)
    down = LogistiCurve(x, start=peak, end=0, steepness=0.9, midpoint=2022)
    y = yavg*np.ones(len(x))
    y = y + up + down - peak
    return y

def PlotResponse(IOS, y, figs=False):
    plt.figure(figsize=(10,4))
    #plt.plot(x, y, label='Input')
    plt.plot(IOS['x'], IOS['Stock'], 'k-', label='Stock')
    plt.plot(IOS['x'], IOS['Infl'], 'b-', label='Inflow')
    plt.plot(IOS['x'], IOS['Outf'], 'r-', label='Outflow')
    #plt.plot(IOS['x'], IOS['Control'], label='Control')
    plt.legend(loc='upper left')
    plt.ylim(-.0*max(y),1.2*max(y))
    plt.ylabel('Number of units')
    plt.xlabel('Year')
    plt.savefig(str('figures/inoutstock.png'), dpi=300)
    if figs is True: 
        plt.show()
        
def PlotHistograms(IOS, x, y, dt, figs=False, sel = [.1, .3, .5, .7, .9], bar=False):
    l=len(IOS['Hist'])

    if len(sel)==1:
        plt.figure(figsize=(10,4))
        for i in sel:
            data = IOS['Hist'][int(i*l)]
            plt.bar(range(len(data)), data, label='histogram t='+str(int(i*l+2000)))
        plt.ylabel('Number of units')
        plt.xlabel('Age cohort')
        plt.legend(loc='upper right')
        plt.savefig('figures/inoutstock_hist.png', dpi=300)
        plt.show()
    
    if len(sel)>1:
        if figs is True:
            plt.figure(figsize=(10,4))
            for i in sel:
                plt.plot(IOS['Hist'][int(i*l)], label='histogram t='+str(int(i*l*dt+2000)))
    
            plt.legend(loc='upper right')
            plt.ylim(-100,2.*max(IOS['Hist'][int(0.3*l)]))
            plt.xlim(0, 2*len(IOS['Hist'][int(0.3*l)][IOS['Hist'][int(0.3*l)]>0.001]))
            plt.ylabel('Number of units')
            plt.xlabel('Age cohort')
            plt.savefig('figures/inoutstock_hist.png', dpi=300)
            plt.show()



def PlotHistograms1(IOS, x, y, figs=False, sel = [.1, .3, .5, .7, .9]):
    w = 0.1
    l=len(IOS['Hist'])
    if figs is True:
#        fig, ax = plt.subplots()
        plt.figure(figsize=(10,4))
        for i in sel:
            plt.plot(IOS['Hist'][int(i*l)], label='histogram t='+str(int(i*l)))

        plt.legend(loc='upper right')
        plt.ylim(-1,2.*max(IOS['Hist'][int(0.3*l)]))
        plt.xlim(0, 1.5*len(IOS['Hist'][int(0.3*l)][IOS['Hist'][int(0.3*l)]>0.001]))
        plt.ylabel('Number of units')
        plt.xlabel('Age cohort')
        plt.savefig('figures/inoutstock_hist.png', dpi=300)
        plt.show()
    
def LifespanCorrection(x, y, log=True):
    
    i=0
    lscor = pd.DataFrame()
    
    for ls in [2, 4, 7, 10, 20, 30, 50]:
        AvgLs = ls*np.ones(len(x))
        
        for shape in [1.2, 1.4, 1.8, 2, 3, 4, 5.5, 6, 8, 10]:
            IOS, dt = InOutStock(\
                                 x,
                                 y,
                                 AvgLs,
                                 scaleflow = 'dt', # either 'year' or 'dt'
                                 shape = shape, # shape for weibull distribution
                                 )
            lscor.at[i, 'lifespan'] = ls
            lscor.at[i, 'shape'] = shape
            lscor.at[i, 'exp. lifespan'] = round(ls, 2)
            lscor.at[i, 'real lifespan'] = round(IOS['Stock'].mean()/IOS['Outf'].mean()*dt,2)
            lscor.at[i, 'lt factor'] = lscor['real lifespan'][i]/lscor['exp. lifespan'][i]
            lscor.at[i, 'exp. outflow'] = round(IOS['Stock'].mean()/ls, 2)
            lscor.at[i, 'real outflow'] = round(IOS['Outf'].mean()/dt,2)
            lscor.at[i, 'OF factor'] = lscor['real outflow'][i]/lscor['exp. outflow'][i]
            i+=1
            if log is True:
                print(i, shape, ls)
    return lscor
