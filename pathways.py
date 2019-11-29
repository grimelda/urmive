import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scipy
import scipy.stats
import stockflow as sf

#####   F U N C T I O N S

### generic functions

def PolyFit(x, deg, wlo, scn, col, n=3, figs=False):
    wlo = wlo.dropna(subset=[col])
    
    ### fourth order polyfit
    poly = np.poly1d(np.polyfit(wlo.index, wlo[col], deg))
    
    ### plot
    plt.plot(x, poly(x), 'r-',
             list(wlo.index)[:-n], wlo[col][:-n], 'b.', 
             list(wlo.index)[-n:], wlo[col][-n:], 'g*', 
            )
#     for mode in list(wlo.columns[1:]):
#         plt.plot(list(wlo.index), wlo[mode],'.')
    plt.legend(('interpolation','historical','WLO'), loc='right')
    if scn=='HP' or scn=='LP':
        plt.ylabel('Person-kilometers')
    if scn=='HF' or scn=='LF':
        plt.ylabel('Ton-kilometers')
    plt.ylim(0,1.1*max(poly(x)))
    plt.savefig(str('figures/WLO_Service_'\
                    +scn+'_'
                    +str(int(min(x)))\
                    +'-'+str(int(max(x)))\
                    +'.png'), dpi=400)
    if figs is True:
        plt.show()
    plt.clf()
    return poly(x)


def InnoDiff(x, start=0, end=1, steepness=1, midpoint=2030):
    if end < start:
        y = np.ones(len(x))*start - np.ones(len(x))*(start-end) / (1+np.exp(-steepness*(x-midpoint)))
    elif end >= start: 
        y = np.ones(len(x))*start + np.ones(len(x))*(end-start) / (1+np.exp(-steepness*(x-midpoint)))
    return y


def BinaryShifts(x, D, mode, figs=False):
    for key in D.keys():
        pal = sns.color_palette("Set1")
        y = [D[key], 1-D[key] ]
        plt.stackplot(x, y, colors=pal, alpha=0.6 )
        plt.savefig(str('figures/WLOHP_Share'\
                        +'_'+mode
                        +key
                        +'.png'), dpi=300)
        if figs is True:
            plt.show()
        plt.clf()

    
def PlotService(x, df, y='Share', 
                scn='HoogPerson', 
                figs=False, 
                savefig='pdf', 
                color='Mode',
                group='Vehicle',
                ):
    fig = px.area(df, x = 'Year', y = y, 
                      color = color, 
                      line_group = group,
                      width = 800,
                      height = 500,
                      ).update_layout(legend=dict(
                                                  y=0.5, 
                                                  traceorder='reversed', 
                                                  font_size=10,
                                                  ))
    if figs is True:
        fig.show()
        
    fig.write_image(str('figures/WLO_'\
                                +scn+'_'\
                                +y+'_'\
                                +str(int(min(x)))\
                                +'-'+str(int(max(x)))\
                                +'.'+savefig))


def StackPlot(x, a, b, path, figs=False):
    plt.stackplot(x, a, b)
    plt.savefig(str('figures/'+path), dpi=300)
    if figs is True:
        plt.show()
    plt.clf()
    
    
def WeibCorr(x, start=2000, scale=10, shape=2, magn=1e3):
    weibcorr = magn * scipy.stats.weibull_min.pdf(x,shape, loc=start, scale=scale)
    return weibcorr


###   PERSON TRANSPORT

def CarsFirst(x, poly, PW='BAU'):
    df = pd.DataFrame()
    keys = ['icev', 'ev', 'hev', 'ptrain', 'bus', 'ebike', 'bike', 'emoped', 'moped', 'walk']
    for key in keys:
        temp = pd.DataFrame()
        temp['Year'] = x
        temp['Vehicle'] = key
        temp['Share'] = np.nan
        df = pd.concat([df, temp], ignore_index=True, sort=False)
    
    D = dict()
        
    ### for drive first
    

    st=0.76; en=st; 
    if PW=='TF': en=0.1 
    D['drive'] = InnoDiff(x, start=st, end=en,
                          steepness=0.25, midpoint=2035)
    
    st=0.99; en=st;
    if PW=='ST': en=0.15 
    D['icev'] = InnoDiff(x,start=st,end=en,
                        steepness=0.3,midpoint=2030)

    st=0.01; en=st; 
    if PW=='ST': en=0.95 
    D['ev'] = InnoDiff(x, start=st, end=en,
                   steepness=0.3, midpoint=2025)

    st=0.5; en=st;
    if PW=='TF': en=0.72 
    D['public'] = InnoDiff(x, start=st, end=en,
                     steepness=0.2, midpoint=2030 )

    # no changes modelled 
    D['ptrain'] = InnoDiff(x, start=0.77, end=0.77,
                      steepness=1,  midpoint=2040 )

    # no changes modelled 
    D['cycle'] = InnoDiff(x, start=0.80, end=0.80,
                     steepness=1, midpoint=2035 )

    st=0.93; en=st;
    if PW=='TF': en=0.85 
    D['bicycle'] = InnoDiff(x, start=st, end=en,
                     steepness=0.3, midpoint=2030 )

    st=1; en=.9;
    if PW=='ST': en=0.8 
    D['bike'] = InnoDiff(x, start=st, end=en,
                     steepness=0.25, midpoint=2022 )

    st=0.999; en=st;
    if PW=='ST': en=0.001 
    D['moped'] = InnoDiff(x, start=st, end=en,
                      steepness=0.3, midpoint=2025 )
    
    ones = np.ones(len(x))
    ### cars
    df.loc[df['Vehicle']=='icev', 'Share'] = D['drive'] * D['icev']
    df.loc[df['Vehicle']=='ev', 'Share'] = D['drive'] * (ones - D['icev']) * D['ev']
    df.loc[df['Vehicle']=='hev', 'Share'] = D['drive'] * (ones - D['icev'])* (ones - D['ev'])

    ### public transport
    df.loc[df['Vehicle']=='ptrain', 'Share'] = (ones - D['drive']) * D['public'] * D['ptrain']
    df.loc[df['Vehicle']=='bus', 'Share'] = (ones - D['drive']) * D['public'] * (ones - D['ptrain'])

    ### slow transport
    df.loc[df['Vehicle']=='bike', 'Share'] = (ones - D['drive']) * (ones - D['public']) * D['cycle'] * D['bicycle'] * D['bike']
    df.loc[df['Vehicle']=='ebike', 'Share'] = (ones - D['drive']) * (ones - D['public']) * D['cycle'] * D['bicycle'] * (ones - D['bike'])
    df.loc[df['Vehicle']=='moped', 'Share'] = (ones - D['drive']) * (ones - D['public']) * D['cycle'] * (ones - D['bicycle']) * D['moped']
    df.loc[df['Vehicle']=='emoped', 'Share'] = (ones - D['drive']) * (ones - D['public']) * D['cycle'] * (ones - D['bicycle']) * (ones - D['moped'])
    df.loc[df['Vehicle']=='walk', 'Share'] = (ones - D['drive']) * (ones - D['public']) * (ones - D['cycle'])
    
    df['Person-kilometers'] = None
    for i in df['Vehicle'].unique():
        df.loc[df['Vehicle']==i, 'Person-kilometers'] = df.loc[df['Vehicle']==i, 'Share'].multiply(poly)
        
    modemap = {'Driving' : ['icev', 'ev', 'hev'],
               'Transit' : ['ptrain', 'bus'],
               'Cycling' : ['ebike', 'bike', 'moped', 'emoped'],
               'Walking' : ['walk'],
               }
    for key in list(modemap.keys()):
        for i in range(len(modemap[key])):
            df.loc[df['Vehicle']==modemap[key][i], 'Mode'] = key
    
    for each in df['Mode'].unique():
        print(each, round(df.loc[df['Year']==2050]\
                            .loc[df['Mode']==each,'Share'].sum(),3)\
              )
    return df, D


def ServiceToPersonVehicles(x, df, service='Person-kilometers', figs=False, PW='BAU'):
    STV = dict()
    
    ### cars per person kilometer
    CarPerPkm = 6.34e6/136.5
    CarCurve = InnoDiff(x, start=CarPerPkm, end=CarPerPkm*0.5,
                        steepness=0.3, midpoint=2030 )
    CarCorr = WeibCorr(x, start=2000, scale=20, shape=3, magn=2.3e5)
#    STV['icev'] = CarPerPkm*np.ones(len(CarCorr))
    STV['icev'] = InnoDiff(x, start=CarPerPkm, end=CarPerPkm*1.22, steepness=.37, midpoint=2007 )
    if PW=='RA':
        STV['icev'] = CarCurve + CarCorr 
    STV['ev'] = STV['icev']
    STV['hev'] = STV['icev']
    StackPlot(x, CarCurve, CarCorr, 'CarsPerPersonkm.png', figs=False)
    
    ### bikes per person kilometer, assumed to be constant
    BikePerPkm = 17.8e6/15.5/20*19
    BikeCurve = InnoDiff(x, start=BikePerPkm, end=BikePerPkm*0.75,
                      steepness=0.3, midpoint=2030 )
    BikeCorr = WeibCorr(x, start=2000, scale=20, shape=3, magn=5e6)
#    STV['bike'] = BikePerPkm*np.ones(len(BikeCorr))
    STV['bike'] = InnoDiff(x, start=BikePerPkm, end=BikePerPkm*1.42, steepness=.37, midpoint=2014 )
    if PW=='RA':
        STV['bike'] = BikeCurve +  BikeCorr
    STV['ebike'] = STV['bike']
    StackPlot(x, BikeCurve, BikeCorr, 'BikesPerPersonkm.png', figs=False)
    
    ### train capacity per person kilometer, assumed to be constant
    TrainPerPkm = 1301/17.1
    TrainCurve = InnoDiff(x, start=TrainPerPkm, end=TrainPerPkm,
                      steepness=1, midpoint=2035 )
    STV['ptrain'] = TrainCurve

    ### buses per person kilometer, assumed to be constant
    BusPerPkm = 11634/6.7
    BusCurve = InnoDiff(x, start=BusPerPkm, end=BusPerPkm,
                      steepness=1, midpoint=2035 )
    STV['bus'] = BusCurve
    
    ### mopeds per person kilometer, assumed to be constant
    MopedPerPkm = 0.7e6/1
    st=MopedPerPkm; en=st
    if PW=='ST': en=en*0.3
    MopedCurve = InnoDiff(x, start=st, end=en,
                      steepness=0.3, midpoint=2035 )
    STV['moped'] = MopedCurve
    STV['emoped'] = MopedCurve
    
    ###   M A T H 
    df['VehicleCount'] = None
    for key in STV.keys():
        df.loc[df['Vehicle']==key, 'VehicleCount'] = STV[key] * df.loc[df['Vehicle']==key, service]
        
    return df, STV


###   FREIGHT TRANSPORT

def RoadFirst(x, poly, PW='BAU'):
    df = pd.DataFrame()
    keys = ['40tlorry', '28tlorry', '16tlorry', 'ftrain', 'xlbarge', 'lbarge', 'mbarge', 'sbarge']
    for key in keys:
        temp = pd.DataFrame()
        temp['Year'] = x
        temp['Vehicle'] = key
        temp['Share'] = np.nan
        df = pd.concat([df, temp], ignore_index=True, sort=False)
        
    D = dict()
    st=.6; en=st
    if PW=='TF': en=.29
    D['ROAD'] = InnoDiff(x, start=st, end=en,
                         steepness=0.2, midpoint=2035)
    st=.06; en=st
    if PW=='ST': en=.1
    D['16TL'] = InnoDiff(x, start=st, end=en,
                         steepness=0.2, midpoint=2030)
    st=.22; en=st
    if PW=='ST': en=.4
    D['40TL'] = InnoDiff(x, start=st, end=en,
                         steepness=0.2, midpoint=2030)
    st=.11; en=st
    if PW=='ST': en=.4
    D['RAIL'] = InnoDiff(x, start=st, end=en,
                         steepness=0.2, midpoint=2030)
    st=.47; en=st
    if PW=='ST': en=.2
    D['SBARGE'] = InnoDiff(x, start=st, end=en,
                         steepness=0.1, midpoint=2030)
    st=.45; en=st
    if PW=='ST': en=.2
    D['MBARGE'] = InnoDiff(x, start=st, end=en,
                         steepness=0.1, midpoint=2030)
    st=.6; en=st
    if PW=='ST': en=.4
    D['LBARGE'] = InnoDiff(x, start=st, end=en,
                         steepness=0.1, midpoint=2030)
    
    
    ones = np.ones(len(x))
    df.loc[df['Vehicle']=='16tlorry', 'Share'] = D['ROAD'] * D['16TL']
    df.loc[df['Vehicle']=='28tlorry', 'Share'] = D['ROAD'] * (ones - D['16TL']) * (ones - D['40TL'])
    df.loc[df['Vehicle']=='40tlorry', 'Share'] = D['ROAD'] * (ones - D['16TL']) * D['40TL']

    df.loc[df['Vehicle']=='ftrain', 'Share'] = (ones - D['ROAD']) * D['RAIL']
    
    df.loc[df['Vehicle']=='sbarge', 'Share'] = (ones - D['ROAD']) * (ones - D['RAIL']) * D['SBARGE']
    df.loc[df['Vehicle']=='mbarge', 'Share'] = (ones - D['ROAD']) * (ones - D['RAIL']) * (ones - D['SBARGE']) * D['MBARGE'] 
    df.loc[df['Vehicle']=='lbarge', 'Share'] = (ones - D['ROAD']) * (ones - D['RAIL']) * (ones - D['SBARGE']) * (ones - D['MBARGE']) * D['LBARGE'] 
    df.loc[df['Vehicle']=='xlbarge', 'Share'] = (ones - D['ROAD']) * (ones - D['RAIL']) * (ones - D['SBARGE']) * (ones - D['MBARGE']) * (ones - D['LBARGE'])
    
    df['Ton-kilometers'] = None
    for i in df['Vehicle'].unique():
        df.loc[df['Vehicle']==i, 'Ton-kilometers'] = df.loc[df['Vehicle']==i, 'Share'].multiply(poly)
    
    modemap = {'Road' : ['40tlorry', '28tlorry', '16tlorry'],
               'Rail' : ['ftrain'],
               'Inland' : ['xlbarge', 'lbarge', 'mbarge', 'sbarge'],
               }
    for key in list(modemap.keys()):
        for i in range(len(modemap[key])):
            df.loc[df['Vehicle']==modemap[key][i], 'Mode'] = key
    
    for each in df['Mode'].unique():
        print(each, round(df.loc[df['Year']==2050]\
                            .loc[df['Mode']==each,'Share'].sum(),3)\
              )
        
    return df, D


def ServiceToFreightVehicles(x, df, service='Ton-kilometers', figs=False, PW='BAU'):
    STV = dict()
    
    ### lorries per ton kilometer
    lorry16PerT = 73418/54.1 * 20.3/16 #136000/54.1*28/16
    lorry28PerT = 73418/54.1 * 20.3/28 #136000/54.1*28/28
    lorry40PerT = 73418/54.1 * 20.3/40 #236000/54.1*28/40
    L16Curve = InnoDiff(x, start=lorry16PerT, end=lorry16PerT*0.75 if PW=='RC' else lorry16PerT, steepness=1, midpoint=2035 )
    L28Curve = InnoDiff(x, start=lorry28PerT, end=lorry28PerT*0.75 if PW=='RC' else lorry28PerT, steepness=1, midpoint=2035 )
    L40Curve = InnoDiff(x, start=lorry40PerT, end=lorry40PerT*0.75 if PW=='RC' else lorry40PerT, steepness=1, midpoint=2035 )
    STV['16tlorry'] = L16Curve 
    STV['28tlorry'] = L28Curve
    STV['40tlorry'] = L40Curve
    
    ### trains per ton-kilometer
    trainPerT = 1000/5.9
    trainCurve = InnoDiff(x, start=trainPerT, end=trainPerT,
                        steepness=1, midpoint=2035 )
    STV['ftrain']  = trainCurve

    ### inland barges per ton-kilometer
    XlbPerT = 5382/46.6 * 3500/1553
    LbPerT = 5382/46.6 * 2500/1553
    MbPerT = 5382/46.6 * 1500/1553
    SbPerT = 5382/46.6 * 750/1553
    xlbCurve = InnoDiff(x, start=XlbPerT, end=XlbPerT*0.75 if PW=='RC' else XlbPerT, steepness=1, midpoint=2035 )
    lbCurve = InnoDiff(x, start=LbPerT, end=LbPerT*0.75 if PW=='RC' else LbPerT, steepness=1, midpoint=2035 )
    mbCurve = InnoDiff(x, start=MbPerT, end=MbPerT*0.75 if PW=='RC' else MbPerT, steepness=1, midpoint=2035 )
    sbCurve = InnoDiff(x, start=SbPerT, end=SbPerT*0.75 if PW=='RC' else SbPerT, steepness=1, midpoint=2035 )
    STV['xlbarge']  = xlbCurve
    STV['lbarge']  = lbCurve  
    STV['mbarge']  = mbCurve  
    STV['sbarge']  = sbCurve  
    
    ### for output
    df['VehicleCount'] = None
    for key in STV.keys():
        df.loc[df['Vehicle']==key, 'VehicleCount'] = STV[key] * df.loc[df['Vehicle']==key, service]
        
    return df, STV


###   FLIGHT TRANSPORT

def Flights(x, poly, PW='BAU'):
    df = pd.DataFrame()
    keys = ['A330', 'B787']
    for key in keys:
        temp = pd.DataFrame()
        temp['Year'] = x
        temp['Vehicle'] = key
        temp['Share'] = np.nan
        df = pd.concat([df, temp], ignore_index=True, sort=False)
        
    D = dict()
    st=.9; en=st
    if PW=="ST": en=.1
    D['A330'] = InnoDiff(x, start=st, end=en,
                         steepness=0.2, midpoint=2030)    
    
    ones = np.ones(len(x))
    df.loc[df['Vehicle']=='A330', 'Share'] = D['A330'] 
    df.loc[df['Vehicle']=='B787', 'Share'] = (ones - D['A330'])
    
    key = 'Person-movements'
    df[key] = None
    for i in df['Vehicle'].unique():
        df.loc[df['Vehicle']==i, key] = df.loc[df['Vehicle']==i, 'Share'].multiply(poly)
    
    modemap = {'Air' : ['A330', 'B787'],
               }
    for key in list(modemap.keys()):
        for i in range(len(modemap[key])):
            df.loc[df['Vehicle']==modemap[key][i], 'Mode'] = key
    
    ### prints mode shares at last timestep
    for each in df['Mode'].unique():
        print(each, round(df.loc[df['Year']==2050]\
                            .loc[df['Mode']==each,'Share'].sum(),3)\
              )
        
    return df, D

def ServiceToFlightVehicles(x, df, service='Person-movements', figs=False, PW='BAU'):
    STV = dict()
    
    ### aircraft per passenger movements 
    PM = np.array(df.groupby('Year')['Person-movements'].sum())

    ### for output
    df['VehicleCount'] = None
    df.at[df['Vehicle']=='A330', 'VehicleCount'] = (145 + 1.13 * PM) * df.loc[df['Vehicle']=='A330', 'Share']#STV[key] * df.loc[df['Vehicle']==key, service]
    df.at[df['Vehicle']=='B787', 'VehicleCount'] = (145 + 1.13 * PM) * df.loc[df['Vehicle']=='B787', 'Share']#STV[key] * df.loc[df['Vehicle']==key, service]

    return df, STV

###   SEA VESSEL TRANSPORT

def SeaVessels(x, poly, driver='SeaTGW', PW='BAU'):

    shares = pd.read_csv('data/modalshift/ships.csv', header=0, index_col=None)
    keys = list(shares['Type'].unique())

    df = pd.DataFrame()
    for key in keys:
        temp = pd.DataFrame()
        temp['Year'] = x
        temp['Vehicle'] = key
        temp['Share'] = np.nan
        df = pd.concat([df, temp], ignore_index=True, sort=False)
    
    D = dict.fromkeys(shares['Type'].unique())
    for key in D.keys():
        D[key] = shares.loc[shares['Type']==key, 'Share'].values * np.ones(len(x))
        df.at[df['Vehicle']==key, 'Share'] = list(D[key])

    driver = 'SeaTGW'
    df[driver] = None
    for i in df['Vehicle'].unique():
        df.loc[df['Vehicle']==i, driver] = df.loc[df['Vehicle']==i, 'Share'].multiply(poly)
    
    modemap = {'Sea vessels' : keys,
               }
    for key in list(modemap.keys()):
        for i in range(len(modemap[key])):
            df.loc[df['Vehicle']==modemap[key][i], 'Mode'] = key
    
    ### prints mode shares at last timestep
    for each in df['Mode'].unique():
        print(each, round(df.loc[df['Year']==2050]\
                            .loc[df['Mode']==each,'Share'].sum(),3)\
              )
        
    return df, D

def ServiceToSeaVehicles(x, df, service='SeaTGW', figs=False, PW='BAU'):
    STV = dict()
    shares = pd.read_csv('data/modalshift/ships.csv', header=0, index_col=None)
    keys = list(shares['Type'].unique())

    ### sea vessel per SeaTGW
    GDP = np.array(df.groupby('Year')['SeaTGW'].sum())

    ### for output
    df['VehicleCount'] = None
    for key in keys:
        df.at[df['Vehicle']==key, 'VehicleCount'] = (1400 / 6070 * GDP) * df.loc[df['Vehicle']==key, 'Share']

    return df, STV

def RunPW(PW, figs=False, wlo='laag'):
    ### set x-axis and fidelity.
    startmodel = 1999
    endmodel = 2051
    x = np.linspace(startmodel, endmodel, endmodel-startmodel+1)
    
    ### read data
    wloHP = pd.read_csv('data/modalshift/WLOHpassenger.csv', index_col=0,header=0)
    wloLP = pd.read_csv('data/modalshift/WLOLpassenger.csv', index_col=0,header=0)
    wloHF = pd.read_csv('data/modalshift/WLOHfreight.csv', index_col=0,header=0)
    wloLF = pd.read_csv('data/modalshift/WLOLfreight.csv', index_col=0,header=0)
    wloLSV = pd.read_csv('data/modalshift/SeaL.csv', index_col=0,header=0)
    wloHSV = pd.read_csv('data/modalshift/SeaH.csv', index_col=0,header=0)
    
    ### convert to numeric column
    ### force polynomials to flatten out at 2050.
    wloHP.loc[2040,'Total']=250
    wloLP.loc[2040,'Total']=210
    wloHF.loc[2040,'Total']=170
    wloLF.loc[2040,'Total']=140
    wloHP.loc[2040,'Luchtvaart']=140
    
    ### "PolyXX" is the service demand over time.
    PolyHP = PolyFit(x, 4, wloHP, 'HP', 'Total')
    PolyLP = PolyFit(x, 4, wloLP, 'LP', 'Total')
    PolyHF = PolyFit(x, 4, wloHF, 'HF', 'Total')
    PolyLF = PolyFit(x, 4, wloLF, 'LF', 'Total')
    PolyLAC = PolyFit(x, 1, wloLP, 'LAC', 'Luchtvaart')
    PolyHAC = PolyFit(x, 4, wloHP, 'HAC', 'Luchtvaart')
    PolyLSV = PolyFit(x, 2, wloLSV, 'LSV', 'SeaGDP', n=32)
    PolyHSV = PolyFit(x, 2, wloHSV, 'HSV', 'SeaGDP', n=32)

    ###   P E R S O N A L   T R A N S P O R T 
    if wlo=='hoog':    
        dfP, DP = CarsFirst(x, PolyHP, PW=PW)
        BinaryShifts(x, DP, 'HoogPerson')
        dfP, STVp = ServiceToPersonVehicles(x, dfP, PW=PW)
        PlotService(x, dfP, y='Share', scn='HoogPerson')
        PlotService(x, dfP, y='Person-kilometers', scn='HoogPerson', figs=figs)
        PlotService(x, dfP, y='VehicleCount', scn='HoogPerson', figs=figs)
    
    if wlo=='laag':
        dfP, DP = CarsFirst(x, PolyLP, PW=PW)
        BinaryShifts(x, DP, 'LaagPerson')
        dfP, STVp = ServiceToPersonVehicles(x, dfP, PW=PW)
        PlotService(x, dfP, y='Share', scn='LaagPerson')
        PlotService(x, dfP, y='Person-kilometers', scn='LaagPerson', figs=figs)
        PlotService(x, dfP, y='VehicleCount', scn='LaagPerson', figs=figs)
    
    ###   F R E I G H T   T R A N S P O R T 
    if wlo=='hoog':    
        dfF, DF = RoadFirst(x, PolyHF)
        BinaryShifts(x, DF, 'HoogFreight')
        dfF, STVf = ServiceToFreightVehicles(x, dfF, PW=PW)
        PlotService(x, dfF, y='Share', scn='HoogFreight')
        PlotService(x, dfF, y='Ton-kilometers', scn='HoogFreight', figs=figs)
        PlotService(x, dfF, y='VehicleCount', scn='HoogFreight', figs=figs)
    
    if wlo=='laag':
        dfF, DF = RoadFirst(x, PolyLF, PW=PW)
        BinaryShifts(x, DF, 'LaagFreight')
        dfF, STVf = ServiceToFreightVehicles(x, dfF, PW=PW)
        PlotService(x, dfF, y='Share', scn='LaagFreight')
        PlotService(x, dfF, y='Ton-kilometers', scn='LaagFreight', figs=figs)
        PlotService(x, dfF, y='VehicleCount', scn='LaagFreight', figs=figs)
    
    ### F L I G H T   M O D E L I N G 
    if wlo=='hoog':    
        dfAC, DAC = Flights(x, PolyHAC)
        BinaryShifts(x, DAC, 'HoogAir')
        dfAC, STVac = ServiceToFlightVehicles(x, dfAC, service='Person-movements')
        PlotService(x, dfAC, y='Share', scn='HoogAir')
        PlotService(x, dfAC, y='Person-movements', scn='HoogAir')
        PlotService(x, dfAC, y='VehicleCount', scn='HoogAir', figs=figs, color='Vehicle')
    
    if wlo=='laag':
        dfAC, DAC = Flights(x, PolyLAC, PW=PW)
        BinaryShifts(x, DAC, 'LaagAir')
        dfAC, STVlac = ServiceToFlightVehicles(x, dfAC, service='Person-movements', PW=PW)
        PlotService(x, dfAC, y='Share', scn='LaagAir')
        PlotService(x, dfAC, y='Person-movements', scn='LaagAir')
        PlotService(x, dfAC, y='VehicleCount', scn='LaagAir', figs=figs, color='Vehicle')
    
    
    ### S E A   M O D E L I N G    
    if wlo=='hoog':    
        dfSV, DSV = SeaVessels(x, PolyHSV)
        BinaryShifts(x, DSV, 'HoogTGW')
        dfSV, STVhsv = ServiceToSeaVehicles(x, dfSV, service='SeaTGW')
        PlotService(x, dfSV, y='Share', scn='HoogSea')
        PlotService(x, dfSV, y='SeaTGW', scn='HoogSea')
        PlotService(x, dfSV, y='VehicleCount', scn='HoogSea', figs=figs, color='Vehicle')
    
    if wlo=='laag':
        dfSV, DSV = SeaVessels(x, PolyLSV, PW=PW)
        BinaryShifts(x, DSV, 'LaagTGW')
        dfSV, STVlsv = ServiceToSeaVehicles(x, dfSV, service='SeaTGW', PW=PW)
        PlotService(x, dfSV, y='Share', scn='LaagSea')
        PlotService(x, dfSV, y='SeaTGW', scn='LaagSea')
        PlotService(x, dfSV, y='VehicleCount', scn='LaagSea', figs=figs, color='Vehicle')

    
    ### combines all dataframes to a single Vehicle Count dataframe
    vc = pd.DataFrame() 
    for df in [dfP, dfF, dfAC, dfSV]:
        vc = pd.concat([vc, df[['Year', 'Vehicle', 'VehicleCount']]], ignore_index=True, sort=False)
    
    ### dictionary, VehicleInOutStock, to gather all stock and flow data
    vios = dict()
    for veh in list(vc['Vehicle'].unique()):
        vios[veh] = dict.fromkeys(['i', 'o', 's'])
        vios[veh]['s'] = np.array(vc.loc[vc['Vehicle']==veh, 'VehicleCount'])
    del vios['walk']
    
    ### read data for vehicles
    lifespan = pd.read_csv('data/cdf/lifespan.csv', header=0, index_col=None)
    for v in vios.keys():
#        print(v)
        vios[v]['lifespan'] = np.ones(len(x))*lifespan.loc[lifespan['Vehiclename']==v, 'lifespan'].values[0]
        if PW=='RC': vios[v]['lifespan'] = vios[v]['lifespan'] * ((0.0164 * x) - 31.8)
        vios[v]['Vehicle'] = lifespan.loc[lifespan['Vehiclename']==v, 'Vehicle'].values[0]
        vios[v]['shape'] = lifespan.loc[lifespan['Vehiclename']==v, 'shape'].values[0]
        vios[v]['Class'] = lifespan.loc[lifespan['Vehiclename']==v, 'Class'].values[0]
        vios[v]['Vmass'] = lifespan.loc[lifespan['Vehiclename']==v, 'Vmass'].values[0]
    
    ### calculate in and outflows, store in dictionary
    for v in vios.keys():
    #     print(v)
        IOS, dt = sf.InOutStock(\
                                x,
                                vios[v]['s'],
                                vios[v]['lifespan'],
                                shape = vios[v]['shape'],
                                scaleflow = 'dt',
                                )
        vios[v]['i'] = np.array(IOS['Infl'])
        vios[v]['o'] = np.array(IOS['Outf'])
    
    ### serve vios dictionary to dataframe
    VIOS = pd.DataFrame()
    for key in vios.keys():
        temp = pd.DataFrame()
        temp.at[:, 'Year'] = pd.Series(x)
        temp.at[:, 'Vehiclename'] = key
        temp.at[:, 'Stock'] = pd.Series(vios[key]['s'])
        temp.at[:, 'Inflow'] = pd.Series(vios[key]['i'])
        temp.at[:, 'Outflow'] = pd.Series(vios[key]['o'])
        temp.at[:, 'Class'] = vios[key]['Class']
        temp.at[:, 'Vehicle'] = vios[key]['Vehicle']
        temp.at[:, 'Vmass'] = vios[key]['Vmass']
        temp.at[:, 'lifespan'] = vios[key]['lifespan'].mean()

        VIOS = pd.concat([VIOS, temp], ignore_index=True, sort=False)
    
    ### drops trailing year entries
    VIOS = VIOS.loc[~(VIOS['Year'].isin([1999.0,2051.0]))]
    VIOS['PW'] = PW
    VIOS['wlo'] = wlo
    
    if figs is True:
        fig = px.area(VIOS, x = 'Year', y = 'Stock', 
                      color = 'Class', 
                      line_group = 'Vehicle',
                      ).update_layout(yaxis_title="Stock of vehicle units in NL",
                                      legend=dict(\
                                                  y=0.5, 
                                                  traceorder='reversed', 
                                                  font_size=10,
                                                  ))
        fig.show()
        fig = px.area(VIOS, x = 'Year', y = 'Outflow', 
                      color = 'Class', 
                      line_group = 'Vehicle',
                      ).update_layout(yaxis_title="Outflow of vehicle units within NL",
                                      legend=dict(\
                                                  y=0.5, 
                                                  traceorder='reversed', 
                                                  font_size=10,
                                                  ))
        fig.show()
        fig = px.area(VIOS, x = 'Year', y = 'Inflow', 
                      color = 'Class', 
                      line_group = 'Vehicle',
                      ).update_layout(yaxis_title="Inflow of vehicle units within NL",
                                      legend=dict(\
                                                  y=0.5, 
                                                  traceorder='reversed', 
                                                  font_size=10,
                                                  ))
        fig.show()
    return VIOS