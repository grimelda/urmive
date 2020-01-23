import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scipy
import scipy.stats
import stockflow as sf
import stocks

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


def BinaryShifts(x, D, mode, figs=False, PW='err'):
    for key in D.keys():
        pal = sns.color_palette("Set1")
        y = [D[key], 1-D[key] ]
        plt.stackplot(x, y, colors=pal, alpha=0.6 )
        plt.savefig(str('figures/Binary'\
                        +'_'+PW
                        +'_'+mode
                        +key
                        +'.pdf'))#, dpi=300)
        if figs is True:
            plt.show()
        plt.clf()

    
def PlotService(x, df, y='Share', 
                scn='HoogPerson', 
                figs=False, 
                savefig='pdf', 
                color='Mode',
                group='Vehicle',
                PW='BAU'
                ):
    fig = px.area(df, x = 'Year', y = y, 
                      color = color, 
                      line_group = group,
                      width = 950,
                      height = 400,
                      ).update_layout(\
#                                      xaxis_title="Year",
#                                      yaxis_title="Mass [tons]" if ylabel is False else ylabel,
                                      font = dict(
                                              size=11,
                                              family='Lato, sans serif'
                                              ),
                                      legend=dict(
                                              y=0.5, 
                                              traceorder='reversed',
                                              ),
                                      margin=dict(
                                              l=50,r=0,b=30,t=0,pad=1
                                              )
                                      )
    fig.for_each_trace(
        lambda trace: trace.update(name=trace.name.replace('Mode=', '')),
        )
    fig.for_each_trace(
        lambda trace: trace.update(name=trace.name.replace('Vehicle=', '')),
        )

    if figs is True:
        fig.show()
        
    fig.write_image(str('figures/'\
                                +PW+'_'\
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
    st=.76; en=.1 if PW=='TF' else st
    D['drive'] = InnoDiff(x, start=st, end=en, steepness=0.25, midpoint=2035)
    
    st=.99999; en=.15 if PW=='ST' else st
    D['icev'] = InnoDiff(x,start=st,end=en, steepness=0.3,midpoint=2030)

    st=.00001; en=.95 if PW=='ST' else st
    D['ev'] = InnoDiff(x, start=st, end=en, steepness=0.3, midpoint=2025)

    st=.5; en=.72 if PW=='TF' else st
    D['public'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2030 )

    # no changes modelled 
    D['ptrain'] = InnoDiff(x, start=0.77, end=0.77, steepness=1,  midpoint=2040 )

    # no changes modelled 
    D['cycle'] = InnoDiff(x, start=0.80, end=0.80, steepness=1, midpoint=2035 )

    st=.93; en=.85 if PW=='TF' else st
    D['bicycle'] = InnoDiff(x, start=st, end=en, steepness=0.3, midpoint=2030 )

    st=.999; en=.8 if PW=='ST' else st
    D['bike'] = InnoDiff(x, start=st, end=en, steepness=0.25, midpoint=2022 )

    st=.999; en=.01 if PW=='ST' else st
    D['moped'] = InnoDiff(x, start=st, end=en, steepness=0.3, midpoint=2025 )
    
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
#    CarCurve = InnoDiff(x, start=CarPerPkm, end=CarPerPkm*0.5,
#                        steepness=0.3, midpoint=2030 )
#    CarCorr = WeibCorr(x, start=2000, scale=20, shape=3, magn=2.3e5)
    STV['icev'] = InnoDiff(x, start=CarPerPkm, end=CarPerPkm*1.22, steepness=.37, midpoint=2007 )
    if PW=='RA': STV['icev'] = STV['icev'] - InnoDiff(x, start=0, end=CarPerPkm*0.59, steepness=.3, midpoint=2030 )
    STV['ev'] = STV['icev']
    STV['hev'] = STV['icev']
#    StackPlot(x, CarCurve, CarCorr, 'CarsPerPersonkm.png', figs=False)
    
    ### bikes per person kilometer, assumed to be constant
    BikePerPkm = 17.8e6/15.5/20*19
#    BikeCurve = InnoDiff(x, start=BikePerPkm, end=BikePerPkm*0.75,
#                      steepness=0.3, midpoint=2030 )
#    BikeCorr = WeibCorr(x, start=2000, scale=20, shape=3, magn=5e6)
    STV['bike'] = InnoDiff(x, start=BikePerPkm, end=BikePerPkm*1.42, steepness=.37, midpoint=2014 )
    if PW=='RA': STV['bike'] = STV['bike'] - InnoDiff(x, start=0, end=CarPerPkm*0.47, steepness=.3, midpoint=2030 )
    STV['ebike'] = STV['bike']
#    StackPlot(x, BikeCurve, BikeCorr, 'BikesPerPersonkm.png', figs=False)
    
    ### train capacity per person kilometer, assumed to be constant
    TrainPerPkm = 1301/17.1
    TrainCurve = InnoDiff(x, start=TrainPerPkm, end=TrainPerPkm, steepness=1, midpoint=2035 )
    STV['ptrain'] = TrainCurve

    ### buses per person kilometer, assumed to be constant
    BusPerPkm = 11634/6.7
    BusCurve = InnoDiff(x, start=BusPerPkm, end=BusPerPkm, steepness=1, midpoint=2035 )
    STV['bus'] = BusCurve
    
    ### mopeds per person kilometer, assumed to be constant
    MopedPerPkm = 0.7e6/1
    st=MopedPerPkm; en=st
    if PW=='ST': en=en*0.3
    MopedCurve = InnoDiff(x, start=st, end=en, steepness=0.3, midpoint=2035 )
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
    keys = ['evan', 'icevan', '40tlorry', '28tlorry', '16tlorry', 'ftrain', 'xlbarge', 'lbarge', 'mbarge', 'sbarge']
    for key in keys:
        temp = pd.DataFrame()
        temp['Year'] = x
        temp['Vehicle'] = key
        temp['Share'] = np.nan
        df = pd.concat([df, temp], ignore_index=True, sort=False)
        
    D = dict()
    st=.0797; en=.0797
    D['VAN'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2035)

    st=.001; en=.85 if PW=='ST' else st
    D['EVAN'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2035)

    st=.6; en=.29 if PW=='TF' else st
    D['ROAD'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2035)
    
    st=.06; en=.1 if PW=='ST' else st
    D['16TL'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2030)
    
    st=.22; en=.4 if PW=='ST' else st
    D['40TL'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2030)
    
    st=.11; en=.4 if PW=='TF' else st
    D['RAIL'] = InnoDiff(x, start=st, end=en, steepness=0.2, midpoint=2030)
    
    st=.47; en=.2 if PW=='ST' else st
    D['SBARGE'] = InnoDiff(x, start=st, end=en, steepness=0.1, midpoint=2030)
    
    st=.45; en=.2 if PW=='ST' else st
    D['MBARGE'] = InnoDiff(x, start=st, end=en, steepness=0.1, midpoint=2030)
    
    st=.6; en=.4 if PW=='ST' else st
    D['LBARGE'] = InnoDiff(x, start=st, end=en, steepness=0.1, midpoint=2030)
    
    
    ones = np.ones(len(x))
    df.loc[df['Vehicle']=='evan', 'Share'] = D['VAN'] * D['EVAN']
    df.loc[df['Vehicle']=='icevan', 'Share'] = D['VAN'] * (ones - D['EVAN'])

    df.loc[df['Vehicle']=='16tlorry', 'Share'] = (ones - D['VAN']) * D['ROAD'] * D['16TL']
    df.loc[df['Vehicle']=='28tlorry', 'Share'] = (ones - D['VAN']) * D['ROAD'] * (ones - D['16TL']) * (ones - D['40TL'])
    df.loc[df['Vehicle']=='40tlorry', 'Share'] = (ones - D['VAN']) * D['ROAD'] * (ones - D['16TL']) * D['40TL']

    df.loc[df['Vehicle']=='ftrain', 'Share'] = (ones - D['ROAD']) * D['RAIL']
    
    df.loc[df['Vehicle']=='sbarge', 'Share'] = (ones - D['VAN']) * (ones - D['ROAD']) * (ones - D['RAIL']) * D['SBARGE']
    df.loc[df['Vehicle']=='mbarge', 'Share'] = (ones - D['VAN']) * (ones - D['ROAD']) * (ones - D['RAIL']) * (ones - D['SBARGE']) * D['MBARGE'] 
    df.loc[df['Vehicle']=='lbarge', 'Share'] = (ones - D['VAN']) * (ones - D['ROAD']) * (ones - D['RAIL']) * (ones - D['SBARGE']) * (ones - D['MBARGE']) * D['LBARGE'] 
    df.loc[df['Vehicle']=='xlbarge', 'Share'] = (ones - D['VAN']) * (ones - D['ROAD']) * (ones - D['RAIL']) * (ones - D['SBARGE']) * (ones - D['MBARGE']) * (ones - D['LBARGE'])
    
    df['Ton-kilometers'] = None
    for i in df['Vehicle'].unique():
        df.loc[df['Vehicle']==i, 'Ton-kilometers'] = df.loc[df['Vehicle']==i, 'Share'].multiply(poly)
    
    modemap = {'Road' : ['evan', 'icevan', '40tlorry', '28tlorry', '16tlorry'],
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
    vanPerT = 852632/10
    VanCurve = InnoDiff(x, start=vanPerT, end=vanPerT*0.75 if PW=='RA' else vanPerT, steepness=.2, midpoint=2035 )
    STV['evan'] = VanCurve 
    STV['icevan'] = VanCurve 

    ### lorries per ton kilometer
    lorry16PerT = 73418/54.1 * 20.3/16 #136000/54.1*28/16
    lorry28PerT = 73418/54.1 * 20.3/28 #136000/54.1*28/28
    lorry40PerT = 73418/54.1 * 20.3/40 #236000/54.1*28/40
    L16Curve = InnoDiff(x, start=lorry16PerT, end=lorry16PerT*0.75 if PW=='RA' else lorry16PerT, steepness=.2, midpoint=2035 )
    L28Curve = InnoDiff(x, start=lorry28PerT, end=lorry28PerT*0.75 if PW=='RA' else lorry28PerT, steepness=.2, midpoint=2035 )
    L40Curve = InnoDiff(x, start=lorry40PerT, end=lorry40PerT*0.75 if PW=='RA' else lorry40PerT, steepness=.2, midpoint=2035 )
    STV['16tlorry'] = L16Curve 
    STV['28tlorry'] = L28Curve
    STV['40tlorry'] = L40Curve
    
    ### trains per ton-kilometer
    trainPerT = 1000/5.9
    trainCurve = InnoDiff(x, start=trainPerT, end=trainPerT,
                        steepness=1, midpoint=2035 )
    STV['ftrain']  = trainCurve

    ### inland barges per ton-kilometer
    XlbPerT = 2.45 * 5382/46.6 * 1153/3500
    LbPerT = 2.45 * 5382/46.6 * 1553/2500
    MbPerT = 2.45 * 5382/46.6 * 1553/1500
    SbPerT = 2.45 *5382/46.6 * 1553/750
    xlbCurve = InnoDiff(x, start=XlbPerT, end=XlbPerT*0.75 if PW=='RA' else XlbPerT, steepness=.2, midpoint=2035 )
    lbCurve = InnoDiff(x, start=LbPerT, end=LbPerT*0.75 if PW=='RA' else LbPerT, steepness=.2, midpoint=2035 )
    mbCurve = InnoDiff(x, start=MbPerT, end=MbPerT*0.75 if PW=='RA' else MbPerT, steepness=.2, midpoint=2035 )
    sbCurve = InnoDiff(x, start=SbPerT, end=SbPerT*0.75 if PW=='RA' else SbPerT, steepness=.2, midpoint=2035 )
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
    
    modemap = {'Seavessels' : keys,
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
        df.at[df['Vehicle']==key, 'VehicleCount'] = (0.84 * 1400 / 6070 * GDP) * df.loc[df['Vehicle']==key, 'Share']

    return df, STV

def RunPW(PW, figs=False, wlo='laag'):
    ### set x-axis and fidelity.
    startmodel = 2000
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
        BinaryShifts(x, DP, 'HoogPerson', PW=PW)
        dfP, STVp = ServiceToPersonVehicles(x, dfP, PW=PW)
        PlotService(x, dfP, y='Share', scn='HoogPerson')
        PlotService(x, dfP, y='Person-kilometers', scn='HoogPerson', figs=figs, PW=PW)
        PlotService(x, dfP, y='VehicleCount', scn='HoogPerson', figs=figs, PW=PW)
    
    if wlo=='laag':
        dfP, DP = CarsFirst(x, PolyLP, PW=PW)
        BinaryShifts(x, DP, 'LaagPerson', PW=PW)
        dfP, STVp = ServiceToPersonVehicles(x, dfP, PW=PW)
        PlotService(x, dfP, y='Share', scn='LaagPerson')
        PlotService(x, dfP, y='Person-kilometers', scn='LaagPerson', figs=figs, PW=PW)
        PlotService(x, dfP, y='VehicleCount', scn='LaagPerson', figs=figs, PW=PW)
    
    ###   F R E I G H T   T R A N S P O R T 
    if wlo=='hoog':    
        dfF, DF = RoadFirst(x, PolyHF)
        BinaryShifts(x, DF, 'HoogFreight', PW=PW)
        dfF, STVf = ServiceToFreightVehicles(x, dfF, PW=PW)
        PlotService(x, dfF, y='Share', scn='HoogFreight')
        PlotService(x, dfF, y='Ton-kilometers', scn='HoogFreight', figs=figs, PW=PW)
        PlotService(x, dfF, y='VehicleCount', scn='HoogFreight', figs=figs, PW=PW)
    
    if wlo=='laag':
        dfF, DF = RoadFirst(x, PolyLF, PW=PW)
        BinaryShifts(x, DF, 'LaagFreight', PW=PW)
        dfF, STVf = ServiceToFreightVehicles(x, dfF, PW=PW)
        PlotService(x, dfF, y='Share', scn='LaagFreight')
        PlotService(x, dfF, y='Ton-kilometers', scn='LaagFreight', figs=figs, PW=PW)
        PlotService(x, dfF, y='VehicleCount', scn='LaagFreight', figs=figs, PW=PW)
    
    ### F L I G H T   M O D E L I N G 
    if wlo=='hoog':    
        dfAC, DAC = Flights(x, PolyHAC)
        BinaryShifts(x, DAC, 'HoogAir', PW=PW)
        dfAC, STVac = ServiceToFlightVehicles(x, dfAC, service='Person-movements', PW=PW)
        PlotService(x, dfAC, y='Share', scn='HoogAir')
        PlotService(x, dfAC, y='Person-movements', scn='HoogAir', figs=figs, PW=PW)
        PlotService(x, dfAC, y='VehicleCount', scn='HoogAir', figs=figs, color='Vehicle', PW=PW)
    
    if wlo=='laag':
        dfAC, DAC = Flights(x, PolyLAC, PW=PW)
        BinaryShifts(x, DAC, 'LaagAir', PW=PW)
        dfAC, STVlac = ServiceToFlightVehicles(x, dfAC, service='Person-movements', PW=PW)
        PlotService(x, dfAC, y='Share', scn='LaagAir')
        PlotService(x, dfAC, y='Person-movements', scn='LaagAir', figs=figs, PW=PW)
        PlotService(x, dfAC, y='VehicleCount', scn='LaagAir', figs=figs, color='Vehicle', PW=PW)
    
    
    ### S E A   M O D E L I N G    
    if wlo=='hoog':    
        dfSV, DSV = SeaVessels(x, PolyHSV)
        BinaryShifts(x, DSV, 'HoogTGW')
        dfSV, STVhsv = ServiceToSeaVehicles(x, dfSV, service='SeaTGW', PW=PW)
        PlotService(x, dfSV, y='Share', scn='HoogSea')
        PlotService(x, dfSV, y='SeaTGW', scn='HoogSea', figs=figs, PW=PW)
        PlotService(x, dfSV, y='VehicleCount', scn='HoogSea', figs=figs, color='Vehicle', PW=PW)
    
    if wlo=='laag':
        dfSV, DSV = SeaVessels(x, PolyLSV, PW=PW)
        BinaryShifts(x, DSV, 'LaagTGW')
        dfSV, STVlsv = ServiceToSeaVehicles(x, dfSV, service='SeaTGW', PW=PW)
        PlotService(x, dfSV, y='Share', scn='LaagSea')
        PlotService(x, dfSV, y='SeaTGW', scn='LaagSea', figs=figs, PW=PW)
        PlotService(x, dfSV, y='VehicleCount', scn='LaagSea', figs=figs, color='Vehicle', PW=PW)

    
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
        if PW=='RC': vios[v]['lifespan'] = vios[v]['lifespan'] * np.append(np.ones(20),((0.0164 * x) - 31.8))[:-20]
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
        vios[v]['lifespan'] = np.array(IOS['lifespan'])

    
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
        temp.at[:, 'lifespan'] = vios[key]['lifespan']

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


def PW5(\
        PWs=['BAU', 'ST', 'RA', 'RC', 'TF'],
        recalc_mass=False,
        recalc_pw=False,
        ):
    
    ### force mass recalc if pw are recalculated
    if recalc_pw is True: 
        recalc_mass=True
    
    if recalc_mass is False:
        mat = pd.read_csv('data/PW5.csv', 
                          index_col=False,
                          na_filter=False,
                          dtype = {\
                                   'Year':'int16',
                                   'Vehiclename':'str',
                                   'Stock':'float64',
                                   'Inflow':'float64',
                                   'Outflow':'float64',
                                   'Vehicle':'str',
                                   'Class':'str',
                                   'Vmass':'float64',
                                   'lifespan':'float16',
                                   'PW':'str',
                                   'wlo':'str',
                                   'Material Group':'str',
                                   'Material':'str',
                                   'Unitmass':'float64',
                                   'Component':'str',
                                   'Mstock':'float64',
                                   'Minflow':'float64',
                                   'Moutflow':'float64',
                                   }
                          )
        dbx = pd.read_csv('data/VIOS.csv', index_col=False)
        
    if recalc_mass is True:
        
        if recalc_pw is False:
            dbx = pd.read_csv('data/VIOS.csv', index_col=False)
            
        if recalc_pw is True:
            
            dbx = pd.DataFrame()
            for PW in PWs:
                print('\nPathway: '+PW)
                df = RunPW(PW, figs=False, wlo='laag')
                dbx = pd.concat([dbx, df], ignore_index=True, sort=False)
            
            dbx = stocks.FixMatColumnTypes(dbx,
                                    coltypes = {'Year':'int16',
                                                'Vehicle':'str',
                                                'Stock':'float64',
                                                'Inflow':'float64',
                                                'Outflow':'float64',
                                                'Class':'str',
                                                'Vehiclename':'str',
                                                'Vmass':'float64',
                                                'lifespan':'float16',
                                                'PW':'str',
                                                'wlo':'str',
                                                })
            dbx.to_csv('data/VIOS.csv', index=False)
        
    ### DO THE MASS CALCULATION
    
    ### prepare mass dataframe
    path = 'data/fmass/'
    cols = ['Material Group', 'Material', 'Unitmass', 'Vehicle', 'Component']
    dbm = pd.DataFrame()
    for i in os.listdir(path):
        df = pd.read_csv(path+i)
        if 'Component' not in df.columns: df['Component'] = ''
        dbm = pd.concat([dbm, df[cols]], ignore_index=True, sort=False)
    dbm = dbm[dbm['Unitmass']>0]
    dbm = stocks.FixMatColumnTypes(dbm,
                            coltypes = {'Material Group':'str',
                                        'Material':'str',
                                        'Unitmass':'float64',
                                        'Vehicle':'str',
                                        'Component':'str',
                                        })
    mat = pd.DataFrame()
    for v in dbm['Vehicle'].unique():
        df = pd.merge(dbx.loc[dbx['Vehicle']==v],
                      dbm.loc[dbm['Vehicle']==v],
                      on='Vehicle',
                      how='outer',
                      )
        df['Mstock'] = df['Stock'] * df['Unitmass'] * df['Vmass'] /1e3
        df['Minflow'] = df['Inflow'] * df['Unitmass'] * df['Vmass'] /1e3
        df['Moutflow'] = df['Outflow'] * df['Unitmass'] * df['Vmass'] /1e3
        mat = pd.concat([mat, df], ignore_index=True, sort=False)
    mat = mat.dropna(subset=['Year'])
    mat = stocks.FixMatColumnTypes(mat,
                    coltypes = {\
                                'Year':'int16',
                                'Mstock':'float64',
                                'Minflow':'float64',
                                'Moutflow':'float64',
                                })
    
    mat = mat.drop(columns=['Component', 'wlo', 'lifespan'])
    mat
    mat.to_csv('data/PW5.csv', index=False)
    
    first = 2020
    last = 2051
    mat['MinflowCum'] = 0
    mat['MoutflowCum'] = 0
    for year in list(range(first,last)):
        mat.loc[mat['Year']==year, 'MinflowCum'] = mat.loc[mat['Year']==year, 'Minflow'] + np.array(mat.loc[mat['Year']==year-1, 'MinflowCum'])
        mat.loc[mat['Year']==year, 'MoutflowCum'] = mat.loc[mat['Year']==year, 'Moutflow'] + np.array(mat.loc[mat['Year']==year-1, 'MoutflowCum'])

        
    
    return dbx, dbm, mat