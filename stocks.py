#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 12:22:44 2019
# @author: jvanderzaag

import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#%% high level defs

def CalcStocks(start=2000, end=2017):
    ### calc materials from vehicle units
    (
     dbx,
     dbm,
     dba,
     ) = ReadData()

    (
     dbx,
     Vtypes,
     ) = UnifyCountData(dbx, start, end)

    (
     dbm
     ) = UnifyMassData(dbx, dbm)

    mat = CalcMass(dbx, dbm) 
    return mat


def UnifyCountData(dbx, 
                   startyear=2000, 
                   endyear=2017,
                   ):
    
    dbx = CleanDataframes(dbx)
    dbx = AddVehicleTypeColumn(dbx)
    Vtypes = MakeVehicleTypeList(dbx)
    dbx = CombineDataframes(dbx, startyear, endyear)
    dbx = dbx.dropna(subset = ['Waarde'])
    dbx = dbx.rename(columns = {
                                'Vtype' : 'Vehicle',
                                'Waarde' : 'Value',
                                'Perioden' : 'Year',
                                })
    dbx = RemapVehicles(dbx)
    dbx = dbx[~dbx['Onderwerp'].str.contains('ota', na=False)]

    return dbx, Vtypes



#%% complexer defs

def CalcMass(
             dbx, 
             dbm,
             ):
    
    missing = list(set(dbx['Vehicle']) - set(dbm.keys()))
    if len(missing) > 0:
        print('No material data was found for',len(missing),'vehicle types...\n')
    
    mat = pd.DataFrame() 
    for key in list(dbm.keys()):
        df = dbx.loc[dbx['Vehicle'] == key].merge(
                                                  dbm[key], 
                                                  on='Vehicle', 
                                                  how='outer',
                                                  )
        df['Mass'] = df['Value'] * df['Unitmass']
        
        mat = pd.concat([mat, df], ignore_index=True, sort=False)
        
    mat = FixMatColumnTypes(mat)
    
    ### for each vehicle type which has weight info in count table:
    mat = CalcWeightFromCount(mat, slx=[
                                        'icevan',
                                        'lorry16t',
                                        'lorry28t',
                                        'lorry40t',
                                        'bus',
                                        'hmaxin',
                                        'hmax',
                                        'tanker',
                                        'bulker',
                                        'ptrain',
                                        'locomotive',
#                                         'wagon',
                                        ])
    return mat

def FixMatColumnTypes(df):
    coltypes = dict(\
                    Value='int',
                    Year='int',
                    Unitmass='float',
                    Mass='float',
                    Vehicle='str',
                    Onderwerp='str',
                    Class='str',
                    )
    for key in coltypes.keys():
        df.loc[:,key] = df.loc[:,key].astype(coltypes[key])
        
    return df
    

def CalcWeightFromCount(mat, slx):
    for i in slx:
        mat.loc[mat['Vehicle']==i, 'Mass'] = (mat.loc[mat['Vehicle']==i,'Mass'] 
                                              * 
                                              mat.loc[mat['Vehicle']==i,'Onderwerp'].astype('float')
                                              )
    return mat


def RemapVehicles(dbx):
    ### preserve original vehicle names
    dbx['Vehiclenames'] = dbx['Vehicle']
    
#     ### manually set some avg weight values
#     dbx = SetWeightManual(dbx, slx={
#                                     'Seavessel:handel' : 11097664, # compared to eurostat estimate: 87009621.9
#                                     'Seavessel:waterbouw': 10483008, # compared to eurostat estimate: 1880253.8
#                                     'Seavessel:zeesleepvaart' : 2052046, # compared to eurostat estimate: 378541.7
#                                     })
    
    ### 
    MAP = pd.read_csv('data/datamap.csv', header=None, index_col=False)
    for i in range(len(MAP)):
        dbx.loc[:,'Vehicle'] = dbx.loc[:,'Vehicle'].replace(MAP.loc[i,0],MAP.loc[i,1])

    return dbx


def UnifyMassData(dbx, dbm):
    veh = [x for x in os.listdir('data/mass/') if x.endswith('.csv')]
    veh = [x.replace('.csv','') for x in veh]
    
    df = pd.DataFrame()
    for key in veh:
        if len(dbx[dbx['Vehicle'] == key]) == 0:
            print('No count data found for vehicle:', key)
        else:
            df = pd.concat(#
                           [df, dbx[dbx.loc[:,'Vehicle'] == key]], 
                           ignore_index=True, 
                           sort=False,
                           )
    ma = dict()
    for key in veh:
        if len(dbx[dbx.loc[:,'Vehicle'] == key]) == 0: continue
        else:
            ma[key] = dbm[key].loc[:,['Material','Unitmass','Class']]#.set_index('material')
            ma[key]['Vehicle'] = key
    
            ma[key].loc[:,'Unitmass'] = pd.to_numeric(ma[key]['Unitmass'], errors='coerce')
            ma[key] = ma[key].dropna(subset=['Unitmass'])

    return ma

    
def AddVehicleTypeColumn(db): #2
    db['perscars']['Vtype'] = (pd.Series(['Personalcar:']*len(db['perscars'])) 
                              + db['perscars']['Onderwerp']
                              )
    db['compcars']['Vtype'] = (db['compcars']['Voertuigtype']
                              + pd.Series([':']*len(db['compcars']))
                              + db['compcars']['Onderwerp']
                              )
    db['mobikes']['Vtype'] = (pd.Series(['Motorbike:']*len(db['mobikes'])) 
                              + db['mobikes']['Onderwerp']
                              )
    db['mopeds']['Vtype'] = (pd.Series(['Mopeds:']*len(db['mopeds'])) 
                              + db['mopeds']['Onderwerp']
                              )
    db['planes']['Vtype'] = (pd.Series(['Aircraft:']*len(db['planes'])) 
                           + db['planes']['Onderwerp']
                           )
    db['Vlootboek']['Vtype'] = db['Vlootboek'].loc[:,'VlootboekNaam']
    db['Vlootboek'] = db['Vlootboek'].rename(columns={'VehicleAvgWeight':'Onderwerp'})
    db['Vlootboek'] = db['Vlootboek'].loc[db['Vlootboek']['Flow']=='stock']


    db['inships']['Vtype'] = (pd.Series(['Inlandvessel:']*len(db['inships'])) 
                              + db['inships']['Onderwerp']
                              )
    return db


def CombineDataframes(dbx, startyear, endyear):

    count = [x.replace('.csv','') for x in os.listdir('data/count/') if x.endswith('.csv')]
    DBX = pd.DataFrame()
    for key in count:    
        DBX = pd.concat([DBX, dbx[key].loc[:,['Vtype','Waarde', 'Perioden','Onderwerp']]], ignore_index=True, sort=False)
        
    DBX['Waarde'] = pd.to_numeric(DBX['Waarde'], errors='coerce')
    DBX['Perioden'] = pd.to_numeric(DBX['Perioden'], errors='coerce')
    DBX = DBX.loc[(DBX['Perioden'] >= startyear) & (DBX['Perioden'] <= endyear)]
    
    return DBX


def CleanDfEurostat(df, colnames):
    df = StripChars(df, 
                    'Value', 
                    [' ', ':'],
                    )
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.rename(columns=colnames)
    
    return df


def CleanDataframes(db): #1

    db['compcars'] = db['compcars'].dropna(subset=['Waarde'])
    db['compcars'] = db['compcars'][~db['compcars']['Onderwerp'].str.contains('otaal')]
    db['compcars'] = db['compcars'][~db['compcars']['Voertuigtype'].str.contains('otaal')]
    db['compcars'].loc[:,'Onderwerp'] = db['compcars']['Onderwerp'].str.replace(' ','')
    db['compcars'].loc[:,'Onderwerp'] = db['compcars']['Onderwerp'].str.replace('kg','')
    
    db['compcars'] = db['compcars'].reset_index(drop=True)
    
    db['perscars'] = ReplaceValues(db['perscars'],
                                  'Onderwerp',
                                  '2451enmeer',
                                  '2451-3500')
    db['compcars'] = ReplaceValues(db['compcars'],
                                  'Onderwerp',
                                  '30000enmeer',
                                  '30000-50000')
    db['compcars'] = ReplaceValues(db['compcars'],
                                  'Onderwerp',
                                  'Leeggewichtonbekend',
                                  '10000-10001')    
    db['compcars'] = ReplaceValues(db['compcars'],
                                  'Onderwerp',
                                  '<500',
                                  '0-500')
    ### get middle weight for compcars
    for o in ['compcars', 'inships']:
        for i in list(set(db[o]['Onderwerp'])):
            db[o] = ReplaceValues(db[o],
                                  'Onderwerp',
                                  i,
                                  str(int(np.mean([int(x) for x in i.split('-')])))
                                  )
    return db


def MakeVehicleTypeList(dbx):

    count = [x.replace('.csv','') for x in os.listdir('data/count/') if x.endswith('.csv')]

    Vtypes = []
    for key in count:
        if key in [
                   #'seaships',
                   #'inships',
                   ]: continue # handle exceptions
        newlist = list(dbx[key]['Vtype'])
        Vtypes = Vtypes + newlist
    Vtypes = [x for x in list(set(Vtypes)) if str(x) != 'nan']
    Vtypes = [x for x in Vtypes if 'otaal' not in x]
    return Vtypes


def ReadData():
    
    v = dict.fromkeys(['count',
                       'mass',
                       'att',
                       ])
    for key in list(set(v.keys())):
        v[key] = [x for x in os.listdir('data/'+key) if x.endswith('.csv')]
        v[key] = [x.replace('.csv','') for x in v[key]]
    
    dbx = dict.fromkeys(v['count'])
    for key in v['count']:
        dbx[key] = pd.read_csv(str('data/count/' + key + '.csv'))
        
    
    dbm = dict.fromkeys(v['mass'])
    for key in v['mass']:
        dbm[key] = pd.read_csv(str('data/mass/' + key + '.csv'))
        
    dba = dict.fromkeys(v['att'])
    for key in v['att']:
        dba[key] = pd.read_csv(str('data/att/' + key + '.csv'))

    return dbx, dbm, dba


def CleanWeightClassData(dba): # not used in stocks.py but keeping it for manual output

    dba['PC_wclass']['Voertuigtype'] = 'Personenauto'
    
    wclass = ['CC_wclass','PC_wclass']
    
    #drop rows with nans or totals 
    for i in wclass:
        dba[i] = dba[i].dropna(subset=['Waarde'])
        dba[i] = dba[i][~dba[i]['Onderwerp'].str.contains('otaal')]
        dba[i] = dba[i][~dba[i]['Voertuigtype'].str.contains('otaal')]
        dba[i].loc[:,'Onderwerp'] = dba[i]['Onderwerp'].str.replace(' ','')
        dba[i].loc[:,'Onderwerp'] = dba[i]['Onderwerp'].str.replace('kg','')
    
    # clean personal vehicles
    dba['PC_wclass'].loc[dba['PC_wclass']['Onderwerp']
                    .str.contains('2451enmeer'), 'Onderwerp'] = '2451-3500'
    
    # clean company vehicles
    dba['CC_wclass'].loc[dba['CC_wclass']['Onderwerp']
                    .str.contains('30000enmeer'), 'Onderwerp'] = '300000-50000'
    dba['CC_wclass'].loc[dba['CC_wclass']['Onderwerp']
                    .str.contains('Leeggewichtonbekend'), 'Onderwerp'] = '100000-10001'
    dba['CC_wclass'].loc[dba['CC_wclass']['Onderwerp']
                    .str.contains('<500'), 'Onderwerp'] = '0-500'
    
    # concatenate dataframes and drop old ones
    dba['Wclass'] = pd.concat([dba['CC_wclass'], dba['PC_wclass']], 
                                   ignore_index=True,
                                   sort=False)
    del dba['CC_wclass']
    del dba['PC_wclass']
    
    # calculate middle weight
    dba['Wclass']['unitweight'] = [np.mean(list(map(int, x.split('-')))) 
                               for x in 
                               dba['Wclass']['Onderwerp']]
    dba['Wclass']['Weight'] = dba['Wclass']['unitweight']*dba['Wclass']['Waarde']

    return dba


def CarAvgWeight(dba): # not used in stocks.py but keeping it for manual output
    # aggregate data for vehicle types
    CavW = pd.pivot_table(dba['Wclass'], 
                      index = ['Voertuigtype'], 
                      values = ['Waarde','Weight'],
                      aggfunc = {'Waarde' : np.sum,
                                 'Weight' : np.sum}
                      )
    # calculate avg weight per vehicle type
    CavW['MeanKG'] = CavW['Weight'] / CavW['Waarde']
    
    return CavW['MeanKG']

#%%
###  visualisation plots plotting defs
def PrepMat(mat, materials, vehicles, classes):
    
    for each in [materials['include'], 
                 vehicles['include'],
                 classes['include'],
                 ]:
        each = StrToList(each)
    
    ### filter for selected materials
    if materials['include'] == ['All']: pass
    else: mat = mat.loc[mat['Material'].isin(materials['include'])]
    
    ### exlcude or include certain classes. be sane plz, can be mutually exclusive
    if classes['include'] == ['All']:
        classes['include'] = list(set(mat['Class']))
    mat = mat.loc[mat['Class'].isin(classes['include'])]
    if classes['exclude'] == [None]: pass
    else: mat = mat.loc[~(mat['Class'].isin(classes['exclude']))]
    
    ### exlcude or include certain vehicles. be sane plz, can be mutually exclusive
    if vehicles['include'] == ['All']:
        vehicles['include'] = [x.replace('.csv','') for x in os.listdir('data/mass/') if x.endswith('.csv')]
    mat = mat.loc[mat['Vehicle'].isin(vehicles['include'])]
    if vehicles['exclude'] == [None]: pass
    else: mat = mat.loc[~(mat['Vehicle'].isin(vehicles['exclude']))]
    
    return mat

def PlotMass2Dim(
               mat, 
               Dim=['Material', 'Vehicle'],
               materials = {'include' : ['All'],
                            'exclude' : [None],
                            },
               vehicles = {'include' : ['All'],
                           'exclude' : [None],
                           },
               classes = {'include' : ['All'],
                          'exclude' : [None],
                          },
               exportpdf=False,  
               ):
    
    ### prepare mat df according to selection criteria
    mat = PrepMat(mat, materials, vehicles, classes)
        
    ### combine masses
    mat = mat.groupby(['Year',Dim[0],Dim[1]]).sum().reset_index(drop=False)
    
    ### allow sorting by material and vehicle in plot. mat[mat['Year']==max(mat['Year'])]
    for i in range(len(Dim)):
        mat[str(Dim[i]+'Sum')] = mat[Dim[i]].map(dict(mat.groupby(by=Dim[i]).sum()['Mass']))
    mat = mat.sort_values([str(Dim[0]+'Sum'), str(Dim[1]+'Sum')], ascending=[True, True])
    
    ### plot the shit
    fig = px.area(mat, x = 'Year', y = 'Mass', 
                  color = Dim[0], 
                  width = 800,
                  height = 500,
                  line_group = Dim[1],
                  #category_orders = {'idx' : list(mat.index)}
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    
    fig.show()
    if exportpdf is True:
        fig.write_image(str('figures/Sarea'
                            +''.join(map(str, Dim))
                            +'M-'.join(map(str, materials['include']))[0:25]
                            +'V-'.join(map(str, vehicles['include']))[0:25]
                            +'.pdf'))

    
def PlotMass1Dim(
               mat,
               Dim='Vehicle',
               materials = {'include' : ['All'],
                            'exclude' : [None],
                            },
               vehicles = {'include' : ['All'],
                           'exclude' : [None],
                           },
               classes = {'include' : ['All'],
                          'exclude' : [None],
                          },
               exportpdf=False, 
               category_orders=False
               ):
    
    ### prepare mat df according to selection criteria
    mat = PrepMat(mat, materials, vehicles, classes)
    
    ### combine masses
    mat = mat.groupby(['Year', Dim]).sum().reset_index(drop=False)
    
    ### allow sorting by material and vehicle in plot. mat[mat['Year']==max(mat['Year'])]
    mat[str(Dim+'Sum')] = mat[Dim].map(dict(mat.groupby(by=Dim)
                                           .sum()['Mass']
                                           .sort_values(ascending=False)))
    mat = mat.sort_values(str(Dim+'Sum'), ascending=True)
    
    ### plot the shit
    fig = px.area(mat, x = 'Year', y = 'Mass', 
                  color = Dim, 
                  width = 800,
                  height = 500,
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    
    fig.show()
    if exportpdf is True:
        fig.write_image(str('figures/Sarea'
                            +''.join(map(str, Dim))
                            +'M-'.join(map(str, materials['include']))[0:25]
                            +'V-'.join(map(str, vehicles['include']))[0:25]
                            +'.pdf'))
        
def TreeChart(mat,
              slx = [],
              cat = [],
              year = 2017,
              scale = None,
              exportpdf=False,
              lim=8,
              values=False,
              labels=False,
              ):
    import plotly.graph_objects as go
    import colorlover as cl
    import squarify
    
    if lim>8: lim=8

    if values is False and labels is False:
        values, labels = SelectVehicleYear(mat, slx, cat, year, lim)
    
    fig = go.Figure()
    
    x = 0.
    y = 0.
    width = 1.
    height = 1.

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    colorlover = list(reversed(cl.scales[str(len(values))]['seq']['Blues']))
    shapes = []
    annotations = []
    counter = 0

    for r, val, color, label in zip(rects, values, colorlover, labels):
        shapes.append(
            dict(
                type = 'rect',
                x0 = r['x'],
                y0 = r['y'],
                x1 = r['x']+r['dx'],
                y1 = r['y']+r['dy'],
                line = dict( width = 2 ),
                fillcolor = color
            )
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = (label[:8] + '..') if len(label) > 9 else label,
                showarrow = False,
                bgcolor='#ffffff',
                opacity=0.7,
                borderpad=4,
            )
        )

    # For hover text
    fig.add_trace(go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects ],
        y = [ r['y']+(r['dy']/2) for r in rects ],
        text = [ str(str(v)+' kg') for v in values ],
        mode = 'text',
    ))
    if scale == None:
        if len(slx)==2:
            scale = (mat.loc[mat['Year']==2011].loc[mat[slx[0]].isin(slx[1]), 'Mass'].sum()/300)**0.5
        if len(cat)==1:
            scale = (mat.loc[mat['Year']==2011, 'Mass'].sum()/300)**0.5

    fig.update_layout(
        height=scale,
        width=scale,
        xaxis=dict(showgrid=False,zeroline=True,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=True,showticklabels=False),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
    )
    
    fig.show()
    if exportpdf is True:
        fig.write_image(str('figures/Tree'\
                            +(cat[0] if len(cat) > 0 else '')\
                            +(slx[0] if len(slx) > 0 else '')\
                            +(''.join(map(str, slx[1])) if len(slx) > 0 else '')\
                            +str(year)\
                            +'.pdf')) 
    
### Pie or Donut chart
def DonutChart(mat,
               slx = [],
               cat = [],
               year = 2017,
               lim = 8,
               exportpdf=False,
               ):
    
    values, labels = SelectVehicleYear(mat, slx, cat, year, lim)

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, 
                                 values=values, 
                                 hole=.6)])

    fig.show()
    if exportpdf is True:
        fig.write_image(str('figures/Donut'\
                            +(cat[0] if len(cat) > 0 else '')\
                            +(slx[0] if len(slx) > 0 else '')\
                            +(''.join(map(str, slx[1])) if len(slx) > 0 else '')\
                            +str(year)\
                            +'.pdf'))        


### for plots, groupby materials for particular vehicle in particular year
def SelectVehicleYear(mat, slx=[], cat=[], year=2017, lim=8):
    if len(cat)==0 and len(slx)==0:
        print('Please input either slx OR cat')
        
    if len(slx) == 2:
        val = mat.loc[mat['Year']==year]\
                 .loc[mat[slx[0]].isin(slx[1])]\
                 .groupby('Material').sum()\
                 .loc[:, 'Mass']\
                 .sort_values(ascending=False)
        if len(val)>lim:
            v = val[:lim]
            v['Other'] = sum(val[lim:])
        else: 
            v=val
        
    if len(cat) == 1:
        val = mat.loc[mat['Year']==year]\
                 .groupby(cat).sum()\
                 .loc[:, 'Mass']\
                 .sort_values(ascending=False)
        if len(val)>lim:
            v = val[:lim]
            v['Other'] = sum(val[lim:])
        else: 
            v=val

    values = v.astype('int')
    labels = v.index.tolist()
    
    return values, labels

#%% simple defs
    
def StrToList(obj):
    if isinstance(obj,list): pass
    else: list(obj)
    return obj

def ReplaceValues(df, col, oldval, newval):
    df.loc[df.loc[:,col] == oldval, col] = newval
    return df

def StripChars(df, col, chars):
    for char in chars:
        df.loc[:,col] = [i.replace(char,'') for i in df[col]]
    return df

def TimePrint(start):
    print(round(time.time()-start,5),'s have elapsed, all is good!')

def SetWeightManual(dbx, slx):
    for i in list(slx.keys()):
        dbx.loc[dbx['Vehicle']==i, 'Onderwerp'] = slx[i]
    return dbx
