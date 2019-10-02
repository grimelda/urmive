#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 12:22:44 2019
# @author: jvanderzaag

import os
import time
import numpy as np
import pandas as pd
import plotly.express as px

#%% this is useful to have at the top

def RemapVehicles(dbx):
    
    MAP = {
           'Aircraft:Ballonvaartuigen' : 'balloon',
           'Aircraft:||4-motorig' : 'B747',
           'Aircraft:||2-motorig' : 'A330',
           'Personalcar:Elektriciteit' : 'evcar',
           'Personalcar:Benzine' : 'icecar',
           'Personalcar:CNG' : 'icecar',
           'Personalcar:§§iesel' : 'icecar',
           'Personalcar:LPG' : 'icecar',
           'Personalcar:Benzine' : 'icecar',
           'Personalcar:Overig/Onbekend' : 'icecar',
           #'Bestelauto:Elektriciteit' : 'evvan',
           #'Bestelauto:Benzine' : 'icevan',
           #'Bestelauto:CNG' : 'icevan',
           #'Bestelauto:Diesel' : 'icevan',
           #'Bestelauto:LPG' : 'icevan',
           #'Bestelauto:Benzine' : 'icevan',
           #'Bestelauto:Overig/Onbekend' : 'icevan',
           'Bestelauto:1249':'icevan',
           'Bestelauto:1749':'icevan',
           'Bestelauto:2249':'icevan',
           'Bestelauto:250':'icevan',
           'Bestelauto:2749':'icevan',
           'Bestelauto:3249':'icevan',
           'Bestelauto:749':'icevan',
           'Trekker voor oplegger:10249':'lorry28t',
           'Trekker voor oplegger:10749':'lorry28t',
           'Trekker voor oplegger:11249':'lorry28t',
           'Trekker voor oplegger:11749':'lorry40t',
           'Trekker voor oplegger:12249':'lorry40t',
           'Trekker voor oplegger:1249':'lorry16t',
           'Trekker voor oplegger:12749':'lorry40t',
           'Trekker voor oplegger:1749':'lorry16t',
           'Trekker voor oplegger:2249':'lorry16t',
           'Trekker voor oplegger:250':'lorry16t',
           'Trekker voor oplegger:2749':'lorry16t',
           'Trekker voor oplegger:3249':'lorry16t',
           'Trekker voor oplegger:3749':'lorry16t',
           'Trekker voor oplegger:4249':'lorry16t',
           'Trekker voor oplegger:4749':'lorry16t',
           'Trekker voor oplegger:5249':'lorry16t',
           'Trekker voor oplegger:5749':'lorry16t',
           'Trekker voor oplegger:6249':'lorry16t',
           'Trekker voor oplegger:6749':'lorry16t',
           'Trekker voor oplegger:7249':'lorry16t',
           'Trekker voor oplegger:749':'lorry16t',
           'Trekker voor oplegger:7749':'lorry16t',
           'Trekker voor oplegger:8249':'lorry28t',
           'Trekker voor oplegger:8749':'lorry28t',
           'Trekker voor oplegger:9249':'lorry28t',
           'Trekker voor oplegger:9749':'lorry28t',
           'Inlandvessel:1000000-2000000' : 'hmax',
           'Inlandvessel:2000000-3000000' : 'hmax',
           'Inlandvessel:3000000-4000000' : 'hmax',
           'Inlandvessel:500000-1000000' : 'hmax',
           }
    
    for key in list(MAP.keys()):
        dbx.loc[:,'Vehicle'] = dbx.loc[:,'Vehicle'].replace(key,MAP[key])
    
    return dbx

#%% high level defs

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
    
    ### for each vehicle type which has weight info in count table:
    mat = CalcWeightFromCount(mat, slx=[
                                        'icevan',
                                        'lorry16t',
                                        'lorry28t',
                                        'lorry40t',
                                        'hmax',
                                        ])
        
    return mat

def CalcWeightFromCount(mat, slx):
    for i in slx:
        mat.loc[mat['Vehicle']==i, 'Mass'] = (mat.loc[mat['Vehicle']==i,'Mass'] 
                                              * 
                                              mat.loc[mat['Vehicle']==i,'Onderwerp'].astype('int')
                                              )
    return mat


def UnifyMassData(dbm, dbx):
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
    db['seaships']['Vtype'] = (pd.Series(['Seavessel:']*len(db['seaships'])) 
                              + db['seaships']['Onderwerp']
                              )
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



#%% visualisation plots plotting defs
    
def PrepMat(mat, materials, vehicles, classes):
    
    for each in [materials['include'], 
                 vehicles['include'],
                 classes['include'],
                 ]:
        each = StrToList(each)
    
    ### filter for selected materials
    if materials['include'] == 'All': pass
    else: mat = mat.loc[mat['Material'].isin(materials['include'])]
    
    ### exlcude or include certain classes. be sane plz, can be mutually exclusive
    if classes['include'] == 'All':
        classes['include'] = list(set(mat['Class']))
    mat = mat.loc[mat['Class'].isin(classes['include'])]
    if classes['exclude'] is None: pass
    else: mat = mat.loc[~(mat['Class'].isin(classes['exclude']))]
    
    ### exlcude or include certain vehicles. be sane plz, can be mutually exclusive
    if vehicles['include'] == 'All':
        vehicles['include'] = [x.replace('.csv','') for x in os.listdir('data/mass/') if x.endswith('.csv')]
    mat = mat.loc[mat['Vehicle'].isin(vehicles['include'])]
    if vehicles['exclude'] is None: pass
    else: mat = mat.loc[~(mat['Vehicle'].isin(vehicles['exclude']))]
    
    return mat

    
def PlotMass2Dim(
               mat, 
               Dim=['Material', 'Vehicle'],
               materials = {'include' : 'All',
                            'exclude' : None,
                            },
               vehicles = {'include' : 'All',
                           'exclude' : None,
                           },
               classes = {'include' : 'All',
                          'exclude' : None,
                          },
               ):
    
    ### prepare mat df according to selection criteria
    mat = PrepMat(mat, materials, vehicles, classes)
        
    ### combine masses
    mat = mat.groupby(['Year',Dim[0],Dim[1]]).sum().reset_index(drop=False)
    
    ### allow sorting by material and vehicle in plot. mat[mat['Year']==max(mat['Year'])]
    for i in range(len(Dim)):
        mat[str(Dim[i]+'Sum')] = mat[Dim[i]].map(dict(mat.groupby(by=Dim[i]).sum()['Mass']))
    mat = mat.sort_values([str(Dim[0]+'Sum'), str(Dim[1]+'Sum')], ascending=[True, False])
    
    ### plot the shit
    fig = px.area(mat, x = 'Year', y = 'Mass', 
                  color = Dim[0], 
                  line_group = Dim[1],
                  #category_orders = {'idx' : list(mat.index)}
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    if materials['include'] == 'All':
        materials['include'] = ['All']
    
    if vehicles['include'] == 'All':
        vehicles['include'] = ['All']    
    
    fig.show()
    fig.write_image(str('figures/Mass'
                        +''.join(map(str, Dim))
                        +'M-'.join(map(str, materials['include']))[0:25]
                        +'V-'.join(map(str, vehicles['include']))[0:25]
                        +'.pdf'))

    
def PlotMass1Dim(
               mat,
               Dim='Vehicle',
               materials = {'include' : 'All',
                            'exclude' : None,
                            },
               vehicles = {'include' : 'All',
                           'exclude' : None,
                           },
               classes = {'include' : 'All',
                          'exclude' : None,
                          },
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
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    if materials['include'] == 'All':
        materials['include'] = ['All']
    
    if vehicles['include'] == 'All':
        vehicles['include'] = ['All']    
    
    fig.show()
    fig.write_image(str('figures/Mass'
                        +''.join(map(str, Dim))
                        +'M-'.join(map(str, materials['include']))[0:25]
                        +'V-'.join(map(str, vehicles['include']))[0:25]
                        +'.pdf'))
    

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

#%% i might ever need these snippets


    