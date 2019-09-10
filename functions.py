#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 12:22:44 2019
# @author: jvanderzaag

import os
import pandas as pd
import numpy as np
import plotly.express as px


#%% high level defs

def UnifyCountData(dbx):
    
    dbx = CleanDataframes(dbx)    
    dbx = AddVehicleTypeColumn(dbx)
    Vtypes = MakeVehicleTypeList(dbx)
    dbx = CombineDataframes(dbx)
    dbx = dbx.dropna(subset = ['Waarde'])
    dbx = dbx.rename(columns = {
                                'Vtype' : 'Vehicle',
                                'Waarde' : 'Value',
                                'Perioden' : 'Year',
                                })
    dbx = RemapVehicles(dbx)
    return dbx, Vtypes



#%% complexer defs


    
def RemapVehicles(dbx):
    
    MAP = {
           'Aircraft:||4-motorig' : 'B747',
           'Aircraft:||2-motorig' : 'A330',
           'Personalcar:Elektriciteit' : 'carev',
           }
    
    for key in list(MAP.keys()):
        dbx.loc[:,'Vehicle'] = dbx['Vehicle'].replace(key,MAP[key])
    
    return dbx
    
def CalcMass(
             dbx, 
             dbm,
             ):
    missing = list(set(dbx['Vehicle']) - set(dbm.keys()))
    if len(missing) > 0:
        print('No material data was found for',len(missing),'vehicle types...\n')
    
    mat = pd.DataFrame() 
    for key in list(dbm.keys()):
        df = dbx[dbx['Vehicle'] == key].merge(
                                              dbm[key], 
                                              on='Vehicle', 
                                              how='outer',
                                              )
        df['Mass'] = df['Value'] * df['Unitmass']
        
        mat = pd.concat([mat, df], ignore_index=True, sort=False)
        
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
                           [df, dbx[dbx['Vehicle'] == key]], 
                           ignore_index=True, 
                           sort=False,
                           )
    ma = dict()
    for key in veh:
        if len(dbx[dbx['Vehicle'] == key]) == 0: continue
        else:
            ma[key] = dbm[key].loc[:,['Material','Unitmass']]#.set_index('material')
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


def CombineDataframes(dbx):

    count = [x.replace('.csv','') for x in os.listdir('data/count/') if x.endswith('.csv')]
    #count = [x. for x in count]
    DBX = pd.DataFrame()
    for key in count:    
        DBX = pd.concat([DBX, dbx[key].loc[:,['Vtype','Waarde', 'Perioden']]], ignore_index=True, sort=False)
        
    DBX['Waarde'] = pd.to_numeric(DBX['Waarde'], errors='coerce')
    DBX['Perioden'] = pd.to_numeric(DBX['Perioden'], errors='coerce')
    DBX = DBX[(DBX['Perioden'] >= 1996) & (DBX['Perioden'] <= 2017)]
    #DBX = DBX.dropna(subset='Waarde')
    
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
    db['perscars'] = ReplaceValues(db['perscars'],
                                  'Onderwerp',
                                  '2451 kg en meer',
                                  '2451-3500 kg')
    db['compcars'] = ReplaceValues(db['compcars'],
                                  'Onderwerp',
                                  '30 000 kg en meer',
                                  '30 000 - 50 000 kg')
    db['compcars'] = ReplaceValues(db['compcars'],
                                  'Onderwerp',
                                  'Leeggewicht onbekend',
                                  '10 000 - 10 001kg')
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


def CleanWeightClassData(dba):

    dba['PC_wclass']['Voertuigtype'] = 'Personenauto'
    
    #drop rows with nans or totals 
    for i in ['CC_wclass','PC_wclass']:
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


def CarAvgWeight(dba):
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

def PlotVehicleMass(mat, 
                    exclude=[None],
                    include=[x.replace('.csv','') for x in os.listdir('data/mass/') if x.endswith('.csv')]):

    mat = mat[~(mat['Vehicle'].isin(exclude))]
    mat = mat[mat['Vehicle'].isin(include)]
        
    ### combine masses
    mat = mat.groupby(['Year','Vehicle']).sum().reset_index(drop=False)
    
    ### allow sorting by material and vehicle in plot
    mat['VehSum'] = mat['Vehicle'].map(dict(mat[mat['Year']==2017].groupby(by='Vehicle')
                                                .sum()['Mass']
                                                .sort_values(ascending=False)))
    mat = mat.sort_values('VehSum', ascending=True)
    
    ### plot the shit
    fig = px.area(mat, x = 'Year', y = 'Mass', 
                  color = 'Vehicle', 
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    fig.show()
    
    
def PlotMaterialVehicle(mat, 
                        exclude=[None],
                        include=[x.replace('.csv','') for x in os.listdir('data/mass/') if x.endswith('.csv')]):

    mat = mat[~(mat['Vehicle'].isin(exclude))]
    mat = mat[mat['Vehicle'].isin(include)]
        
    ### combine masses
    mat = mat.groupby(['Year','Vehicle','Material']).sum().reset_index(drop=False)
    
    ### allow sorting by material and vehicle in plot
    mat['MatSum'] = mat['Material'].map(dict(mat.groupby(by='Material')
                                                .sum()['Mass']
                                                .sort_values(ascending=False)))
    mat['VehSum'] = mat['Vehicle'].map(dict(mat.groupby(by='Vehicle')
                                               .sum()['Mass']
                                               .sort_values(ascending=False)))
    mat = mat.sort_values(['MatSum', 'VehSum'], ascending=['True', 'True'])
    
    ### plot the shit
    fig = px.area(mat, x = 'Year', y = 'Mass', 
                  color = 'Material', 
                  line_group='Vehicle',
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    fig.show()
    
    
def PlotVehicleMaterial(mat, 
                        exclude=[None],
                        include=[x.replace('.csv','') for x in os.listdir('data/mass/') if x.endswith('.csv')]):

    mat = mat[~(mat['Vehicle'].isin(exclude))]
    mat = mat[mat['Vehicle'].isin(include)]
        
    ### combine masses
    mat = mat.groupby(['Year','Vehicle','Material']).sum().reset_index(drop=False)
    
    ### allow sorting by material and vehicle in plot
    mat['MatSum'] = mat['Material'].map(dict(mat.groupby(by='Material')
                                                .sum()['Mass']
                                                .sort_values(ascending=False)))
    mat['VehSum'] = mat['Vehicle'].map(dict(mat[mat['Year']==2017].groupby(by='Vehicle')
                                               .sum()['Mass']
                                               .sort_values(ascending=False)))
    mat = mat.sort_values(['VehSum', 'MatSum'], ascending=['True', 'True'])
    
    '''
    ### allow sorting by material and vehicle in plot
    mat['VehSum'] = mat['Vehicle'].map(dict(mat[mat['Year']==2017].groupby(by='Vehicle')
                                                .sum()['Mass']
                                                .sort_values(ascending=False)))
    mat = mat.sort_values('VehSum', ascending=True)
    '''
    
    ### plot the shit
    fig = px.area(mat, x = 'Year', y = 'Mass', 
                  color = 'Vehicle', 
                  line_group='Material',
                  ).update_layout(legend=dict(
                                              y=0.5, 
                                              traceorder='reversed', 
                                              font_size=10,
                                              ))
    fig.show()

    

#%% simple defs

def ReplaceValues(df, col, oldval, newval):
    df.loc[df[col] == oldval, col] = newval
    return df


def StripChars(df, col, chars):
    for char in chars:
        df.loc[:,col] = [i.replace(char,'') for i in df[col]]
    return df


#%% i might ever need these snippets


    