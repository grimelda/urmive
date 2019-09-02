#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 12:22:44 2019
# @author: jvanderzaag

import pandas as pd
import numpy as np



#%%

def ReplaceValues(df, col, oldval, newval):
    df.loc[df[col] == oldval, col] = newval
    return df

def StripChars(df, col, chars):
    for char in chars:
        df.loc[:,col] = [i.replace(char,'') for i in df[col]]
    return df


def AddVehicleTypeColumn(db):
    db["pv_x_kg"]['Vtype'] = (pd.Series(['Personenauto:']*len(db["pv_x_kg"])) 
                              + db['pv_x_kg']['Onderwerp']
                              )
    db["pv_x_bs"]['Vtype'] = (pd.Series(['Personenauto:']*len(db['pv_x_bs'])) 
                              + db['pv_x_bs']['Onderwerp']
                              )
    db["bv_x_kg"]['Vtype'] = (db["bv_x_kg"]['Voertuigtype'] 
                              + pd.Series([':']*len(db["bv_x_kg"])) 
                              + db["bv_x_kg"]['Onderwerp']
                              )
    db["bv_x_bs"]['Vtype'] = (db["bv_x_bs"]['Voertuigtype'] 
                              + pd.Series([':']*len(db["bv_x_bs"])) 
                              + db["bv_x_bs"]['Onderwerp']
                              )
    db["mf_x_cc"]['Vtype'] = (pd.Series(['Motorfiets:']*len(db["mf_x_cc"])) 
                              + db["mf_x_cc"]['Onderwerp']
                              )
    db["bf_x_bs"]['Vtype'] = (pd.Series(['Bromfiets:']*len(db["bf_x_bs"])) 
                              + db["bf_x_bs"]['Onderwerp']
                              )
    db["lv_x"]['Vtype'] = (pd.Series(['Luchtvaartuig:']*len(db["lv_x"])) 
                           + db["lv_x"]['Onderwerp']
                           )
    db["sv_x_gt"]['Vtype'] = (pd.Series(['Zeevaartuig:']*len(db["sv_x_gt"])) 
                              + db["sv_x_gt"]['Onderwerp']
                              )
    db["iv_x_gt"]['Vtype'] = (pd.Series(['Binnenvaartuig:']*len(db["iv_x_gt"])) 
                              + db["iv_x_gt"]['Onderwerp']
                              )
    
    return db

def CleanDfEurostat(df):
    df = StripChars(df, 
                    'Value', 
                    [" ", ":"],
                    )
    df['Value'] = pd.to_numeric(df['Value'], 
                                errors='coerce')
    
    df = df.rename(columns={
                            'VESSEL':'Onderwerp'
                            })
    return df

def CleanDataframes(db):
    db["pv_x_kg"] = ReplaceValues(db["pv_x_kg"],
                                  'Onderwerp',
                                  '2451 kg en meer',
                                  '2451-3500 kg')
    db["bv_x_kg"] = ReplaceValues(db["bv_x_kg"],
                                  'Onderwerp',
                                  '30 000 kg en meer',
                                  '30 000 - 50 000 kg')
    db["bv_x_kg"] = ReplaceValues(db["bv_x_kg"],
                                  'Onderwerp',
                                  'Leeggewicht onbekend',
                                  '10 000 - 10 001kg')
    
    db['iv_x_gt'] = CleanDfEurostat(db['iv_x_gt'])
    db['sv_x_gt'] = CleanDfEurostat(db['sv_x_gt'])
    '''
    db['sv_x_gt'] = StripChars(db['sv_x_gt'], 
                               'Value', 
                               [" ", ":"],
                               )
    db['sv_x_gt']['Value'] = pd.to_numeric(db['sv_x_gt']['Value'], 
                                           errors='coerce')
    
    db['sv_x_gt'] = db['sv_x_gt'].rename(columns={
                                                  'VESSEL':'Onderwerp'
                                                  })
    '''
    return db


def MakeVehicleTypeList(DATA, db):
    Vtypes = []
    for key in list(DATA.keys()):
        if key in [
                   'lv_kg',
                   'sv_x_gt',
                   ]: continue # handle exceptions
        newlist = list(db[key]['Vtype'])
        Vtypes = Vtypes + newlist
    Vtypes = [x for x in list(set(Vtypes))[1:] if "otaal" not in x]
    return Vtypes