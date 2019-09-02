#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 11:59:11 2019
# @author: jvanderzaag

import pandas as pd 
import numpy as np
import functions as func
import time; start = time.time()


#%%

DATA = {
        'lv_x' : 'LV_x.csv',        # aircraft, count
        'lv_kg' : 'LV_kg.csv',      # aircraft, weight
        'sv_x_gt' : 'SV_x_gt.csv',  # seagoing vessel, count, gross tonnage 
        'iv_x_gt' : 'IV_x_gt.csv',  # inland vessel, count, gross tonnage 
        "pv_x_kg" : "PV_x_kg.csv",  # personal vehicles, count, weight class
        'pv_x_bs' : 'PV_x_bs.csv',  # personal vehicles, count, fuel type
        "bv_x_kg" : "BV_x_kg.csv",  # company vehilces, count, weight class
        "bv_x_bs" : "BV_x_bs.csv",  # company vehilces, count, fuel type
        "mf_x_cc" : "MF_x_cc.csv",  # motorbikes, count, cc
        "bf_x_bs" : "BF_x_bs.csv",  # mopeds, count, fuel type
        }

db = dict.fromkeys(list(DATA.keys()))
for key in list(DATA.keys()):
    db[key] = pd.read_csv(str("data/"+DATA[key]),sep=";")

db = func.CleanDataframes(db)    
db = func.AddVehicleTypeColumn(db)

Vtypes = func.MakeVehicleTypeList(DATA, db)


  
print(round(time.time()-start,5),'s have elapsed, all is good!'); del start