#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 11:59:11 2019
# @author: jvanderzaag
import time; start = time.time()
import functions as func


import os
import pandas as pd 
import numpy as np
import plotly.express as px


#%%
(
 dbx,
 dbm,
 dba,
 ) = func.ReadData()

#%%
(
 dbx,
 Vtypes,
 ) = func.UnifyCountData(dbx)

#%%
(
 dbm
 ) = func.UnifyMassData(dbm, dbx)

#%%
mat = func.CalcMass(dbx, dbm)

#%% Plotting
    
func.PlotMaterialVehicle(mat) #, exclude=['bicycle'], include=['A330','B747']
                         
func.PlotVehicleMass(mat)#, exclude=['bicycle'], include=['A330','B747']
    
#%%
print(round(time.time()-start,5),'s have elapsed, all is good!'); del start