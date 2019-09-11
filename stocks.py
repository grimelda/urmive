#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Aug 29 11:59:11 2019
# @author: jvanderzaag
import time; start = time.time()
import functions as func

'''
import os
import pandas as pd 
import numpy as np
import plotly.express as px
'''


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
 ) = func.UnifyCountData(dbx, startyear=2000, endyear=2017)

#%%
(
 dbm
 ) = func.UnifyMassData(dbm, dbx)

#%%
mat = func.CalcMass(dbx, 
                    dbm,
                    #include=['A330','B747'], #['A330','B747']
                    #exclude=['icecar'], # ['bicycle', 'ebicycle'] ['evcar','icecar']
                    ) # include/exclude are kinda mutually exclusive, be sane.


#%% Plotting

func.PlotMass2D(mat, D=['Material', 'Vehicle'])
func.PlotMass2D(mat, D=['Vehicle', 'Material'])
func.PlotMass1D(mat, D='Vehicle')

    
#%%
func.TimePrint(start); del start