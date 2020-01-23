
import os
import pandas as pd
import pathways as pw
import plotly.express as px
import stocks
import time; start = time.time()


#%% 
(
 dbx,
 dbm,
 mat,
 ) = pw.PW5(\
            PWs=['BAU', 'ST', 'RA', 'RC', 'TF'],
            recalc_mass=True,
            recalc_pw=True,
            )

stocks.TimePrint(start)
#%%

PW='BAU'

#%%

stocks.PlotMass2Dim(mat[mat['PW']==PW], 
                    Dim=['Material Group', 'Class'], 
                    materials = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),#['Cars','Bicycles','Transit', 'Airplanes']),
#                    exportpdf = True,
                    ylabel='Stock mass [tons]',
                    flow='Mstock',
                    pathway=PW,
                    filetype='png',
                    )
#%%

stocks.PlotMass1Dim(mat[mat['PW']==PW], 
                    Dim=['Material Group'], 
                    materials = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),#['Cars','Bicycles','Transit', 'Airplanes']),
#                     exportpdf = True,
                    ylabel='Inflow [tons/year]',
                    flow='Minflow',
                    pathway=PW,
                    filetype='png',
                    )
#%%

stocks.PlotMass1Dim(mat[mat['PW']==PW], 
                    Dim=['Material Group'], 
                    materials = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),#['Cars','Bicycles','Transit', 'Airplanes']),
#                     exportpdf = True,
                    ylabel='Outflow [tons/year]',
                    flow='Moutflow',
                    pathway=PW,
                    filetype='png',
                    )
#%%