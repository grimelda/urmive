
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
            ### uncomment if you want to recalculate (takes 20 minutes!)
#            recalc_mass=True,
#            recalc_pw=True,
            )

stocks.TimePrint(start)
#%%

PW='BAU'

#%%

stocks.PlotMass1Dim(mat[mat['PW']==PW], 
                    Dim=['Class'], 
                    materials = dict(include = ['All'], exclude = [None]),
                    matgroup = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),
                    domain = dict(include = ['All'], exclude = [None]),
#                     exportpdf = True,
                    ylabel='Stock Mass [tons]',
                    flow='Mstock',
                    pathway=PW,
#                     filetype='png',
                    w=1000,
                    h=400,
                    )

#%%

stocks.PlotMass1Dim(mat.loc[mat['PW']==PW],#.loc[mat['Material Group']=='Critical Raw Materials, CRM'], 
                    Dim=['Material Group'], 
                    materials = dict(include = ['All'], exclude = [None]),
                    matgroup = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),
                    domain = dict(include = ['All'], exclude = [None]),
#                     exportpdf = True,
                    ylabel='Stock Inflow [tons/year]',
                    flow='Minflow',
                    pathway=PW,
#                     filetype='png',
#                     category_orders={'Class':['Cars','Seavessels','Inlandvessels','Utilitycars','Bicycles','Freighttrains','Transit','Aircraft'][::-1]}
                    w=1000,
                    h=400,
                    )

#%%

stocks.PlotMass1Dim(mat[mat['PW']==PW], 
                    Dim=['Material Group'],
                    materials = dict(include = ['All'], exclude = [None]),
                    matgroup = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),
                    domain = dict(include = ['All'], exclude = [None]),
#                     exportpdf = True,
                    ylabel='Stock Outflow [tons/year]',
                    flow='Moutflow',
                    pathway=PW,
#                     filetype='png',
#                     category_orders={'Class':['Cars','Seavessels','Inlandvessels','Utilitycars','Bicycles','Freighttrains','Transit','Aircraft'][::-1]}
                    w=1000,
                    h=400,
                    )

#%%