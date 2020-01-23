import time; start = time.time()
import stocks
import pandas as pd

#%%
### calc materials from vehicle units
(
 dbx,
 dbm,
 dba,
 ) = stocks.ReadData()

(
 dbx,
 Vtypes,
 ) = stocks.UnifyCountData(dbx, startyear=2000, endyear=2017)

(
 dbm
 ) = stocks.UnifyMassData(dbx, dbm)

mat = stocks.CalcMass(dbx, dbm)


#%%

stocks.PlotMass1Dim(mat, 
                    Dim=['Vehicle'],
                    materials = {'include' : ['All'],
                                 'exclude' : [None],
                                 },
                    matgroup = {'include' : ['Critical Raw Materials, CRM'],
                                'exclude' : [None],
                                },
                    vehicles = {'include' : ['All'],
                                'exclude' : [None],
                                },
                    classes = {'include' : ['All'], #'Cars','Bicycles','Transit','Aircraft'
                               'exclude' : [None],
                               },
                    domain = {'include' : ['All'], 
                              'exclude' : [None],
                              },
                    exportpdf=True,
#                    category_orders={'Class':['Inlandvessels', 'Seavessels', 'Cars','Utilitycars','Bicycles','Transit','Aircraft'][::-1]},
                    w=800,
                    h=400,
                    )



    
#%%
stocks.TimePrint(start); del start