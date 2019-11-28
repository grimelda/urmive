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
hmm = pd.read_csv('data/hist_materials_map.csv', header=0,index_col=None)
mat = pd.merge(mat, hmm, on='Material', how='outer')
mat.at[:, 'Mass'] = mat.loc[:, 'Mass'].multiply(0.001)




#mat.loc[mat['Vehicle']=='hmax','Mass'] * mat.loc[mat['Vehicle']=='hmax','Onderwerp'].astype('int')


#%%
### include and exclude are kinda mutually exclusive, be sane.

stocks.PlotMass2Dim(mat, 
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
                    )
stocks.PlotMass2Dim(mat,
                    Dim=['Vehicle', 'Material'],
                    materials = {'include' : ['All'],
                                 'exclude' : [None],
                                 },
                    vehicles = {'include' : ['All'],
                                'exclude' : [None],
                                },
                    classes = {'include' : ['All'],
                               'exclude' : [None],
                               },                  
                    )
stocks.PlotMass1Dim(mat, 
                    Dim='Class',
                    materials = {'include' : ['All'],
                                 'exclude' : [None],
                                 },
                    vehicles = {'include' : ['All'],
                                'exclude' : [None],
                                },
                    classes = {'include' : ['All'],
                               'exclude' : [None],
                               },                  
                    )

    
#%%
stocks.TimePrint(start); del start