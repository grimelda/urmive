import time; start = time.time()
import functions as func

#%%
(
 dbx,
 dbm,
 dba,
 ) = func.ReadData()

#%%
( ### row 207 in combinedataframes passes listlike warning
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
                    ) 

#%% Plotting

### include and exclude are kinda mutually exclusive, be sane.

func.PlotMass2Dim(mat, 
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
                  )
func.PlotMass2Dim(mat,
                  Dim=['Vehicle', 'Material'],
                  materials = {'include' : 'All',
                               'exclude' : None,
                               },
                  vehicles = {'include' : 'All',
                              'exclude' : None,
                              },
                  classes = {'include' : 'All',
                             'exclude' : None,
                             },                  
                  )
func.PlotMass1Dim(mat, 
                  Dim='Class',
                  materials = {'include' : 'All',
                               'exclude' : None,
                               },
                  vehicles = {'include' : 'All',
                              'exclude' : None,
                              },
                  classes = {'include' : 'All',
                             'exclude' : None,
                             },                  
                  )

    
#%%
func.TimePrint(start); del start