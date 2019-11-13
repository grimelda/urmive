import time; start = time.time()
import stocks

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


#mat.loc[mat['Vehicle']=='hmax','Mass'] * mat.loc[mat['Vehicle']=='hmax','Onderwerp'].astype('int')


#%%
### include and exclude are kinda mutually exclusive, be sane.
'''
stocks.PlotMass2Dim(mat, 
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
stocks.PlotMass2Dim(mat,
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
stocks.PlotMass1Dim(mat, 
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
'''
    
#%%
stocks.TimePrint(start); del start