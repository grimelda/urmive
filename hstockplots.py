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
### include and exclude are kinda mutually exclusive, be sane.
#stocks.PlotMass2Dim(mat[mat['Class']=='Inlandvessels'], 
#                    Dim=['Vehiclenames', 'Material'], 
#                    materials = dict(include = ['All'], exclude = [None]),
#                    vehicles = dict(include = ['All'], exclude = [None]),
#                    classes = dict(include = ['All'], exclude = ['Pleziervaartuigen']),
##                     exportpdf = True,
#                    flow='Value',
#                    )
### include and exclude are kinda mutually exclusive, be sane.
#stocks.PlotMass1Dim(mat[mat['Vehicle']=='icevan2'],
#                    Dim='Vehiclenames',
#                    materials = dict(include = ['All'], exclude = [None]),
#                    vehicles = dict(include = ['All'], exclude = [None]),
#                    classes = dict(include = ['All'], exclude = [None]),#['Pleziervaartuigen', 'Seavessels', 'Inlandvessels']),
#                    exportpdf = True,
#                    flow='Value',
#                    groupnorm='percent',
#                    category_orders={'Vehiclenames':['Bestelauto:10000','Bestelauto:3249','Bestelauto:2749','Bestelauto:2249','Bestelauto:1749','Bestelauto:1249','Bestelauto:749','Bestelauto:250']},
#                    )


#stocks.PlotMass2Dim(mat, 
#                    Dim=['Material', 'Vehicle'], 
#                    materials = {'include' : ['All'],
#                                 'exclude' : [None],
#                                 },
#                    vehicles = {'include' : ['All'],
#                                'exclude' : [None],
#                                },
#                    classes = {'include' : ['All'],
#                               'exclude' : [None],
#                               },                  
#                    )
#stocks.PlotMass2Dim(mat,
#                    Dim=['Vehicle', 'Material'],
#                    materials = {'include' : ['All'],
#                                 'exclude' : [None],
#                                 },
#                    vehicles = {'include' : ['All'],
#                                'exclude' : [None],
#                                },
#                    classes = {'include' : ['All'],
#                               'exclude' : [None],
#                               },                  
#                    )
stocks.PlotMass1Dim(mat, 
                    Dim=['Vehicle'],
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