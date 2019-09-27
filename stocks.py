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
                    #include=['A330','B747'], #['A330','B747']
                    #exclude=['icecar'], # ['bicycle', 'ebicycle'] ['evcar','icecar']
                    ) # include/exclude are kinda mutually exclusive, be sane.


#%% Plotting

### todo: fix that icecar doesnt have class.... wtf

func.PlotMass2D(mat, D=['Material', 'Vehicle'])
func.PlotMass2D(mat, D=['Vehicle', 'Material'])
func.PlotMass1D(mat, D='Vehicle')

    
#%%
func.TimePrint(start); del start