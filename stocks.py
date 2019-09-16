import time; start = time.time()
import functions as func

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

func.PlotMass2D(mat, D=['Material', 'Class'])
func.PlotMass2D(mat, D=['Class', 'Material'])
func.PlotMass1D(mat, D='Class')

    
#%%
func.TimePrint(start); del start