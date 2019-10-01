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
                  Mat=['Copper'],
                  #include=['A330','B747'], #['A330','B747']
                  #exclude=['icecar'], # ['bicycle', 'ebicycle'] ['evcar','icecar']
                  )
func.PlotMass2Dim(mat,
                  Dim=['Class', 'Material'],
                  #include=['A330','B747'], #['A330','B747']
                  #exclude=['icecar'], # ['bicycle', 'ebicycle'] ['evcar','icecar']
                  )
func.PlotMass1Dim(mat, 
                  Dim='Vehicle',
                  #include=['A330','B747'], #['A330','B747']
                  #exclude=['icecar'], # ['bicycle', 'ebicycle'] ['evcar','icecar']
                  )

    
#%%
func.TimePrint(start); del start