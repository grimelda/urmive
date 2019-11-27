import os
import pandas as pd
import pathways as pw
import plotly.express as px
import stocks

#RA = pw.RunPW('RA', figs=True, wlo='laag')

#%%

### Read or create VIOS dataset (Vehicle In Out Stock)
if os.path.exists('data/VIOS.csv'):
    dbx = pd.read_csv('data/VIOS.csv', index_col=0)
else:
    PWs=['BAU', 'ST', 'RA', 'RC', 'TF']

    dbx = pd.DataFrame()
    for PW in PWs:
        df = pw.RunPW(PW, figs=True, wlo='laag')
        dbx = pd.concat([dbx, df], ignore_index=True, sort=False)
    
    dbx.to_csv('data/VIOS.csv')
    
#%%
    
### prepare mass dataframe
path = 'data/fmass/'
cols = ['Material Group', 'Material', 'Unitmass', 'Vehicle', 'Component']
dbm = pd.DataFrame()
for i in os.listdir(path):
    df = pd.read_csv(path+i)
    df = df.loc[df['Unitmass']>0]
    if 'Component' not in df.columns: df['Component'] = None
    dbm = pd.concat([dbm, df[cols]], ignore_index=True, sort=False)
    
#%%

mat = pd.DataFrame()
dbx = dbx.rename(columns={'Mat':'Vtype'})
dbm = dbm.rename(columns={'Vehicle':'Vtype'})
for v in dbm['Vtype'].unique():
    df = pd.merge(dbx.loc[dbx['Vtype']==v],
                  dbm[dbm['Vtype']==v],
                  on='Vtype',
                  how='outer',
                  )
    df['Mstock'] = df['Stock'] * df['Unitmass'] * df['Vmass']
    df['Minflow'] = df['Inflow'] * df['Unitmass'] * df['Vmass']
    df['Moutflow'] = df['Outflow'] * df['Unitmass'] * df['Vmass']
    mat = pd.concat([mat, df], ignore_index=True, sort=False)
    
#%%
df = mat.loc[mat['PW']=='BAU'].loc[mat['Class'].isin(['Cars','Bicycles','Transit', 'Airplanes'])]
#fig = px.area(df, x = 'Year', y = 'Stock', 
#              color = 'Class', 
#              line_group = 'Vehicle',
#              ).update_layout(yaxis_title="Mass of vehicles in NL",
#                              legend=dict(\
#                                          y=0.5, 
#                                          traceorder='reversed', 
#                                          font_size=10,
#                                          ))
#fig.show()

stocks.PlotMass2Dim(df, 
                    Dim=['Class', 'Material'], 
                    materials = dict(include = ['All'], exclude = [None]),
                    vehicles = dict(include = ['All'], exclude = [None]),
                    classes = dict(include = ['All'], exclude = [None]),#['Cars','Bicycles','Transit', 'Airplanes']),
#                     exportpdf = True,
                    flow='Stock',
                    )

