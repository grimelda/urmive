import os
import pandas as pd
import pathways as pw
import plotly.express as px
import stocks

#RA = pw.RunPW('RA', figs=True, wlo='laag')

#%%
if not os.path.exists('data/PW5.csv'):
    mat = pd.read_csv('data/PW5.csv', index_col=0)
else:
    ### Read or create VIOS dataset (Vehicle In Out Stock)
    if not os.path.exists('data/VIOS.csv'):
        dbx = pd.read_csv('data/VIOS.csv', index_col=0)
    else:
        PWs=['BAU']#, 'ST', 'RA', 'RC', 'TF']
    
        dbx = pd.DataFrame()
        for PW in PWs:
            print('\nPathway: '+PW)
            df = pw.RunPW(PW, figs=False, wlo='laag')
            dbx = pd.concat([dbx, df], ignore_index=True, sort=False)
        
        dbx = stocks.FixMatColumnTypes(dbx,
                                coltypes = {'Year':'int16',
                                            'Vehicle':'str',
                                            'Stock':'float32',
                                            'Inflow':'float32',
                                            'Outflow':'float32',
                                            'Class':'str',
                                            'Vehiclename':'str',
                                            'Vmass':'float32',
                                            'lifespan':'float16',
                                            'PW':'str',
                                            'wlo':'str',
                                            })
        dbx.to_csv('data/VIOS.csv')

                
    ### prepare mass dataframe
    path = 'data/fmass/'
    cols = ['Material Group', 'Material', 'Unitmass', 'Vehicle', 'Component']
    dbm = pd.DataFrame()
    for i in os.listdir(path):
        df = pd.read_csv(path+i)
#        df = df.loc[df['Unitmass']>0]
        if 'Component' not in df.columns: df['Component'] = ''
        dbm = pd.concat([dbm, df[cols]], ignore_index=True, sort=False)
    dbm = dbm[dbm['Unitmass']>0]
    dbm = stocks.FixMatColumnTypes(dbm,
                            coltypes = {'Material Group':'str',
                                        'Material':'str',
                                        'Unitmass':'float64',
                                        'Vehicle':'str',
                                        'Component':'str',
                                        })
        
    mat = pd.DataFrame()
    for v in dbm['Vehicle'].unique():
        df = pd.merge(dbx.loc[dbx['Vehicle']==v],
                      dbm.loc[dbm['Vehicle']==v],
                      on='Vehicle',
                      how='outer',
                      )
        df['Mstock'] = df['Stock'] * df['Unitmass'] * df['Vmass']
        df['Minflow'] = df['Inflow'] * df['Unitmass'] * df['Vmass']
        df['Moutflow'] = df['Outflow'] * df['Unitmass'] * df['Vmass']
        mat = pd.concat([mat, df], ignore_index=True, sort=False)
    
    mat.to_csv('data/PW5.csv')


#%%
df = mat.loc[mat['PW']=='BAU'].copy(deep=True)#.loc[mat['Class'].isin(['Cars','Bicycles','Transit', 'Airplanes'])]
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
                    flow='Mstock',
                    )

