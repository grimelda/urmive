import os
import pandas as pd

def UnpivotHandymax():
    path = 'data/unpivot/handymax/'
    tsv = [x.replace('.tsv','') for x in os.listdir(path) if x.endswith('.tsv')]
    
    v = dict.fromkeys(tsv)
    co = pd.DataFrame()
    for key in v:
        v[key] = pd.read_csv(str(path+key+'.tsv'), sep='\t')
        v[key] = v[key].melt(
                             id_vars=['Class',
                                      'Vehicle',
                                      'Component',
                                      ], 
                             var_name='Material',
                             value_name='Unitmass'
                             )
        v[key]['Part'] = key
        co = pd.concat([co,v[key]], ignore_index=True, sort=False)
    
    co = co[~(co['Unitmass'].isna())].reset_index(drop=True)
    
    for vehicle in list(set(co['Vehicle'])):
        co.loc[co['Vehicle'] == vehicle, ['Unitmass','Material', 'Class', 'Component']].to_csv(str('data/mass/'+vehicle+'.csv') ,
                  index=False,
                  header=True,
                  )

def UnpivotCars():
    path = 'data/unpivot/cars/'
    tsv = [x.replace('.tsv','') for x in os.listdir(path) if x.endswith('.tsv')]
    
    v = dict.fromkeys(tsv)
    co = pd.DataFrame()
    for key in v:
        v[key] = pd.read_csv(str(path+key+'.tsv'), sep='\t')
        v[key] = v[key].melt(
                             id_vars=['Material Group','Material/Component','Unit'], 
                             var_name='Component',
                             value_name='Unitmass'
                             )
        v[key]['Part'] = key
        co = pd.concat([co,v[key]], ignore_index=True, sort=False)
        
    co = co[~(co['Unitmass']==0)].reset_index(drop=True)
    
    mapping = pd.read_csv(str(path+'mapping.csv'),header=None).sort_values(by=[0]).reset_index(drop=True)
    maps = dict.fromkeys(mapping[0])
    i=0
    for key in maps.keys():
        maps[key]=mapping[1][i]
        i+=1
        
    co['Material'] = co['Material/Component'].map(maps)
    
    co[~(co['Part']=='icemotor')].to_csv('data/mass/evcar.csv')
    co[~(co['Part']=='evmotor')].to_csv('data/mass/icecar.csv')
    
def UnpivotPleziervaartuigen():
    
    path = 'data/unpivot/pleziervaartuigen/'
    tsv = [x.replace('.tsv','') for x in os.listdir(path) if x.endswith('.tsv')]
    
    v = dict.fromkeys(tsv)
    co = pd.DataFrame()
    for key in v:
        v[key] = pd.read_csv(str(path+key+'.tsv'), sep='\t')
        v[key] = v[key].melt(
                             id_vars=['recreatie vaartuig',
                                      'Materiaal',
                                      'Aantal',
                                      'Gem. lengte',
                                      'Gem. gewicht kg/boot'
                                      ], 
                             var_name='Component',
                             value_name='Gewichtsverhouding'
                             )
        v[key]['Part'] = key
        co = pd.concat([co,v[key]], ignore_index=True, sort=False)
    
    co['Unitmass'] = co['Gem. gewicht kg/boot'] * co['Gewichtsverhouding']
    co = co[~(co['Unitmass']==0)].reset_index(drop=True)
    
    co = co.rename(columns={'recreatie vaartuig':'Vehicle'})
    co.loc[co['Vehicle'] == 'Anders', 'Vehicle'] = 'Anderepleziervaartuigen'
    
    co['Material'] = None
    for row in range(len(co)):
        if co.loc[row,'Component'] == 'Motor':
            co.loc[row,'Material'] = 'Steel' 
        if co.loc[row,'Component'] == 'Kiel':
            co.loc[row,'Material'] = co.loc[row,'Materiaal']
        if co.loc[row,'Component'] == 'Romp':
            co.loc[row,'Material'] = co.loc[row,'Materiaal']
        if co.loc[row,'Component'] == 'Overig':
            co.loc[row,'Material'] = 'Unknown'
    
    co['Class'] = 'Pleziervaartuigen'
    for vehicle in list(set(co['Vehicle'])):
        co.loc[co['Vehicle'] == vehicle, ['Component','Unitmass','Material','Class']].to_csv(str('data/mass/'+vehicle+'.csv') ,
                  index=False,
                  header=True,
                  )
    
    co['Perioden'] = 2014
    co = co.rename(columns={'Vehicle':'Vtype',
                            'Aantal':'Waarde'})
    for vehicle in list(set(co['Vtype'])):
        co.loc[co['Vtype'] == vehicle, ['Perioden','Vtype','Waarde']].to_csv(str('data/count/'+vehicle+'.csv') ,
                  index=False,
                  header=True,
                  )
        
def UnpivotWalpleziervaart():

    path = 'data/unpivot/walpleziervaart/'
    tsv = [x.replace('.tsv','') for x in os.listdir(path) if x.endswith('.tsv')]
    
    v = dict.fromkeys(tsv)
    co = pd.DataFrame()
    for key in v:
        v[key] = pd.read_csv(str(path+key+'.tsv'), sep='\t')
        v[key] = v[key].melt(
                             id_vars=['Vehicle',
                                      'Aantal',
                                      'Totalmass',
                                      ], 
                             var_name='Material',
                             value_name='Gewichtsverhouding'
                             )
        v[key]['Part'] = key
        co = pd.concat([co,v[key]], ignore_index=True, sort=False)
    
    co['Unitmass'] = co['Totalmass'] * co['Gewichtsverhouding']
    co = co[~(co['Unitmass'].isna())].reset_index(drop=True)
        
    co['Class'] = 'Pleziervaartuigen'
    for vehicle in list(set(co['Vehicle'])):
        co.loc[co['Vehicle'] == vehicle, ['Unitmass','Material', 'Class']].to_csv(str('data/mass/'+vehicle+'.csv') ,
                  index=False,
                  header=True,
                  )
    
    co['Perioden'] = 2014
    co = co.rename(columns={'Vehicle':'Vtype',
                            'Aantal':'Waarde'})
    for vehicle in list(set(co['Vtype'])):
        co.loc[co['Vtype'] == vehicle, ['Perioden','Vtype','Waarde']].to_csv(str('data/count/'+vehicle+'.csv') ,
                  index=False,
                  header=True,
                  )



UnpivotCars()
# UnpivotHandymax() ### deprecated, download from drive instead.
UnpivotWalpleziervaart()
UnpivotPleziervaartuigen()
