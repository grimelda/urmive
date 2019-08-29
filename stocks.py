#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:59:11 2019

@author: jvanderzaag
"""

import pandas as pd 
import functions as func
import time; start = time.time()

DATA = {"cbs_pv" : "Personenauto_s__voertuigkenmerken__regio_s__1_januari_29082019_133946.csv",
        "cbs_bv" : "Bedrijfsvoertuigen__voertuigkenmerken__regio_s__1_januari_29082019_134509.csv",
        "cbs_mf" : "Motorfietsen__voertuigkenmerken__regio_s__1_januari__29082019_120413.csv",
        "cbs_bf" : "Bromfietsen__soort_voertuig__brandstof__bouwjaar__1_januari_29082019_121054.csv",
        "cbs_lv" : "Luchtvloot__omvang_en_samenstelling__31_december_29082019_120654.csv",
        }

MAPS = {"lv_weight" : "cbs_ilt_mapping.tsv",
        }


db = dict.fromkeys(list(DATA.keys()))
for key in list(DATA.keys()):
    db[key] = pd.read_csv(str("data/"+DATA[key]),sep=";")
    


db["cbs_pv"]['Vtype'] = (pd.Series(['Personenauto:']*len(db["cbs_pv"])) 
                         + db["cbs_pv"]['Onderwerp']
                         )
db["cbs_bv"]['Vtype'] = (db["cbs_bv"]['Voertuigtype'] 
                         + pd.Series([':']*len(db["cbs_bv"])) 
                         + db["cbs_pv"]['Onderwerp']
                         )
db["cbs_mf"]['Vtype'] = (pd.Series(['Motorfiets:']*len(db["cbs_mf"])) 
                         + db["cbs_mf"]['Onderwerp']
                         )
db["cbs_bf"]['Vtype'] = (pd.Series(['Bromfiets:']*len(db["cbs_bf"])) 
                         + db["cbs_bf"]['Onderwerp']
                         )
db["cbs_lv"]['Vtype'] = (pd.Series(['Luchtvloot:']*len(db["cbs_lv"])) 
                         + db["cbs_lv"]['Onderwerp']
                         )

Vtypes = []
for key in list(DATA.keys()):
    newlist = list(db[key]['Vtype'])
    Vtypes = Vtypes + newlist
Vtypes = list(set(Vtypes))

  
print(round(time.time()-start,5),'s have elapsed, all is good!'); del start