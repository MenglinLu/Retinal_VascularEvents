# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:49:18 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np
import os
import math

data_path = './alldataset_data_pred_risks_morefea.csv'

data = pd.read_csv(data_path)

##划分高中低三组
data_low = data[data['proba_1']<0.25]
data_middle = data.loc[(np.where(data['proba_1']>=0.25,1,0) & np.where(data['proba_1']<=0.75,1,0)==1),:]
data_high = data[data['proba_1']>0.75]

def or_cal(a,b,c,d):
    
    or_res = (a*d)/(b*c*1.0)
    ci_ln_or_lower = np.log(or_res) - 1.96*np.sqrt(1/a+1/b+1/c+1/d)
    ci_ln_or_upper = np.log(or_res) + 1.96*np.sqrt(1/a+1/b+1/c+1/d)
    
    ci_or_lower = np.exp(ci_ln_or_lower)
    ci_or_upper = np.exp(ci_ln_or_upper)
    
    return [or_res, ci_or_lower, ci_or_upper]

a_low_middle = len(data_middle[data_middle['allcause_label']==1])
b_low_middle = len(data_middle[data_middle['allcause_label']==0])
c_low_middle = len(data_low[data_low['allcause_label']==1])
d_low_middle = len(data_low[data_low['allcause_label']==0])

res_low_middle = or_cal(a_low_middle, b_low_middle, c_low_middle, d_low_middle)


a_low_high = len(data_high[data_high['allcause_label']==1])
b_low_high = len(data_high[data_high['allcause_label']==0])
c_low_high = len(data_low[data_low['allcause_label']==1])
d_low_high = len(data_low[data_low['allcause_label']==0])

res_low_high = or_cal(a_low_high, b_low_high, c_low_high, d_low_high)

res_or_3group = pd.DataFrame([res_low_middle, res_low_high], columns=['OR', 'OR_lower', 'OR_upper'])
res_or_3group.to_csv('./OR_3group.csv')

###以下按分位数拆分成十个风险组
qua_1, qua_2, qua_3, qua_4, qua_5, qua_6, qua_7, qua_8, qua_9 = np.quantile(data['proba_1'],0.1), np.quantile(data['proba_1'],0.2), np.quantile(data['proba_1'],0.3), np.quantile(data['proba_1'],0.4), np.quantile(data['proba_1'],0.5), np.quantile(data['proba_1'],0.6), np.quantile(data['proba_1'],0.7), np.quantile(data['proba_1'],0.8), np.quantile(data['proba_1'],0.9)
data_1rd = data[data['proba_1']<qua_1]
data_2nd = data.loc[(np.where(data['proba_1']>=qua_1,1,0) & np.where(data['proba_1']<qua_2,1,0)==1),:]
data_3rd = data.loc[(np.where(data['proba_1']>=qua_2,1,0) & np.where(data['proba_1']<qua_3,1,0)==1),:]
data_4th = data.loc[(np.where(data['proba_1']>=qua_3,1,0) & np.where(data['proba_1']<qua_4,1,0)==1),:]
data_5th = data.loc[(np.where(data['proba_1']>=qua_4,1,0) & np.where(data['proba_1']<qua_5,1,0)==1),:]
data_6th = data.loc[(np.where(data['proba_1']>=qua_5,1,0) & np.where(data['proba_1']<qua_6,1,0)==1),:]
data_7th = data.loc[(np.where(data['proba_1']>=qua_6,1,0) & np.where(data['proba_1']<qua_7,1,0)==1),:]
data_8th = data.loc[(np.where(data['proba_1']>=qua_7,1,0) & np.where(data['proba_1']<qua_8,1,0)==1),:]
data_9th = data.loc[(np.where(data['proba_1']>=qua_8,1,0) & np.where(data['proba_1']<qua_9,1,0)==1),:]
data_10th = data[data['proba_1']>=qua_9]

res_li = []
for data_i in [data_2nd, data_3rd, data_4th, data_5th, data_6th, data_7th, data_8th, data_9th, data_10th]:
    data_i_a = len(data_i[data_i['allcause_label']==1])
    data_i_b = len(data_i[data_i['allcause_label']==0])
    data_i_c = len(data_1rd[data_1rd['allcause_label']==1])
    data_i_d = len(data_1rd[data_1rd['allcause_label']==0])
    
    res_data_i = or_cal(data_i_a, data_i_b, data_i_c, data_i_d)
    res_li.append(res_data_i)

res_or_10group = pd.DataFrame(res_li, columns=['OR', 'OR_lower', 'OR_upper'])
res_or_10group.to_csv('./OR_10group.csv')






    

