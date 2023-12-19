# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:20:29 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

dir_path = './retinal_allcause_10y/result'
data_all_sex = pd.read_csv('./retinal_allcause_10y/data_allcause_v4_10y.csv')[['Eid','sex']].drop_duplicates()

file_list = ["risk.csv"]

tprs_male = []
aucs_male = []
mean_fpr_male = np.linspace(0, 1, 100)
for file_i in file_list:
    file_i_path = os.path.join(dir_path, file_i, 'test_data_pred_risks.csv')
    data_file_i = pd.read_csv(file_i_path)
    data_file_i.columns = ['0', '1', 'label', 'time', 'Eid']
    data_file_i = pd.merge(left=data_file_i, right=data_all_sex, how='left', on='Eid')
    
    data_file_i_male = data_file_i[data_file_i['sex']==1]
    
    pred_epochi = np.array(data_file_i_male[['0','1']])
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)
    data_file_i_male['proba_0'] = y_proba[:,0]
    data_file_i_male['proba_1'] = y_proba[:,1]
    
    fpr1_male, tpr1_male, threshold1_male = roc_curve(data_file_i_male['label'], data_file_i_male['proba_1'])
    
    auc1_male = auc(fpr1_male, tpr1_male)
    
    interp_tpr_male = interp(mean_fpr_male, fpr1_male, tpr1_male)
    
    interp_tpr_male[0] = 0.0
    tprs_male.append(interp_tpr_male)
    aucs_male.append(auc1_male)
    
mean_tpr_male = np.mean(tprs_male, axis=0)
mean_tpr_male[-1] = 1.0
mean_auc_male = auc(mean_fpr_male, mean_tpr_male)
std_auc_male = np.std(aucs_male)

fig, ax = plt.subplots(figsize=(9,9))
ax.plot(mean_fpr_male, mean_tpr_male, color='#99b9e9',
        label=r'Male (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_male, std_auc_male),
        lw=2, alpha=.3)
std_tpr_male = np.std(tprs_male, axis=0)
tprs_upper_male = np.minimum(mean_tpr_male + std_tpr_male, 1)
tprs_lower_male = np.maximum(mean_tpr_male - std_tpr_male, 0)
ax.fill_between(mean_fpr_male, tprs_lower_male, tprs_upper_male, color='grey', alpha=.2)

tprs_female = []
aucs_female = []
mean_fpr_female = np.linspace(0, 1, 100)
for file_i in file_list:
    file_i_path = os.path.join(dir_path, file_i, 'test_data_pred_risks.csv')
    data_file_i = pd.read_csv(file_i_path)
    data_file_i.columns = ['0', '1', 'label', 'time', 'Eid']
    data_file_i = pd.merge(left=data_file_i, right=data_all_sex, how='left', on='Eid')
    
    data_file_i_female = data_file_i[data_file_i['sex']==0]
    
    pred_epochi = np.array(data_file_i_female[['0','1']])
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)
    data_file_i_female['proba_0'] = y_proba[:,0]
    data_file_i_female['proba_1'] = y_proba[:,1]
    
    fpr1_female, tpr1_female, threshold1_female = roc_curve(data_file_i_female['label'], data_file_i_female['proba_1'])
    
    auc1_female = auc(fpr1_female, tpr1_female)
    
    interp_tpr_female = interp(mean_fpr_female, fpr1_female, tpr1_female)
    
    interp_tpr_female[0] = 0.0
    tprs_female.append(interp_tpr_female)
    aucs_female.append(auc1_female)
    
mean_tpr_female = np.mean(tprs_female, axis=0)
mean_tpr_female[-1] = 1.0
mean_auc_female = auc(mean_fpr_female, mean_tpr_female)
std_auc_female = np.std(aucs_female)

ax.plot(mean_fpr_female, mean_tpr_female, color='#e3716e',
        label=r'Female (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_female, std_auc_female),
        lw=2, alpha=.3)
std_tpr_female = np.std(tprs_female, axis=0)
tprs_upper_female = np.minimum(mean_tpr_female + std_tpr_female, 1)
tprs_lower_female = np.maximum(mean_tpr_female - std_tpr_female, 0)
ax.fill_between(mean_fpr_female, tprs_lower_female, tprs_upper_female, color='grey', alpha=.2)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.set_title("Sex-specific ROC curve of all-cause event", fontsize=24)
ax.legend(loc="lower right", fontsize=20)
ax.set_xlabel('Specificity', fontsize=24)
ax.set_ylabel('Sensitivity', fontsize=24)
plt.show()

fig.savefig('./retinal_allcause_10y/result_generate/roc_allcause_sex_specific.png',dpi=300)