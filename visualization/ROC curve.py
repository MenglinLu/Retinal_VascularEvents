# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:23:17 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

dir_path = './result_FPCRF'

###绘制FP+CRF
file_list = os.listdir(dir_path)

tprs_fp = []
aucs_fp = []
mean_fpr_fp = np.linspace(0, 1, 100)
for file_i_fp in file_list:
    file_i_path_fp = os.path.join(dir_path, file_i_fp)
    data_file_i_fp = pd.read_csv(file_i_path_fp)
    data_file_i_fp.columns = ['0', '1', 'label', 'time', 'Eid']
    
    pred_epochi = np.array(data_file_i_fp[['0','1']])
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)
    data_file_i_fp['proba_0'] = y_proba[:,0]
    data_file_i_fp['proba_1'] = y_proba[:,1]
    
    fpr1_fp, tpr1_fp, threshold1_fp = roc_curve(data_file_i_fp['label'], data_file_i_fp['proba_1'])
    
    auc1_fp = auc(fpr1_fp, tpr1_fp)
    
    interp_tpr_fp = interp(mean_fpr_fp, fpr1_fp, tpr1_fp)
    
    interp_tpr_fp[0] = 0.0
    tprs_fp.append(interp_tpr_fp)
    aucs_fp.append(auc1_fp)

mean_tpr_fp = np.mean(tprs_fp, axis=0)
mean_tpr_fp[-1] = 1.0
mean_auc_fp = auc(mean_fpr_fp, mean_tpr_fp)
std_auc_fp = np.std(aucs_fp)
fig, ax = plt.subplots(figsize=(9,9))
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.plot(mean_fpr_fp, mean_tpr_fp, color='green',
        label=r'FP+CRF (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_fp, std_auc_fp),
        lw=2, alpha=.3)
std_tpr_fp = np.std(tprs_fp, axis=0)
tprs_upper_fp= np.minimum(mean_tpr_fp + std_tpr_fp, 1)
tprs_lower_fp = np.maximum(mean_tpr_fp - std_tpr_fp, 0)
ax.fill_between(mean_fpr_fp, tprs_lower_fp, tprs_upper_fp, color='grey', alpha=.2)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.set_title("ROC curve of all-cause event", fontsize=15)
ax.legend(loc="lower right", fontsize=14)
ax.set_xlabel('Specificity', fontsize=15)
ax.set_ylabel('Sensitivity', fontsize=15)
plt.show()

fig.savefig('./roc_allcause_fpcrf.png',dpi=300)
    
    
