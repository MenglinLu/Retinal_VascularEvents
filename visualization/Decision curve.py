# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:37:05 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

col_map_dict = {
    'mpi':'#c2bdde',
    'frs':'#82afda',
    'ascvd':'#9bbf8a',
    'qrisk':'#e8cf92',
    'SCORE':'#8dcec8',
    'egfr':'#e7dbd3',
    'rmf':'#add3e2',
    'rmfmpi':'#3480b8',
    'model1':'#ffbe7a',
    'model2':'#fa8878',
    'model3':'#c82423'
    }

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_mpi, net_benefit_model, net_benefit_pce, net_benefit_frs,net_benefit_qrisk,net_benefit_all,net_benefit_fp_pce,net_benefit_SCORE,net_benefit_RMF,net_benefit_RMFmpi,net_benefit_fp):
    #Plot
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot(thresh_group, net_benefit_mpi, color = col_map_dict['mpi'], label = 'MPI')
    ax.plot(thresh_group, net_benefit_frs, color = col_map_dict['frs'], label = 'FRS')
    ax.plot(thresh_group, net_benefit_pce, color = col_map_dict['ascvd'], label = 'PCE')
    ax.plot(thresh_group, net_benefit_qrisk, color = col_map_dict['qrisk'], label = 'QRISK3')
    ax.plot(thresh_group, net_benefit_SCORE, color = col_map_dict['SCORE'], label = 'SCORE')
    ax.plot(thresh_group, net_benefit_RMF, color = col_map_dict['rmf'], label = 'RMF')
    ax.plot(thresh_group, net_benefit_RMFmpi, color = col_map_dict['rmfmpi'], label = 'RMF+MPI')
    ax.plot(thresh_group, net_benefit_fp, color = col_map_dict['model1'], label = 'FP')
    ax.plot(thresh_group, net_benefit_model, color = col_map_dict['model2'], label = 'FP+MPI')
    ax.plot(thresh_group, net_benefit_fp_pce, color = col_map_dict['model3'], label = 'FP+CRF')
    
    
    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Arial', 'fontsize': 18}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Arial', 'fontsize': 18}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right',fontsize=18)

    return ax

##MPI
data_res_mpi = pd.read_csv('./allcause_mpi_testdata_pred_risks.csv')

y_label_mpi = data_res_mpi['allcause_label']
y_pred_score_mpi = data_res_mpi['proba_1']

##FP+MPI
data_res = pd.read_csv('./allcause_fpmpi_testdata_pred_risks')
y_label = data_res['allcause_label']
y_pred_score = list(data_res['proba_1'])

##FP+PCE
data_res_fp_pce = pd.read_csv('./allcause_fpcrf_testdata_pred_risks.csv').drop_duplicates()
    
pred_epochi_fp_pce = np.array(data_res_fp_pce[['0','1']])
y_proba_fp_pce =  pred_epochi_fp_pce - np.max(pred_epochi_fp_pce, axis= 1, keepdims=True)
y_proba_fp_pce = np.exp(y_proba_fp_pce) / np.sum(np.exp(y_proba_fp_pce), axis=1, keepdims=True)  

y_label_fp_pce = data_res_fp_pce['allcause_label']
y_pred_score_fp_pce = list(y_proba_fp_pce[:,1])

##FP
data_res_fp = pd.read_csv('./allcause_fp_testdata_pred_risks.csv').drop_duplicates()
    
y_label_fp = data_res_fp['allcause_label']
y_pred_score_fp = list(data_res_fp['proba_1'])


##pce
data_res_pce = pd.read_csv('./allcause_ascvd_testdata_pred_risks.csv').drop_duplicates()

y_label_pce = data_res_pce['allcause_label']
y_pred_score_pce = list(data_res_pce['proba_1'])

##frs
data_res_frs = pd.read_csv('./allcause_framingham_testdata_pred_risks.csv').drop_duplicates()

y_label_frs = data_res_frs['allcause_label']
y_pred_score_frs = list(data_res_frs['proba_1'])

##qrisk
data_res_qrisk = pd.read_csv('./allcause_qrisk_testdata_pred_risks.csv').drop_duplicates()
y_label_qrisk = data_res_qrisk['allcause_label']
y_pred_score_qrisk = list(data_res_qrisk['proba_1'])

##SCORE
data_res_SCORE = pd.read_csv('./allcause_SCORE_testdata_pred_risks.csv').drop_duplicates()
y_label_SCORE = data_res_SCORE['allcause_label']
y_pred_score_SCORE = list(data_res_SCORE['proba_1'])

##RMF
data_res_RMF = pd.read_csv('./allcause_traits_nompi_testdata_pred_risks.csv').drop_duplicates()
y_label_RMF = data_res_RMF['allcause_label']
y_pred_score_RMF = list(data_res_RMF['proba_1'])

##RMFmpi
data_res_RMFmpi = pd.read_csv('./allcause_traits_testdata_pred_risks.csv').drop_duplicates()
y_label_RMFmpi = data_res_RMFmpi['allcause_label']
y_pred_score_RMFmpi = list(data_res_RMFmpi['proba_1'])

thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
net_benefit_mpi = calculate_net_benefit_model(thresh_group, y_pred_score_mpi, y_label_mpi)
net_benefit_pce = calculate_net_benefit_model(thresh_group, y_pred_score_pce, y_label_pce)
net_benefit_frs = calculate_net_benefit_model(thresh_group, y_pred_score_frs, y_label_frs)
net_benefit_qrisk = calculate_net_benefit_model(thresh_group, y_pred_score_qrisk, y_label_qrisk)
net_benefit_fp_pce = calculate_net_benefit_model(thresh_group, y_pred_score_fp_pce, y_label_fp_pce)
net_benefit_fp = calculate_net_benefit_model(thresh_group, y_pred_score_fp, y_label_fp)
net_benefit_SCORE = calculate_net_benefit_model(thresh_group, y_pred_score_SCORE, y_label_SCORE)
net_benefit_RMF = calculate_net_benefit_model(thresh_group, y_pred_score_RMF, y_label_RMF)
net_benefit_RMFmpi = calculate_net_benefit_model(thresh_group, y_pred_score_RMFmpi, y_label_RMFmpi)

fig, ax = plt.subplots(figsize=(9,9))
ax = plot_DCA(ax, thresh_group,net_benefit_mpi, net_benefit_model, net_benefit_pce, net_benefit_frs,net_benefit_qrisk,net_benefit_all,net_benefit_fp_pce,net_benefit_SCORE,net_benefit_RMF,net_benefit_RMFmpi,net_benefit_fp)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(pad=0.5)
plt.xlabel('Threshold probability', fontsize=20)
plt.ylabel('Net benefit', fontsize=20,labelpad=0)
plt.legend(fontsize=17)
plt.grid(False)
TK = plt.gca()
TK.spines['top'].set_linewidth(1)#图框上边
TK.spines['right'].set_linewidth(1)#图框右边
TK.spines['top'].set_color('black')#图框上边
TK.spines['right'].set_color('black')#图框右边
plt.show()
fig.savefig('./decision_curve_allcause.png', dpi = 600)
#plt.show()


