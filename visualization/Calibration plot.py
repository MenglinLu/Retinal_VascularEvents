# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:30:00 2023

@author: Lenovo
"""
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss

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

##FP+MPI
data_res = pd.read_csv('./allcause_fpmpi_testdata_pred_risks.csv').drop_duplicates()

y_test = data_res['allcause_label']
probas = list(data_res['proba_1'])
brier_score = brier_score_loss(y_test, probas)

##FP+CRF
data_res_fp_pce = pd.read_csv('./allcause_fpcrf_testdata_pred_risks.csv').drop_duplicates()
    
pred_epochi_fp_pce = np.array(data_res_fp_pce[['0','1']])
y_proba_fp_pce =  pred_epochi_fp_pce - np.max(pred_epochi_fp_pce, axis= 1, keepdims=True)
y_proba_fp_pce = np.exp(y_proba_fp_pce) / np.sum(np.exp(y_proba_fp_pce), axis=1, keepdims=True)  

y_test_fp_pce = data_res_fp_pce['allcause_label']
probas_fp_pce = list(y_proba_fp_pce[:,1])
brier_score_fp_pce = brier_score_loss(y_test_fp_pce, probas_fp_pce)

##FP
data_res_fp = pd.read_csv('./allcause_fp_testdata_pred_risks.csv').drop_duplicates()
    
y_test_fp = data_res_fp['allcause_label']
probas_fp = list(data_res_fp['proba_1'])
brier_score_fp = brier_score_loss(y_test_fp, probas_fp)

##pce
data_res_pce = pd.read_csv('./allcause_ascvd_testdata_pred_risks.csv').drop_duplicates()

y_test_pce = data_res_pce['allcause_label']
probas_pce = list(data_res_pce['proba_1'])
brier_score_pce = brier_score_loss(y_test_pce, probas_pce)

##frs
data_res_frs = pd.read_csv('./allcause_framingham_testdata_pred_risks.csv').drop_duplicates()

y_test_frs = data_res_frs['allcause_label']
probas_frs = list(data_res_frs['proba_1'])
brier_score_frs = brier_score_loss(y_test_frs, probas_frs)

##mpi
data_res_mpi = pd.read_csv('./allcause_mpi_testdata_pred_risks.csv').drop_duplicates()

y_test_mpi = data_res_mpi['allcause_label']
probas_mpi = list(data_res_mpi['proba_1'])
brier_score_mpi = brier_score_loss(y_test_mpi, probas_mpi)

##qrisk
data_res_qrisk = pd.read_csv('./allcause_qrisk_testdata_pred_risks.csv').drop_duplicates()
y_test_qrisk = data_res_qrisk['allcause_label']
probas_qrisk = list(data_res_qrisk['proba_1'])
brier_score_qrisk = brier_score_loss(y_test_qrisk, probas_qrisk)

##SCORE
data_res_SCORE = pd.read_csv('./allcause_SCORE_testdata_pred_risks.csv').drop_duplicates()
y_test_SCORE = data_res_SCORE['allcause_label']
probas_SCORE = list(data_res_SCORE['proba_1'])
brier_score_SCORE = brier_score_loss(y_test_SCORE, probas_SCORE)

##RMF
data_res_RMF = pd.read_csv('./allcause_traits_nompi_testdata_pred_risks.csv').drop_duplicates()
y_test_RMF = data_res_RMF['allcause_label']
probas_RMF = list(data_res_RMF['proba_1'])
brier_score_RMF = brier_score_loss(y_test_RMF, probas_RMF)

##RMF_mpi
data_res_RMFmpi = pd.read_csv('./allcause_traits_testdata_pred_risks.csv').drop_duplicates()
y_test_RMFmpi = data_res_RMFmpi['allcause_label']
probas_RMFmpi = list(data_res_RMFmpi['proba_1'])
brier_score_RMFmpi = brier_score_loss(y_test_RMFmpi, probas_RMFmpi)


from matplotlib.colors import LinearSegmentedColormap
candidate_colors = ['#c2bdde',
'#82afda',
'#9bbf8a',
'#e8cf92',
'#8dcec8',
'#add3e2',
'#3480b8',
'#ffbe7a',
'#fa8878',
'#c82423']

# 创建自定义colormap
def create_custom_cmap(colors):
    cmap_name = 'custom_cmap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

custom_cmap = create_custom_cmap(candidate_colors)

fig = plt.figure()
skplt.metrics.plot_calibration_curve(y_test, 
                                     [probas_mpi,probas_frs,probas_pce,probas_qrisk,
                                      probas_SCORE, probas_RMF, probas_RMFmpi,                                   
                                      probas_fp,probas,probas_fp_pce], 
                                     clf_names=['MPI(BS='+str("{:.3f}".format(brier_score_mpi)),
                                                'FRS(BS='+str("{:.3f}".format(brier_score_frs)),
                                                'PCE(BS='+str("{:.3f}".format(brier_score_pce)),
                                                'QRISK3(BS='+str("{:.3f}".format(brier_score_qrisk)),
                                                'SCORE(BS='+str("{:.3f}".format(brier_score_SCORE)),
                                                'RMF(BS='+str("{:.3f}".format(brier_score_RMF)),
                                                'RMF+MPI(BS='+str("{:.3f}".format(brier_score_RMFmpi)),
                                                'FP(BS='+str("{:.3f}".format(brier_score_fp)),
                                                'FP+MPI(BS='+str("{:.3f}".format(brier_score)),
                                                'FP+CRF(BS='+str("{:.3f}".format(brier_score_fp_pce))], n_bins=10,title=None,figsize=(13,7),text_fontsize=18.5,cmap=custom_cmap)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=18.5,ncol=2)
plt.xticks(fontsize=19.5)
plt.yticks(fontsize=19.5)
plt.tight_layout()
# plt.show()
plt.savefig('./calibration_plot_allcause.png', dpi=600)