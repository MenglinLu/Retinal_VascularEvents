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

file_name_list = [
    #'all_data_pred_risks.csv',
                  'test_data_pred_risks.csv']


for file_name in file_name_list:
    ##FP
    data_res_fp = pd.read_csv('./retinal_allcause_10y/result/lefteye_withoutmeta_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/test_data_pred_risks.csv')
    
    pred_epochi_fp = np.array(data_res_fp[['0','1']])
    y_proba_fp =  pred_epochi_fp - np.max(pred_epochi_fp, axis= 1, keepdims=True)
    y_proba_fp = np.exp(y_proba_fp) / np.sum(np.exp(y_proba_fp), axis=1, keepdims=True)  
    
    y_test_fp = data_res_fp['label']
    probas_fp = list(y_proba_fp[:,1])
    brier_score_fp = brier_score_loss(y_test_fp, probas_fp)
    
    ##FP+CRF
    data_res = pd.read_csv('./retinal_allcause_10y/result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/'+file_name)
    
    pred_epochi = np.array(data_res[['0','1']])
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)  
    
    y_test = data_res['label']
    probas = list(y_proba[:,1])
    brier_score = brier_score_loss(y_test, probas)
    
    
    ##FP+PCE
    data_res_fp_pce = pd.read_csv('./retinal_allcause_10y/result/result_v4_pce/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_/test_calibration_pred_risks.csv').drop_duplicates()
    
    pred_epochi_fp_pce = np.array(data_res_fp_pce[['0','1']])
    y_proba_fp_pce =  pred_epochi_fp_pce - np.max(pred_epochi_fp_pce, axis= 1, keepdims=True)
    y_proba_fp_pce = np.exp(y_proba_fp_pce) / np.sum(np.exp(y_proba_fp_pce), axis=1, keepdims=True)  
    
    y_test_fp_pce = data_res_fp_pce['label']
    probas_fp_pce = list(y_proba_fp_pce[:,1])
    brier_score_fp_pce = brier_score_loss(y_test_fp_pce, probas_fp_pce)
    
    
    ##pce
    data_res_pce = pd.read_csv('./retinal_allcause_10y/result/ascvd_result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/allcause_ascvd_testdata_pred_risks.csv').drop_duplicates()
    
    y_test_pce = data_res_pce['label']
    probas_pce = list(data_res_pce['proba_1'])
    brier_score_pce = brier_score_loss(y_test_pce, probas_pce)
    
    ##frs
    data_res_frs = pd.read_csv('./retinal_allcause_10y/result/ascvd_result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/allcause_framingham_testdata_pred_risks.csv').drop_duplicates()
    
    y_test_frs = data_res_frs['label']
    probas_frs = list(data_res_frs['proba_1'])
    brier_score_frs = brier_score_loss(y_test_frs, probas_frs)
     
    fig = plt.figure()
    skplt.metrics.plot_calibration_curve(y_test, [probas_fp,probas,probas_fp_pce,probas_pce,probas_frs], clf_names=['FP (BS = '+str(round(brier_score_fp,3))+' ± 0.007)','FP+MPI (BS = '+str(round(brier_score,3))+' ± 0.005)','FP+CRF (BS = '+str(round(brier_score_fp_pce,3))+' ± 0.005)','PCE (BS = '+str(round(brier_score_pce,3))+' ± 0.006)','FRS (BS = '+str(round(brier_score_frs,3))+' ± 0.010)'], n_bins=10,title=None,figsize=(9,6),text_fontsize=20,cmap='Set2')
    plt.legend(loc="lower right", fontsize=17)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.show()
    plt.savefig('./retinal_allcause_10y/result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/calibration_plot_'+file_name+'_pce.png', dpi=600)
    