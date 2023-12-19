# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:37:05 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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

def plot_DCA(ax, thresh_group, net_benefit_fp, net_benefit_model, net_benefit_pce, net_benefit_frs,net_benefit_all,net_benefit_fp_pce):
    #Plot
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot(thresh_group, net_benefit_fp, color = 'green', label = 'FP')
    ax.plot(thresh_group, net_benefit_model, color = 'b', label = 'FP+MPI')
    ax.plot(thresh_group, net_benefit_fp_pce, color = '#2983b1', label = 'FP+CRF')
    ax.plot(thresh_group, net_benefit_pce, color = 'purple', label = 'PCE')
    ax.plot(thresh_group, net_benefit_frs, color = '#db6968', label = 'FRS')
    
    
    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Arial', 'fontsize': 20}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Arial', 'fontsize': 20}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right',fontsize=18)

    return ax

file_name_list = [
    #'all_data_pred_risks.csv',
                  'test_data_pred_risks.csv']

for file_name in file_name_list:
    ##FP
    data_res_fp = pd.read_csv('./retinal_allcause_10y/result/lefteye_withoutmeta_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/test_data_pred_risks.csv')
    
    pred_epochi_fp = np.array(data_res_fp[['0','1']])
    y_proba_fp =  pred_epochi_fp - np.max(pred_epochi_fp, axis= 1, keepdims=True)
    y_proba_fp = np.exp(y_proba_fp) / np.sum(np.exp(y_proba_fp), axis=1, keepdims=True)  
    
    y_label_fp = data_res_fp['label']
    y_pred_score_fp = y_proba_fp[:,1]
    
    ##FP+CRF
    data_res = pd.read_csv('./retinal_allcause_10y/result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/'+file_name)
    pred_epochi = np.array(data_res[['0','1']])
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)  
    thresh_group = np.arange(0,1,0.01)
    y_pred_score = y_proba[:,1]
    y_label = data_res['label']
    
    ##FP+PCE
    data_res_fp_pce = pd.read_csv('./retinal_allcause_10y/result/result_v4_pce/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_/test_calibration_pred_risks.csv')
    
    pred_epochi_fp_pce = np.array(data_res_fp_pce[['0','1']])
    y_proba_fp_pce =  pred_epochi_fp_pce - np.max(pred_epochi_fp_pce, axis= 1, keepdims=True)
    y_proba_fp_pce = np.exp(y_proba_fp_pce) / np.sum(np.exp(y_proba_fp_pce), axis=1, keepdims=True)  
    
    y_label_fp_pce = data_res_fp_pce['label']
    y_pred_score_fp_pce = y_proba_fp_pce[:,1]
    
    ##pce
    data_res_pce = pd.read_csv('./retinal_allcause_10y/result/ascvd_result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/allcause_ascvd_testdata_pred_risks.csv').drop_duplicates()
    
    y_label_pce = data_res_pce['label']
    y_pred_score_pce = list(data_res_pce['proba_1'])
    
    ##frs
    data_res_frs = pd.read_csv('./retinal_allcause_10y/result/ascvd_result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/allcause_framingham_testdata_pred_risks.csv').drop_duplicates()
    
    y_label_frs = data_res_frs['label']
    y_pred_score_frs = list(data_res_frs['proba_1'])
    
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    net_benefit_fp = calculate_net_benefit_model(thresh_group, y_pred_score_fp, y_label_fp)
    net_benefit_pce = calculate_net_benefit_model(thresh_group, y_pred_score_pce, y_label_pce)
    net_benefit_frs = calculate_net_benefit_model(thresh_group, y_pred_score_frs, y_label_frs)
    net_benefit_fp_pce = calculate_net_benefit_model(thresh_group, y_pred_score_fp_pce, y_label_fp_pce)

    fig, ax = plt.subplots(figsize=(10.5,7))
    ax = plot_DCA(ax, thresh_group,net_benefit_fp, net_benefit_model, net_benefit_pce, net_benefit_frs, net_benefit_all, net_benefit_fp_pce)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.tick_params(pad=0.5)
    plt.xlabel('Threshold probability', fontsize=27)
    plt.ylabel('Net benefit', fontsize=26,labelpad=0)
    plt.legend(fontsize=24)
    plt.grid(False)
    TK = plt.gca()
    TK.spines['top'].set_linewidth(1)#图框上边
    TK.spines['right'].set_linewidth(1)#图框右边
    TK.spines['top'].set_color('black')#图框上边
    TK.spines['right'].set_color('black')#图框右边
    plt.show()
    fig.savefig('./retinal_allcause_10y/result/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905/decision_curve_'+file_name+'_pce.png', dpi = 300)
    #plt.show()


