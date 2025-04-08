import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from sklearn.model_selection import train_test_split

import torch # For building the networks 

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.backends.cudnn as cudnn
import warnings
from tqdm import tqdm
from ..utils.pytorchtools import EarlyStopping
from ..utils.utils import setup_seed, weight_init
from ..utils.metrics import metrics_cls
from ..data.data_preprocess import DataPreprocess
from ..data.MyDataset import MyDataset
from ..model.model import NetFNNCNN_good
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

#python main.py -lr1 1e-4 -lr2 1e-2 -dv 3 -fe demofundus -fmo resnet18 -task stroke

warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() > 0 else "cpu")

setup_seed(42)

image_transform_train = transforms.Compose([
    transforms.Resize(256),              
    transforms.CenterCrop(224),     
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(30),  
    transforms.ColorJitter(brightness=0.1,contrast=0,saturation=0,hue=0),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.0,0.0,0.0], std=[1, 1, 1])
])
image_transform_left = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),        
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.0,0.0,0.0], std=[1, 1, 1])   
])

DIR_PATH = '..'
IMAGE_DIR_PATH = './data'
loss_func = nn.CrossEntropyLoss().to(device)

def train_epoch(net, train_dataloader, optimizer, feature, epoch):
    net.train()
    total_loss = []
    pred_epochi = torch.tensor([]).to(device)
    target_epochi = torch.tensor([], dtype=torch.int64).to(device)
    demo_epochi = torch.tensor([]).to(device)

    for batch_index, data in enumerate(tqdm(train_dataloader)):
        #break
        if(feature == 'demofundus'):
            demo_bh, image_21015_bh, label_bh, time_bh, eid_bh = data
            demo_bh, image_21015_bh, label_bh, time_bh, eid_bh = demo_bh, image_21015_bh, label_bh, time_bh, eid_bh
            demo_bh, image_21015_bh, label_bh, time_bh = torch.FloatTensor(demo_bh.numpy()).to(device), torch.FloatTensor(image_21015_bh).to(device), torch.LongTensor(label_bh.numpy()).to(device), torch.FloatTensor(time_bh.numpy()).to(device)
            outputs = net(image_21015_bh, demo_bh)

        optimizer.zero_grad()

        loss = loss_func(outputs, label_bh)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        pred_epochi = torch.cat((pred_epochi, outputs), 0)
        target_epochi = torch.cat((target_epochi, label_bh), 0)
        demo_epochi = torch.cat((demo_epochi, demo_bh), 0)

    avg_loss = np.mean(total_loss)
    cls_metric = metrics_cls(pred_epochi, target_epochi, demo_epochi)
    print('Training Epoch: {epoch}, Average Loss: {avg_loss}, acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, auroc: {auroc}, auprc: {auprc}'.format(
        epoch=epoch,
        avg_loss=avg_loss,
        acc=cls_metric['acc'],
        precision=cls_metric['precision'],
        recall=cls_metric['recall'],
        f1=cls_metric['f1'],
        auroc=cls_metric['auroc'],
        auprc=cls_metric['auprc']
    ))
    return avg_loss, cls_metric

def test_epoch(net, test_dataloader, feature, epoch):
    net.eval()
    total_loss = []
    pred_epochi = torch.tensor([]).to(device)
    target_epochi = torch.tensor([], dtype=torch.int64).to(device)
    time_epochi = torch.tensor([]).to(device)
    eid_epochi = torch.tensor([]).to(device)
    demo_epochi = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(test_dataloader)):
            if(feature == 'demofundus'):
                demo_bh, image_21015_bh, label_bh, time_bh, eid_bh = data
                demo_bh, image_21015_bh, label_bh, time_bh, eid_bh = torch.FloatTensor(demo_bh.numpy()).to(device), torch.FloatTensor(image_21015_bh).to(device), torch.LongTensor(label_bh.numpy()).to(device), torch.FloatTensor(time_bh.numpy()).to(device), torch.LongTensor(eid_bh.numpy()).to(device)
                outputs = net(image_21015_bh, demo_bh)

            loss = loss_func(outputs, label_bh)
            total_loss.append(loss.item())
            pred_epochi = torch.cat((pred_epochi, outputs), 0)
            time_epochi = torch.cat((time_epochi, time_bh), 0)
            eid_epochi = torch.cat((eid_epochi, eid_bh), 0)
            target_epochi = torch.cat((target_epochi, label_bh), 0)
            demo_epochi = torch.cat((demo_epochi, demo_bh), 0)

    avg_loss = np.mean(total_loss)
    cls_metric = metrics_cls(pred_epochi, target_epochi, demo_epochi)
    print('Testing Epoch: {epoch}, Average Loss: {avg_loss}, acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, auroc: {auroc}, auprc: {auprc}'.format(
        epoch=epoch,
        avg_loss=avg_loss,
        acc=cls_metric['acc'],
        precision=cls_metric['precision'],
        recall=cls_metric['recall'],
        f1=cls_metric['f1'],
        auroc=cls_metric['auroc'],
        auprc=cls_metric['auprc']
    ))
    return avg_loss, cls_metric, pred_epochi, target_epochi, time_epochi, eid_epochi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batchsize', dest='rbatchsize', required=False, help='BATCH SIZE', default=32)
    parser.add_argument('-lr1', '--learningrate1', dest='rlr1', required=False, help='learning rate fundus', default=1e-3)
    parser.add_argument('-lr2', '--learningrate2', dest='rlr2', required=False, help='learning rate demo', default=1e-3)
    parser.add_argument('-dv', '--device', dest='rdv', required=False, help='device id', default='2')
    parser.add_argument('-eye', '--eyeflag', dest='reye', required=False, help='eye flag', default='left')
    parser.add_argument('-fe', '--feature', dest='rfeature', required=False, help='feature', default='demofundus')
    parser.add_argument('-fmo', '--fmodel', dest='rfmodel', required=False, help='fundus model', default='cnn')
    parser.add_argument('-task', '--task', dest='rtask', required=True, help='task')
    parser.add_argument('-flag', '--flag', dest='rflag', required=False, help='flag', default='')
    parser.add_argument('-rrs', '--rrs', dest='rrs', required=True, help='randomstate')

    args = parser.parse_args()
 
    LR1 = float(args.rlr1)
    LR2 = float(args.rlr2)
    BATCH_SIZE = int(args.rbatchsize)
    DEVICE_ID = str(args.rdv)
    EYE_FLAG = str(args.reye)
    FEATURE = args.rfeature 
    FMODEL = str(args.rfmodel)
    TASK = args.rtask
    EPOCH = 300
    PATIENCE = 20
    FLAG = args.rflag
    RS = int(args.rrs)

    output_dir = os.path.join(DIR_PATH, 'result_v4_pce', 'lefteye_'+TASK+'_'+str(FEATURE)+'_'+str(FMODEL)+'_LR1'+str(LR1)+'_LR2'+str(LR2)+'_RS_'+str(RS)+'_'+FLAG)
    if(os.path.exists(output_dir) == False):
        os.makedirs(output_dir)

    data_pathh = DIR_PATH+'/data/data_allcause_10y.csv'
    stroke_data_2 = DataPreprocess(data_pathh, EYE_FLAG, TASK, RS)
    print('All data sample count: '+str(len(stroke_data_2)))
    print('Positive data sample count: '+str(stroke_data_2['allcause_label'].sum()))
    
    train_x, test_x, train_y, test_y = train_test_split(stroke_data_2, stroke_data_2['allcause_label'], test_size=0.2, stratify=stroke_data_2['allcause_label'], random_state=RS)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_x['allcause_label'], test_size=0.125, stratify=train_x['allcause_label'], random_state=RS)
    
    ##class balance
    majority_class_samples = (train_y == 0).sum()
    
    under_sampler = RandomUnderSampler(sampling_strategy={0: majority_class_samples // 3, 1: (train_y == 1).sum()}, random_state=RS)
    train_x_under, train_y_under = under_sampler.fit_resample(train_x, train_y)
    
    smote = SMOTE(sampling_strategy='auto', random_state=RS)
    
    train_x_resampled, train_y_resampled = smote.fit_resample(train_x_under, train_y_under)
        
    dataset_train = MyDataset(IMAGE_DIR_PATH, train_x_resampled, train_y_resampled, TASK, EYE_FLAG, FEATURE, image_transform_train)
    dataset_val = MyDataset(IMAGE_DIR_PATH, val_x, val_y, TASK, EYE_FLAG, FEATURE, image_transform_left)
    dataset_test = MyDataset(IMAGE_DIR_PATH, test_x, test_y,  TASK, EYE_FLAG, FEATURE, image_transform_left)

    dl_train = DataLoader(dataset_train, BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val = DataLoader(dataset_val, BATCH_SIZE, shuffle=False, drop_last=True)
    dl_test = DataLoader(dataset_test, BATCH_SIZE, shuffle=False, drop_last=True)

    if(FEATURE == 'demofundus'):
        if(FMODEL == 'cnn'):
         net = NetFNNCNN_good(2).to(device)
    net.apply(weight_init)
    
    if(FEATURE == 'demofundus'):
        fnn_params = list(map(id, net.fnn_fc1.parameters()))
        fnn_params1 = filter(lambda p: id(p) in fnn_params, net.parameters())
        cnn_params = filter(lambda p: id(p) not in fnn_params, net.parameters())
        optimizer = torch.optim.Adam([
        {'params': fnn_params1},
        {'params': cnn_params, 'lr': LR1}], LR2, (0.9, 0.999), eps=1e-08, weight_decay=1e-6)
    else:
        optimizer = torch.optim.Adam(net.parameters(), LR1, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    
    early_stopping = EarlyStopping(PATIENCE, verbose=True, path=output_dir+'/checkpoint.pt')

    res_li = []
    res_li_test = []
    for epoch in tqdm(range(EPOCH)):
        print('epoch----------'+str(epoch))
        avg_loss, metrics_cls_train = train_epoch(net, dl_train, optimizer, FEATURE, epoch)
        res_li.append([epoch, avg_loss, metrics_cls_train['acc'], metrics_cls_train['precision'], metrics_cls_train['recall'], 
                       metrics_cls_train['f1'], metrics_cls_train['auroc'], metrics_cls_train['auprc'], metrics_cls_train['precision_macro'], metrics_cls_train['recall_macro'], 
                       metrics_cls_train['f1_macro']])
        
        res_df = pd.DataFrame(res_li, columns=['epoch','loss','ACC','Precision','Recall','F1-score','AUROC','AUPRC','Precision_macro','Recall_macro','F1-score_macro'])
        
        avg_loss_val, _, _, _, _, _ = test_epoch(net, dl_val, FEATURE, epoch)

        avg_loss_test, metrics_cls_test, _, _, _, _ = test_epoch(net, dl_test, FEATURE, epoch)
        res_li_test.append([epoch, avg_loss_test, metrics_cls_test['acc'], metrics_cls_test['precision'], metrics_cls_test['recall'], 
                       metrics_cls_test['f1'], metrics_cls_test['auroc'], metrics_cls_test['auprc'], metrics_cls_test['precision_macro'], metrics_cls_test['recall_macro'], 
                       metrics_cls_test['f1_macro']])        
        res_df_test = pd.DataFrame(res_li_test, columns=['epoch','loss','ACC','Precision','Recall','F1-score','AUROC','AUPRC','Precision_macro','Recall_macro','F1-score_macro'])
        
        early_stopping(avg_loss_val, net)
        if(early_stopping.early_stop):
            print("Early stopping")
            break




