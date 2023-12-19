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
import random
import torch.backends.cudnn as cudnn
import warnings
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from pytorchtools import EarlyStopping

#python main.py -lr1 1e-4 -lr2 1e-2 -dv 3 -fe demofundus -fmo resnet18 -task stroke

warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() > 0 else "cpu")
#device = 'cpu'
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False
 
setup_seed(42)

def metrics_cls(pred_li, target_li, demo_li):
    pred_epochi, target_epochi, demo_epochi = pred_li.cpu().detach().numpy(), target_li.cpu().detach().numpy(), demo_li.cpu().detach().numpy()
    #sex_li = demo_epochi[:,1]
    y_pred = np.argmax(pred_epochi, axis=1)
    accuracy = accuracy_score(target_epochi, y_pred)
    precision = precision_score(target_epochi, y_pred)
    recall = recall_score(target_epochi, y_pred)
    f1 = f1_score(target_epochi, y_pred)
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)
    #print(y_proba[:,1])
    print('num less 0.3')
    print(str(sum(y_proba[:,1]<=0.3)))
    print('num more 0.7')
    print(str(sum(y_proba[:,1]>=0.7)))
    # print('less 0.3 sex')
    # print(sum(sex_li[y_proba[:,1]<=0.3]))
    # print('more 0.7 sex')
    # print(sum(sex_li[y_proba[:,1]>=0.7]))
    auroc = roc_auc_score(target_epochi, y_proba[:,1])
    auprc = average_precision_score(target_epochi, y_proba[:,1])
    precision_macro = precision_score(target_epochi, y_pred, average='macro')
    recall_macro = recall_score(target_epochi, y_pred, average='macro')
    f1_macro = f1_score(target_epochi, y_pred, average='macro')
    metrics_res = {'acc': accuracy,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1,
                   'auroc': auroc,
                   'auprc': auprc,
                   'precision_macro': precision_macro,
                   'recall_macro': recall_macro,
                   'f1_macro': f1_macro
                   }
    return metrics_res

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class NetFNNCNN_good(nn.Module):
    def __init__(self, out_features=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 32)

        self.fc2 = nn.Linear(32+4, out_features)

        self.fnn_fc1 = nn.Linear(2, 4)
        #self.fnn_bn1 = nn.BatchNorm1d(2)
        #self.fnn_dp1 = nn.Dropout(0.1)

        #self.fnn_fc2 = nn.Linear(8, 8)
        #self.fnn_bn2 = nn.BatchNorm1d(8)
        #self.fnn_dp2 = nn.Dropout(0.1)

    def forward(self, x_left, x_demo):
        x_left = self.conv1(x_left)
        x_left = self.conv2(x_left)
        #print(x_left.shape)
        x_left = self.conv3(x_left)
        x_left = x_left.view(x_left.shape[0], -1)
        x_left = self.relu(self.fc1(x_left))

        output_demo = self.relu(self.fnn_fc1(x_demo))

        output_concat = torch.cat((x_left, output_demo), axis=1)
        x_concat = self.fc2(output_concat)
        return x_concat

def DataPreprocess(data_path, eye_flag, task, random_state):
    stroke_data = pd.read_csv(data_path)
    stroke_data = stroke_data.dropna(subset=['fundus_left_good'], how='any')
    
    numeric_fea = ['age', 'bmi', 'systolic_bp']
    category_fea = ['sex', 'smoking']

    stroke_data[numeric_fea] = stroke_data[numeric_fea].fillna(stroke_data[numeric_fea].mean())

    scaler = StandardScaler()
    stroke_data[numeric_fea] = scaler.fit_transform(stroke_data[numeric_fea]).astype('float32')

    stroke_data[category_fea] = stroke_data[category_fea].fillna(stroke_data[category_fea].median())
    
    stroke_data_2 = pd.concat([stroke_data[numeric_fea], stroke_data[category_fea], stroke_data[['Eid', 'fundus_left_good', 'fundus_right_good', 'allcause_label', 'allcause_time']]], axis=1)

    stroke_data_2_positive = stroke_data_2[stroke_data_2['allcause_label']==1]
    stroke_data_2_negative = stroke_data_2[stroke_data_2['allcause_label']==0]

    stroke_data_2_negative = stroke_data_2_negative.sample(n=len(stroke_data_2_positive)*1,random_state=random_state,axis=0)
    stroke_data_3 = pd.concat([stroke_data_2_positive, stroke_data_2_negative], axis=0)
    stroke_data_3.columns = stroke_data_2.columns
    
    return stroke_data_3

class MyDataset(Dataset):
    def __init__(self, image_dir, data_all, task, eye_flag, feature, transform_left=None):
        self.image_dir = image_dir
        self.data_all = data_all
        self.transform_left = transform_left
        self.time = self.data_all['allcause_time']
        self.label = self.data_all['allcause_label']
        self.eye_flag = eye_flag
        self.feature = feature
        self.data_all_demo = self.data_all[['age','sex']]

    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, idx):
        label_idx = self.label.iloc[idx]
        time_idx = self.time.iloc[idx]

        if(self.feature == 'demofundus'):
            demo_fea = self.data_all_demo.iloc[idx,:]
            eid_idx = self.data_all['Eid'].iloc[idx]
            img_21015_name_idx = str(eid_idx) + '_' + self.data_all['fundus_left_good'].iloc[idx] + '.png'
            image_21015 = Image.open(os.path.join(self.image_dir, '21015', img_21015_name_idx))
            image_21015 = self.transform_left(image_21015)
            return (np.array(demo_fea), image_21015, label_idx, time_idx, eid_idx)
        
        elif(self.feature == 'demooctfundus'):
            pass

        else:
            print('ERROR!')

image_transform_train = transforms.Compose([
    transforms.Resize(256),             
    transforms.CenterCrop(224),    
    #transforms.Resize(128),        
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
    #transforms.Resize(128),  
    #transforms.RandomHorizontalFlip(),  
    #transforms.RandomVerticalFlip(), 
    #transforms.RandomRotation(30),  
    #transforms.ColorJitter(brightness=0.1,contrast=0,saturation=0,hue=0),
    transforms.ToTensor(),              
    transforms.Normalize(mean=[0.0,0.0,0.0], std=[1, 1, 1])  
])

DIR_PATH = '/home/lumenglin/retinal_allcause'
IMAGE_DIR_PATH = '/data'
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
            #print('demo_bh-----------')
            #print(demo_bh)
            #print('image_21015----------')
            #print(image_21015_bh)
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
            #break
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

    ###运行命令
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe demo > both_demo_0.01.txt
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe fundus -fmo cnn > both_fundus_0.01_cnn.txt
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe fundus -fmo resnet18 > both_fundus_0.01_resnet18.txt
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe fundus -fmo resnet50 > both_fundus_0.01_resnet50.txt
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe demofundus -fmo cnn > both_demofundus_0.01_cnn.txt
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe demofundus -fmo resnet18 > both_demofundus_0.01_resnet18.txt
    #python main_both_eye.py -lr 0.01 -dv 2 -eye both -fe demofundus -fmo resnet50 > both_demofundus_0.01_resnet50.txt
 
    LR1 = float(args.rlr1)
    LR2 = float(args.rlr2)
    BATCH_SIZE = int(args.rbatchsize)
    DEVICE_ID = str(args.rdv)
    EYE_FLAG = str(args.reye)
    FEATURE = args.rfeature ##demofundus demo fundus
    FMODEL = str(args.rfmodel)
    TASK = args.rtask
    EPOCH = 300
    PATIENCE = 20
    FLAG = args.rflag
    RS = int(args.rrs)

    #LR1 = 1e-2
    #LR2 = 1e-2
    #BATCH_SIZE = 32
    #DEVICE_ID = 2
    #EYE_FLAG = 'both'
    #PART_DATA = False
    #FEATURE = 'demofundus'
    #FMODEL = 'cnn'
    #TASK = 'mi'
    #EPOCH = 300
    #PATIENCE = 10

    if(EYE_FLAG not in ['both', 'left']):
        print('ERROR EYE FLAG!')
    if(FEATURE not in ['demo', 'fundus', 'demofundus', 'demooctfundus']):
        print('ERROR FEATURE!')

    output_dir = os.path.join(DIR_PATH, 'result_v4', 'lefteye_'+TASK+'_'+str(FEATURE)+'_'+str(FMODEL)+'_LR1'+str(LR1)+'_LR2'+str(LR2)+'_RS_'+str(RS)+'_'+FLAG)
    if(os.path.exists(output_dir) == False):
        os.makedirs(output_dir)

    data_pathh = DIR_PATH+'/data/data_allcause_v4_10y.csv'
    stroke_data_2 = DataPreprocess(data_pathh, EYE_FLAG, TASK, RS)
    print('All data sample count: '+str(len(stroke_data_2)))
    print('Positive data sample count: '+str(stroke_data_2['allcause_label'].sum()))
    
    train_x, test_x, train_y, test_y = train_test_split(stroke_data_2, stroke_data_2['allcause_label'], test_size=0.2, stratify=stroke_data_2['allcause_label'], random_state=RS)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_x['allcause_label'], test_size=0.125, stratify=train_x['allcause_label'], random_state=RS)
    train_x.to_csv(output_dir+'/train_data.csv', index=False)
    test_x.to_csv(output_dir+'/test_data.csv', index=False)
    dataset_train = MyDataset(IMAGE_DIR_PATH, train_x, TASK, EYE_FLAG, FEATURE, image_transform_train)
    dataset_val = MyDataset(IMAGE_DIR_PATH, val_x,  TASK, EYE_FLAG, FEATURE, image_transform_left)
    dataset_test = MyDataset(IMAGE_DIR_PATH, test_x,  TASK, EYE_FLAG, FEATURE, image_transform_left)
    dataset_all = MyDataset(IMAGE_DIR_PATH, stroke_data_2,  TASK, EYE_FLAG, FEATURE, image_transform_left)

    dl_train = DataLoader(dataset_train, BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val = DataLoader(dataset_val, BATCH_SIZE, shuffle=False, drop_last=True)
    dl_test = DataLoader(dataset_test, BATCH_SIZE, shuffle=False, drop_last=True)
    dl_all = DataLoader(dataset_all, BATCH_SIZE, shuffle=False, drop_last=True)

    if(FEATURE == 'demofundus'):
        if(FMODEL == 'cnn'):
         net = NetFNNCNN_good(2).to(device)
    #print(net)
    net.apply(weight_init)
    
    if(FEATURE == 'demofundus'):
        fnn_fc1_params = list(map(id, net.fnn_fc1.parameters()))
        #fnn_bn1_params = list(map(id, net.fnn_bn1.parameters()))
        #fnn_dp1_params = list(map(id, net.fnn_dp1.parameters()))
        #fnn_fc2_params = list(map(id, net.fnn_fc2.parameters()))
        #fnn_bn2_params = list(map(id, net.fnn_bn2.parameters()))
        #fnn_dp2_params = list(map(id, net.fnn_dp2.parameters()))
        fnn_params = fnn_fc1_params
        #fnn_params = fnn_fc1_params + fnn_bn1_params + fnn_dp1_params + fnn_fc2_params + fnn_bn2_params + fnn_dp2_params
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
        res_df.to_csv(output_dir+'/train_res_metrics.csv', index=False)
        
        avg_loss_val, metrics_cls_val, _, _, _, _ = test_epoch(net, dl_val, FEATURE, epoch)

        avg_loss_test, metrics_cls_test, _, _, _, _ = test_epoch(net, dl_test, FEATURE, epoch)
        res_li_test.append([epoch, avg_loss_test, metrics_cls_test['acc'], metrics_cls_test['precision'], metrics_cls_test['recall'], 
                       metrics_cls_test['f1'], metrics_cls_test['auroc'], metrics_cls_test['auprc'], metrics_cls_test['precision_macro'], metrics_cls_test['recall_macro'], 
                       metrics_cls_test['f1_macro']])        
        res_df_test = pd.DataFrame(res_li_test, columns=['epoch','loss','ACC','Precision','Recall','F1-score','AUROC','AUPRC','Precision_macro','Recall_macro','F1-score_macro'])
        res_df_test.to_csv(output_dir+'/test_res_metrics.csv', index=False)
        
        early_stopping(avg_loss_val, net)
        if(early_stopping.early_stop):
            print("Early stopping")
            break

    net.load_state_dict(torch.load(output_dir+'/checkpoint.pt'))	
    avg_loss_train_bestmodel, metrics_cls_train_bestmodel, pred_risks_train_bestmodel, label_train_bestmodel, time_train_bestmodel, eid_train_bestmodel = test_epoch(net, dl_train, FEATURE, epoch)
    avg_loss_val_bestmodel, metrics_cls_val_bestmodel, pred_risks_val_bestmodel, label_val_bestmodel, time_val_bestmodel, eid_val_bestmodel = test_epoch(net, dl_val, FEATURE, epoch)
    avg_loss_test_bestmodel, metrics_cls_test_bestmodel, pred_risks_test_bestmodel, label_test_bestmodel, time_test_bestmodel, eid_test_bestmodel = test_epoch(net, dl_test, FEATURE, epoch)
    avg_loss_all_bestmodel, metrics_cls_all_bestmodel, pred_risks_all_bestmodel, label_all_bestmodel, time_all_bestmodel, eid_all_bestmodel = test_epoch(net, dl_all, FEATURE, epoch)

    train_pred_res_df, train_label, train_time, train_eid = pred_risks_train_bestmodel.cpu().detach().numpy(), label_train_bestmodel.cpu().detach().numpy(), time_train_bestmodel.cpu().detach().numpy(), eid_train_bestmodel.cpu().detach().numpy()
    train_pred_res_df = pd.DataFrame(train_pred_res_df)
    train_pred_res_df['label'] = train_label
    train_pred_res_df['time'] = train_time
    train_pred_res_df['eid'] = train_eid

    val_pred_res_df, val_label, val_time, val_eid = pred_risks_val_bestmodel.cpu().detach().numpy(), label_val_bestmodel.cpu().detach().numpy(), time_val_bestmodel.cpu().detach().numpy(), eid_val_bestmodel.cpu().detach().numpy()
    val_pred_res_df = pd.DataFrame(val_pred_res_df)
    val_pred_res_df['label'] = val_label
    val_pred_res_df['time'] = val_time
    val_pred_res_df['eid'] = val_eid

    test_pred_res_df, test_label, test_time, test_eid = pred_risks_test_bestmodel.cpu().detach().numpy(), label_test_bestmodel.cpu().detach().numpy(), time_test_bestmodel.cpu().detach().numpy(), eid_test_bestmodel.cpu().detach().numpy()
    test_pred_res_df = pd.DataFrame(test_pred_res_df)
    test_pred_res_df['label'] = test_label
    test_pred_res_df['time'] = test_time
    test_pred_res_df['eid'] = test_eid

    all_pred_res_df, all_label, all_time, all_eid = pred_risks_all_bestmodel.cpu().detach().numpy(), label_all_bestmodel.cpu().detach().numpy(), time_all_bestmodel.cpu().detach().numpy(), eid_all_bestmodel.cpu().detach().numpy()
    all_pred_res_df = pd.DataFrame(all_pred_res_df)
    all_pred_res_df['label'] = all_label
    all_pred_res_df['time'] = all_time
    all_pred_res_df['eid'] = all_eid

    train_pred_res_df.to_csv(output_dir+'/train_data_pred_risks.csv', index=False)
    val_pred_res_df.to_csv(output_dir+'/val_data_pred_risks.csv', index=False)
    test_pred_res_df.to_csv(output_dir+'/test_data_pred_risks.csv', index=False)
    all_pred_res_df.to_csv(output_dir+'/all_data_pred_risks.csv', index=False)

    with open(output_dir+'/bestmodel_result_summary.txt', 'w') as f_res:
        f_res.write('The best model on training data: ' + str(metrics_cls_train_bestmodel)+'\n')
        f_res.write('The best model on validation data: ' + str(metrics_cls_val_bestmodel)+'\n')
        f_res.write('The best model on testing data: ' + str(metrics_cls_test_bestmodel)+'\n')
        f_res.write('The best model on all data: ' + str(metrics_cls_all_bestmodel))
    f_res.close()
