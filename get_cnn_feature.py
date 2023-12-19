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
device = torch.device("cuda:3" if torch.cuda.is_available() > 0 else "cpu")
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
    
    return stroke_data_2

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


image_transform_left = transforms.Compose([
    transforms.Resize(256),               # 把图片resize为256*256
    transforms.CenterCrop(224),           # 中心裁剪448*448
    #transforms.Resize(128),  
    #transforms.RandomHorizontalFlip(),    # 水平翻转
    #transforms.RandomVerticalFlip(),    # 垂直翻转
    #transforms.RandomRotation(30),  
    #transforms.ColorJitter(brightness=0.1,contrast=0,saturation=0,hue=0),
    transforms.ToTensor(),                # 将图像转为Tensor
    transforms.Normalize(mean=[0.0,0.0,0.0], std=[1, 1, 1])   # 标准化
])

DIR_PATH = '/home/lumenglin/retinal_allcause'
IMAGE_DIR_PATH = '/data'
loss_func = nn.CrossEntropyLoss().to(device)

if __name__ == '__main__':

    LR1 = 1e-2
    LR2 = 1e-2
    BATCH_SIZE = 64
    DEVICE_ID = 2
    EYE_FLAG = 'left'
    PART_DATA = False
    FEATURE = 'demofundus'
    FMODEL = 'cnn'
    TASK = 'allcause_label'
    EPOCH = 300
    PATIENCE = 10
    RS = 0

    if(EYE_FLAG not in ['both', 'left']):
        print('ERROR EYE FLAG!')
    if(FEATURE not in ['demo', 'fundus', 'demofundus', 'demooctfundus']):
        print('ERROR FEATURE!')

    data_pathh = DIR_PATH+'/data/data_allcause_v4_10y.csv'
    stroke_data_2 = DataPreprocess(data_pathh, EYE_FLAG, TASK, RS)
    print('All data sample count: '+str(len(stroke_data_2)))
    print('Positive data sample count: '+str(stroke_data_2['allcause_label'].sum()))
    
    dataset_all = MyDataset(IMAGE_DIR_PATH, stroke_data_2.iloc[-100:,:],  TASK, EYE_FLAG, FEATURE, image_transform_left)

    dl_all = DataLoader(dataset_all, 10, shuffle=False, drop_last=True)

    if(FEATURE == 'demofundus'):
        if(FMODEL == 'cnn'):
         net = NetFNNCNN_good(2).to(device)

    output_dir = '/home/lumenglin/retinal_allcause/result_v4/lefteye_allcause_label_demofundus_cnn_LR10.0005_LR20.001_RS_80_0905'

    ###取最优模型，记录模型效果和在四个数据集上的预测结果，用于绘制生存曲线
    net.load_state_dict(torch.load(output_dir+'/checkpoint.pt'))	
    
    net.eval()
    total_loss = []
    time_epochi = torch.tensor([]).to(device)
    eid_epochi = torch.tensor([]).to(device)

    features_out_hook = []
    # 使用 hook 函数
    def hook(module, fea_in, fea_out):
        #features_in_hook.append(fea_in.data)         # 勾的是指定层的输入
        features_out_hook.append(fea_out.data.cpu().detach().numpy())      # 勾的是指定层的输出
        return None

    layer_name = 'fc1'
    for (name, module) in net.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(dl_all)):
            #break
            demo_bh, image_21015_bh, label_bh, time_bh, eid_bh = data
            demo_bh, image_21015_bh, label_bh, time_bh, eid_bh = torch.FloatTensor(demo_bh.numpy()).to(device), torch.FloatTensor(image_21015_bh).to(device), torch.LongTensor(label_bh.numpy()).to(device), torch.FloatTensor(time_bh.numpy()).to(device), torch.LongTensor(eid_bh.numpy()).to(device)

            output = net(image_21015_bh, demo_bh)
            time_epochi = torch.cat((time_epochi, time_bh), 0)
            eid_epochi = torch.cat((eid_epochi, eid_bh), 0)
    
    cnn_fea_df = pd.DataFrame(np.array(features_out_hook).reshape(-1,32))
    cnn_fea_df['Eid'] = eid_epochi.cpu().detach().numpy()
    cnn_fea_df['time'] = time_epochi.cpu().detach().numpy()
    cnn_fea_df.to_csv(output_dir+'/cnn_feature_alldataset_2.csv', index=False)

            
