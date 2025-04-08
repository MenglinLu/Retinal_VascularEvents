from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_dir, data_all_x,  data_all_y,  task, eye_flag, feature, transform_left=None):
        self.image_dir = image_dir
        self.data_all = data_all
        self.transform_left = transform_left
        self.time = self.data_all['allcause_time']
        self.label = data_all_y,  
        self.eye_flag = eye_flag
        self.feature = feature
        self.data_all_demo = self.data_all[['age', 'systolic_bp', 'total_chol', 'high_chol','sex', 'smoking', 'ethnic','hypertensive_treat', 'diabete_ever_icd10']]

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
        
        else:
            print('ERROR!')