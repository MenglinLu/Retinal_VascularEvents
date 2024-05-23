import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def DataPreprocess(data_path, eye_flag, task, random_state):
    stroke_data = pd.read_csv(data_path)
    stroke_data = stroke_data.dropna(subset=['fundus_left_good'], how='any')
    print('original data rows: '+str(len(stroke_data)))
    
    numeric_fea = ['age', 'systolic_bp', 'total_chol', 'high_chol']
    category_fea = ['sex', 'smoking', 'ethnic','hypertensive_treat', 'diabete_ever_icd10']

    data1 = stroke_data[numeric_fea+category_fea].isnull().sum(axis=1)
    stroke_data = stroke_data.loc[data1<4, :]
    print('data rows: '+str(len(stroke_data)))

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