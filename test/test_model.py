import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

data=pd.read_csv('test/data_nyxj_sample.csv')
data_numeric=data.select_dtypes(np.number)
data_numeric.drop(['idx','code'],axis=1,inplace=True)
tag_list=data['tag'].unique()
data_taglist=[]

for x in tag_list:
    data_taglist.append(data[data['tag']==x])

for name,x in zip(tag_list,data_taglist):
    print(name,x.shape)

datalist_pret=[]

for x in data_taglist:
    train,val,oot=x[x['dataSet']=='trainSet'], x[x['dataSet']=='valSet'],x[x['dataSet']=='ootSet']
    b=[train,val,oot]
    c=[]
    for y in b:
        yx=y.select_dtypes(np.number).drop(['idx','code'],axis=1)
        c.append(yx)
    del b
    datalist_pret.append(c)



from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
import numpy as np
import os, sys
import torch

# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "29500"


from inference.predictor import LimiXPredictor

ckpt_path="/mnt/workspace/MiniX-test/cache/.cache/LimiX-16M.ckpt"

# model_file = torch.load(ckpt_path,map_location="cuda:0")

def plot_roc_curves_minix(data_list,sample_ratio=1):
    
    plt.figure(figsize=(10, 8))

    trainx, trainy = data_list[0]
    testx, testy = data_list[2]

    if sample_ratio<1:
        df_xy=pd.concat([trainx,trainy],axis=1)
        df_xy=df_xy.sample(frac=sample_ratio,random_state=42)
        trainx,trainy=df_xy.drop(['target'],axis=1),df_xy['target']
        # df_xyt=pd.concat([testx,testy],axis=1)
        # df_xyt=df_xyt.sample(frac=sample_ratio,random_state=42)
        # testx,testy=df_xyt.drop(['target'],axis=1),df_xyt['target']



    clf = LimiXPredictor(device=torch.device('cuda'), model_path=ckpt_path, inference_config='config/cls_default_noretrieval.json', rbf_chunk_size=10000) # config/cls_default_retrieval.json
    
    prediction = clf.predict(trainx, trainy, testx, task_type="Classification")
    
    auc_results = {}
    
    y_proba = prediction
        
    n_classes = y_proba.shape[1]
    
    if n_classes == 2:

        y_score = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(testy, y_score)
        roc_auc = auc(fpr, tpr)
        
        auc_score = roc_auc_score(testy, y_score)
        auc_results['AUC'] = auc_score
        
        plt.plot(fpr, tpr, color='red', lw=2,
                label=f'(AUC = {roc_auc:.4f})')
        
        print(f" AUC: {auc_score:.4f}")
        
    

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - TabPFN (Train/Val/OOT)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    
    return auc_results



for name,x in zip(tag_list,datalist_pret):
    datafit_list=[]
    if x[0].shape[0]>30000:
        continue
    for y in x:
        xx,yy=y.drop(['target'],axis=1),y['target']
        datafit_list.append([xx,yy])

    auc_res=plot_roc_curves_minix(datafit_list,0.5)
    print(name,auc_res)