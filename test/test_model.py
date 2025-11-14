import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

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

ckpt_path="/mnt/workspace/MiniX-test/cache/.cache/LimiX-2M.ckpt"

# model_file = torch.load(ckpt_path,map_location="cuda:0")

def plot_roc_curves_tabpfn(data_list):
    """
    使用TabPFN计算并绘制多个数据集的ROC曲线
    
    参数:
        data_list: [[trainx, trainy], [valx, valy], [ootx, ooty]]
    """
    # 数据集名称
    dataset_names = ['Train', 'Validation', 'OOT']
    
    # 颜色设置
    colors = ['blue', 'green', 'red']
    
    # 创建图形
    plt.figure(figsize=(10, 8))

    # 训练模型（使用训练集）
    trainx, trainy = data_list[0]
    testx, testy = data_list[2]

    clf = LimiXPredictor(device=torch.device('cuda'), model_path=ckpt_path, inference_config='config/cls_default_2M_retrieval.json', rbf_chunk_size=10000) # config/cls_default_retrieval.json
    prediction = clf.predict(trainx, trainy, testx, task_type="Classification")
    
    # 存储AUC结果
    auc_results = {}
    
    # 遍历每个数据集
   
        
        # 获取预测概率
    y_proba = prediction
        
        # 判断是二分类还是多分类
    n_classes = y_proba.shape[1]
    
    if n_classes == 2:
        # 二分类：使用正类概率
        y_score = y_proba[:, 1]
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(testy, y_score)
        roc_auc = auc(fpr, tpr)
        
        # 计算AUC
        auc_score = roc_auc_score(testy, y_score)
        auc_results[dataset_name] = auc_score
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{dataset_name} (AUC = {roc_auc:.4f})')
        
        print(f"{dataset_name} AUC: {auc_score:.4f}")
        
    else:
        # 多分类：使用macro平均
        auc_score = roc_auc_score(testy, y_proba, 
                                    multi_class='ovr', 
                                    average='macro')
        auc_results[dataset_name] = auc_score
        
        print(f"{dataset_name} AUC (Macro): {auc_score:.4f}")
        
        # 多分类绘制平均ROC（可选）
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(testy, classes=np.unique(trainy))
        
        # 计算微平均
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{dataset_name} (Micro-avg AUC = {roc_auc:.4f})')
    
    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - TabPFN (Train/Val/OOT)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    # 添加注释
    plt.text(0.6, 0.2, f"Train AUC: {auc_results.get('Train', 0):.4f}\n"
                       f"Val AUC: {auc_results.get('Validation', 0):.4f}\n"
                       f"OOT AUC: {auc_results.get('OOT', 0):.4f}",
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return auc_results



for name,x in zip(tag_list,datalist_pret):
    datafit_list=[]
    if x[0].shape[0]>30000:
        continue
    for y in x:
        xx,yy=y.drop(['target'],axis=1),y['target']
        datafit_list.append([xx,yy])

    auc_res=plot_roc_curves_tabpfn(datafit_list)
    print(name,auc_res)