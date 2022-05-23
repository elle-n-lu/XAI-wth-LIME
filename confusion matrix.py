#dataset = pd.read_csv("Dataset_clean_dropna.csv")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import glob

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

# %matplotlib inline

dataset = pd.read_csv("../datasets/DATA_COMBINE/combine2/combine2_withmin.csv")

# Splitting dataset into features and labels.
labels = dataset['Label']
features = dataset.loc[:, dataset.columns != 'Label'].astype('float64')

# print(len(labels))
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(features)
features = scaler.transform(features)

#--------------------------------------------------

#混淆矩阵用到的数据
# features = features[501002:501013]
# labels =labels[501002:501013]

#--------------------------------------------------
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE.fit(labels)
labels = LE.transform(labels)

d = LE.inverse_transform(labels)
d = pd.Series(d)
d.unique()

class2idx=dict(enumerate(d.unique().flatten(), 0))
idx2class = {v: k for k, v in class2idx.items()}
d=dataset['Label'].value_counts()
print(idx2class,d)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = features, features, labels, labels
# X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

X_train = torch.from_numpy(np.array(X_train)).float() # 输入 x 张量
X_test = torch.from_numpy(np.array(X_test)).float()
Y_train = torch.from_numpy(np.array(Y_train)).long() # 输入 y 张量
Y_test = torch.from_numpy(np.array(Y_test)).long()

batch_size=2000
# Dataset
train_dataset = Data.TensorDataset(X_train, Y_train)
test_dataset = Data.TensorDataset(X_test, Y_test)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.3/(i+1)))
        self.inLayer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hiddenLayer = nn.Sequential(*layers)
        self.outLayer = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.inLayer(x)
        out = self.relu(out)
        out = self.hiddenLayer(out)
        out = self.outLayer(out)
        out = self.softmax(out)
        return out
# 网络初始化
input_size, hidden_size, num_classes, n_layers = 77, 128, 15, 3
model = NeuralNet(input_size, hidden_size, num_classes, n_layers)
# print(model)
# 模型的训练
num_epochs = 100
learning_rate = 0.01
# ------------------
# Loss and optimizer
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



model = torch.load('dnn3_data-20220329-03.pt')
#Precision & Recall,F1 Score

y_pred_list = []

with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
#         X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.numpy())


y_pred_list = [a.squeeze() for a in y_pred_list]
X=np.array(y_pred_list)

Z=np.concatenate(X).ravel()

targets = {0:'Benign', 1:'Bot', 2:'Brute Force -Web', 3:'Brute Force -XSS', 4:'DDoS attacks-LOIC-HTTP',5: 'DoS GoldenEye', 6:'DoS Hulk', 7:'DoS Slowhttptest', 8:'DoS Slowloris', 9:'FTP-Patator', 10:'Heartbleed', 11:'Infilteration', 12:'PortScan', 13:'SQL Injection', 14:'SSH-Patator'}


from sklearn.metrics import confusion_matrix, classification_report
# %config InlineBackend.figure_format = 'svg'
plt.figure(figsize=(20,16))

confusion_matrix_df = pd.DataFrame(confusion_matrix(Y_test, Z)).rename(columns=targets, index=targets)#

fig = sns.heatmap(confusion_matrix_df, annot=True)
heatmap = fig.get_figure()
# heatmap.savefig('heat1_newDara3_0330_03.jpg', dpi = 150)
#
print(classification_report(Y_test, Z,target_names=targets.values()))


cnf_matrix=confusion_matrix(Y_test, Z)
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
# FDR = FP/(TP+FP)
# # Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

print(TP,TN,FP,FN,ACC,TPR,TNR,FPR ,FNR)
#


