
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import lime
import lime.lime_tabular
# %matplotlib inline

dataset = pd.read_csv("../datasets/DATA_COMBINE/combine2/combine2_withmin.csv")
# print(dataset[dataset['Label']=='DoS GoldenEye'])
# print(dataset[dataset['Label']=='DoS Slowloris'])
# Splitting dataset into features and labels.
labels = dataset['Label']
features = dataset.loc[:, dataset.columns != 'Label'].astype('float64')
# print(labels[810025])
# print(len(labels))
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(features)
features = scaler.transform(features)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE.fit(labels)
labels2 = LE.transform(labels)
#
print(labels2[2100014])
print(labels2[2100018])

d = LE.inverse_transform(labels2)
d = pd.Series(d)
d.unique()
# print(d)
class2idx=dict(enumerate(d.unique().flatten(), 0))
idx2class = {v: k for k, v in class2idx.items()}
idx2class_2 = {v: k for v, k in class2idx.items()}
d=dataset['Label'].value_counts()
# print(idx2class)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = features, features, labels2, labels2
# X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

X_train = torch.from_numpy(np.array(X_train)).float() # 输入 x 张量
X_test = torch.from_numpy(np.array(X_test)).float()
Y_train = torch.from_numpy(np.array(Y_train)).long() # 输入 y 张量
Y_test = torch.from_numpy(np.array(Y_test)).long()

batch_size=5000
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

model = torch.load('dnn3_data-20220329-04.pt')
def batch_predict(data, model=model):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """
    X_tensor = torch.from_numpy(data).float()
    model.eval()
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_tensor = X_tensor.to(device)
    logits = model(X_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

#------------------LIME部分：：：：：----------------------------

# features = torch.tensor(features)
features_names = [col.replace(' ', '') for col in dataset.columns]
targets = ['Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS', 'DDoS attacks-LOIC-HTTP', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS Slowloris', 'FTP-Patator', 'Heartbleed', 'Infilteration', 'PortScan', 'SQL Injection', 'SSH-Patator']
explainer = lime.lime_tabular.LimeTabularExplainer(features, feature_names=features_names,kernel_width=0.3, class_names=targets, discretize_continuous=True)# feature_selection='lasso_path'

# 解释某一个样本，此处解释第五个样本概率[0.6601497  0.33985034]
from sklearn.linear_model import LinearRegression
local_model = LinearRegression()
exp = explainer.explain_instance(features[1701005],
                                batch_predict,
                                num_features=10,
                                num_samples=750,
                                model_regressor=local_model,
                                top_labels=15,
                                )

exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('./img/510010/1701005.html')
# exp.as_pyplot_figure(label=2)
# plt.savefig('./img/810010/n2.jpg')
# exp.as_pyplot_figure(label=3)
# plt.savefig('./img/810010/n3.jpg')
# exp.as_pyplot_figure(label=6)
# plt.savefig('./img/810010/n6.jpg')
# exp.as_pyplot_figure(label=11)
# plt.savefig('./img/810010/n11.jpg')
# exp.as_pyplot_figure(label=13)
# plt.savefig('./img/810010/n13.jpg')
