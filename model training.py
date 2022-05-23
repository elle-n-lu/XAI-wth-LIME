import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

dataset = pd.read_csv("../datasets/DATA_COMBINE/combine2/combine2_withmin_moreMin.csv")

labels = dataset['Label']
features = dataset.loc[:, dataset.columns != 'Label'].astype('float64')

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(features)
features = scaler.transform(features)

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
#打印class和amount
# print(idx2class,d)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = features, features, labels, labels
# X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# print(labels.shape,type(labels))

X_train = torch.from_numpy(np.array(X_train)).float() # 输入 x 张量
X_test = torch.from_numpy(np.array(X_test)).float()
Y_train = torch.from_numpy(np.array(Y_train)).long() # 输入 y 张量
Y_test = torch.from_numpy(np.array(Y_test)).long()


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
#---------------------/////////////////////////////////////////////---------------------
#kfold 交叉验证
def k_fold1(k, X_train, y_train, num_epochs, learning_rate=0.01, batch_size=20000):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    test_accuracy, train_accuracy = [], []
    # losses = []
    # val_losses = []
    train_acc = []
    val_acc = []
    for i in range(k):
        fold_size = X_train.shape[0] // k
        val_start = i * fold_size
        # 获取训练集和验证集
        if i != k - 1:
            val_end = (i + 1) * fold_size
            X_test, y_test = X_train[val_start:val_end], y_train[val_start:val_end]
            X_train = torch.cat((X_train[0:val_start], X_train[val_end:]), dim=0)
            y_train = torch.cat((y_train[0:val_start], y_train[val_end:]), dim=0)
        else:  # 若是最后一折交叉验证
            X_test, y_test = X_train[val_start:], y_train[val_start:]  # 若不能整除，将多的case放在最后一折里
            X_train = X_train[0:val_start]
            y_train = y_train[0:val_start]

        input_size, hidden_size, num_classes, n_layers = 77, 128, 15, 3
        model = NeuralNet(input_size, hidden_size, num_classes, n_layers)
        # model = NeuralNet(77, 13)
        train_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_train, y_train), batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_test, y_test), batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        print('这是第',i,'折')
        for epoch in range(num_epochs):
            model.train()
            for j, (data, label_s) in enumerate(train_loader):
                # Move tensors to the configured device
                # images = images.reshape(-1, 28*28).to(device)
                # labels = labels.to(device)
                # Forward pass
                outputs = model(data.float())
                label_s = torch.squeeze(label_s.type(torch.LongTensor))
                loss = criterion(outputs, label_s)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (j+1)  % 10 == 0: #考虑10改成7,6可能不论什么数值都最后的batch的最后的echo会不打印，所以隐藏先。
                # 计算每个batch的准确率
                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total += label_s.size(0)
                correct += (predicted == label_s).sum().item()
                # train_acc = 100 * correct / total
                # 打印结果
                print('Epoch [{}/{}], Step [{}/{}], trainAccuracy: {}, Loss: {:.4f}'.format(epoch + 1, num_epochs, j + 1,
                                                                                       total_step,
                                                                                       100 * correct / total,
                                                                                       loss.item()))
                train_acc.append(100 * correct / total)

            # -----------------------------------
            # Test the model(每一个epoch打印一次)
            # -----------------------------------
            # In test phase, we don't need to compute gradients (for memory efficiency)

            # model.eval()
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for data, labe_ls in val_loader:
            #         # images = images.reshape(-1, 28*28).to(device)
            #         # labels = labels.to(device)
            #         outputs = model(data.float())
            #         labe_ls = torch.squeeze(labe_ls.type(torch.LongTensor))
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labe_ls.size(0)
            #         correct += (predicted == labe_ls).sum().item()
            #     #             y_pred_list.append(predicted.cpu().numpy())
            #     print('Accuracy of valid dataset: {} %'.format(100 * correct / total))
            #     val_acc.append(100 * correct / total)

        # test_accuracy.append(val_acc[-1])
        train_accuracy.append(train_acc[-1])
        train_acc_sum += train_acc[-1]
        # valid_acc_sum += val_acc[-1]
        print('-' * 10)

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    # print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))

    torch.save(model, 'dnn3_data-20220329-01.pt')##没有跑代码，所以没有该model
    # plt.xlabel("fold")
    # plt.ylabel("performance")
    # fold = range(1, k + 1)
    # # plt.plot(fold, test_accuracy, 'o-', color='lightgrey', label='test_accuracy')
    # plt.plot(fold, train_accuracy, 'o--', color='grey', label='train_accuracy')
    # plt.legend()
    # plt.show()
    return train_accuracy

def k_fold2(k, X_train, y_train, num_epochs, learning_rate=0.005, batch_size=5000):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    test_accuracy, train_accuracy = [], []
    # losses = []
    # val_losses = []
    train_acc = []
    val_acc = []
    for i in range(k):
        fold_size = X_train.shape[0] // k
        val_start = i * fold_size
        # 获取训练集和验证集
        if i != k - 1:
            val_end = (i + 1) * fold_size
            X_test, y_test = X_train[val_start:val_end], y_train[val_start:val_end]
            X_train = torch.cat((X_train[0:val_start], X_train[val_end:]), dim=0)
            y_train = torch.cat((y_train[0:val_start], y_train[val_end:]), dim=0)
        else:  # 若是最后一折交叉验证
            X_test, y_test = X_train[val_start:], y_train[val_start:]  # 若不能整除，将多的case放在最后一折里
            X_train = X_train[0:val_start]
            y_train = y_train[0:val_start]

        input_size, hidden_size, num_classes, n_layers = 77, 128, 15, 3
        model = NeuralNet(input_size, hidden_size, num_classes, n_layers)
        # model = NeuralNet(77, 13)
        train_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_train, y_train), batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_test, y_test), batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        print('这是第',i,'折')
        for epoch in range(num_epochs):
            model.train()
            for j, (data, label_s) in enumerate(train_loader):
                # Move tensors to the configured device
                # images = images.reshape(-1, 28*28).to(device)
                # labels = labels.to(device)
                # Forward pass
                outputs = model(data.float())
                label_s = torch.squeeze(label_s.type(torch.LongTensor))
                loss = criterion(outputs, label_s)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (j+1)  % 10 == 0: #考虑10改成7,6可能不论什么数值都最后的batch的最后的echo会不打印，所以隐藏先。
                # 计算每个batch的准确率
                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total += label_s.size(0)
                correct += (predicted == label_s).sum().item()
                # train_acc = 100 * correct / total
                # 打印结果
                print('Epoch [{}/{}], Step [{}/{}], trainAccuracy: {}, Loss: {:.4f}'.format(epoch + 1, num_epochs, j + 1,
                                                                                       total_step,
                                                                                       100 * correct / total,
                                                                                       loss.item()))
                train_acc.append(100 * correct / total)

            # -----------------------------------
            # Test the model(每一个epoch打印一次)
            # -----------------------------------
            # In test phase, we don't need to compute gradients (for memory efficiency)

            # model.eval()
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for data, labe_ls in val_loader:
            #         # images = images.reshape(-1, 28*28).to(device)
            #         # labels = labels.to(device)
            #         outputs = model(data.float())
            #         labe_ls = torch.squeeze(labe_ls.type(torch.LongTensor))
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labe_ls.size(0)
            #         correct += (predicted == labe_ls).sum().item()
            #     #             y_pred_list.append(predicted.cpu().numpy())
            #     print('Accuracy of valid dataset: {} %'.format(100 * correct / total))
            #     val_acc.append(100 * correct / total)

        # test_accuracy.append(val_acc[-1])
        train_accuracy.append(train_acc[-1])
        train_acc_sum += train_acc[-1]
        # valid_acc_sum += val_acc[-1]
        print('-' * 10)

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    # print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))

    torch.save(model, 'dnn3_data-20220329-02.pt')##没有跑代码，所以没有该model
    # plt.xlabel("fold")
    # plt.ylabel("performance")
    # fold = range(1, k + 1)
    # # plt.plot(fold, test_accuracy, 'o-', color='lightgrey', label='test_accuracy')
    # plt.plot(fold, train_accuracy, 'o--', color='grey', label='train_accuracy')
    # plt.legend()
    # plt.show()
    return train_accuracy

def k_fold3(k, X_train, y_train, num_epochs, learning_rate=0.001, batch_size=5000):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    test_accuracy, train_accuracy = [], []
    # losses = []
    # val_losses = []
    train_acc = []
    val_acc = []
    for i in range(k):
        fold_size = X_train.shape[0] // k
        val_start = i * fold_size
        # 获取训练集和验证集
        if i != k - 1:
            val_end = (i + 1) * fold_size
            X_test, y_test = X_train[val_start:val_end], y_train[val_start:val_end]
            X_train = torch.cat((X_train[0:val_start], X_train[val_end:]), dim=0)
            y_train = torch.cat((y_train[0:val_start], y_train[val_end:]), dim=0)
        else:  # 若是最后一折交叉验证
            X_test, y_test = X_train[val_start:], y_train[val_start:]  # 若不能整除，将多的case放在最后一折里
            X_train = X_train[0:val_start]
            y_train = y_train[0:val_start]

        input_size, hidden_size, num_classes, n_layers = 77, 128, 15, 3
        model = NeuralNet(input_size, hidden_size, num_classes, n_layers)
        # model = NeuralNet(77, 13)
        train_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_train, y_train), batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_test, y_test), batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        print('这是第',i,'折')
        for epoch in range(num_epochs):
            model.train()
            for j, (data, label_s) in enumerate(train_loader):
                # Move tensors to the configured device
                # images = images.reshape(-1, 28*28).to(device)
                # labels = labels.to(device)
                # Forward pass
                outputs = model(data.float())
                label_s = torch.squeeze(label_s.type(torch.LongTensor))
                loss = criterion(outputs, label_s)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (j+1)  % 10 == 0: #考虑10改成7,6可能不论什么数值都最后的batch的最后的echo会不打印，所以隐藏先。
                # 计算每个batch的准确率
                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total += label_s.size(0)
                correct += (predicted == label_s).sum().item()
                # train_acc = 100 * correct / total
                # 打印结果
                print('Epoch [{}/{}], Step [{}/{}], trainAccuracy: {}, Loss: {:.4f}'.format(epoch + 1, num_epochs, j + 1,
                                                                                       total_step,
                                                                                       100 * correct / total,
                                                                                       loss.item()))
                train_acc.append(100 * correct / total)

            # -----------------------------------
            # Test the model(每一个epoch打印一次)
            # -----------------------------------
            # In test phase, we don't need to compute gradients (for memory efficiency)

            # model.eval()
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for data, labe_ls in val_loader:
            #         # images = images.reshape(-1, 28*28).to(device)
            #         # labels = labels.to(device)
            #         outputs = model(data.float())
            #         labe_ls = torch.squeeze(labe_ls.type(torch.LongTensor))
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labe_ls.size(0)
            #         correct += (predicted == labe_ls).sum().item()
            #     #             y_pred_list.append(predicted.cpu().numpy())
            #     print('Accuracy of valid dataset: {} %'.format(100 * correct / total))
            #     val_acc.append(100 * correct / total)

        # test_accuracy.append(val_acc[-1])
        train_accuracy.append(train_acc[-1])
        train_acc_sum += train_acc[-1]
        # valid_acc_sum += val_acc[-1]
        print('-' * 10)

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    # print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))

    torch.save(model, 'dnn3_data-20220329-03.pt')##没有跑代码，所以没有该model
    # plt.xlabel("fold")
    # plt.ylabel("performance")
    # fold = range(1, k + 1)
    # # plt.plot(fold, test_accuracy, 'o-', color='lightgrey', label='test_accuracy')
    # plt.plot(fold, train_accuracy, 'o--', color='grey', label='train_accuracy')
    # plt.legend()
    # plt.show()
    return train_accuracy

def k_fold4(k, X_train, y_train, num_epochs, learning_rate=0.001, batch_size=2500):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    test_accuracy, train_accuracy = [], []
    # losses = []
    # val_losses = []
    train_acc = []
    val_acc = []
    for i in range(k):
        fold_size = X_train.shape[0] // k
        val_start = i * fold_size
        # 获取训练集和验证集
        if i != k - 1:
            val_end = (i + 1) * fold_size
            X_test, y_test = X_train[val_start:val_end], y_train[val_start:val_end]
            X_train = torch.cat((X_train[0:val_start], X_train[val_end:]), dim=0)
            y_train = torch.cat((y_train[0:val_start], y_train[val_end:]), dim=0)
        else:  # 若是最后一折交叉验证
            X_test, y_test = X_train[val_start:], y_train[val_start:]  # 若不能整除，将多的case放在最后一折里
            X_train = X_train[0:val_start]
            y_train = y_train[0:val_start]

        input_size, hidden_size, num_classes, n_layers = 77, 128, 15, 3
        model = NeuralNet(input_size, hidden_size, num_classes, n_layers)
        # model = NeuralNet(77, 13)
        train_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_train, y_train), batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(Data.TensorDataset(X_test, y_test), batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        print('这是第',i,'折')
        for epoch in range(num_epochs):
            model.train()
            for j, (data, label_s) in enumerate(train_loader):
                # Move tensors to the configured device
                # images = images.reshape(-1, 28*28).to(device)
                # labels = labels.to(device)
                # Forward pass
                outputs = model(data.float())
                label_s = torch.squeeze(label_s.type(torch.LongTensor))
                loss = criterion(outputs, label_s)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (j+1)  % 10 == 0: #考虑10改成7,6可能不论什么数值都最后的batch的最后的echo会不打印，所以隐藏先。
                # 计算每个batch的准确率
                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total += label_s.size(0)
                correct += (predicted == label_s).sum().item()
                # train_acc = 100 * correct / total
                # 打印结果
                print('Epoch [{}/{}], Step [{}/{}], trainAccuracy: {}, Loss: {:.4f}'.format(epoch + 1, num_epochs, j + 1,
                                                                                       total_step,
                                                                                       100 * correct / total,
                                                                                       loss.item()))
                train_acc.append(100 * correct / total)

            # -----------------------------------
            # Test the model(每一个epoch打印一次)
            # -----------------------------------
            # In test phase, we don't need to compute gradients (for memory efficiency)

            # model.eval()
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for data, labe_ls in val_loader:
            #         # images = images.reshape(-1, 28*28).to(device)
            #         # labels = labels.to(device)
            #         outputs = model(data.float())
            #         labe_ls = torch.squeeze(labe_ls.type(torch.LongTensor))
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labe_ls.size(0)
            #         correct += (predicted == labe_ls).sum().item()
            #     #             y_pred_list.append(predicted.cpu().numpy())
            #     print('Accuracy of valid dataset: {} %'.format(100 * correct / total))
            #     val_acc.append(100 * correct / total)

        # test_accuracy.append(val_acc[-1])
        train_accuracy.append(train_acc[-1])
        train_acc_sum += train_acc[-1]
        # valid_acc_sum += val_acc[-1]
        print('-' * 10)

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    # print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))


    torch.save(model, 'dnn3_data-20220329-04.pt')##没有跑代码，所以没有该model

    return train_accuracy
    # plt.xlabel("fold")
    # plt.ylabel("performance")
    # fold = range(1, k + 1)
    # # plt.plot(fold, test_accuracy, 'o-', color='lightgrey', label='test_accuracy')
    # plt.plot(fold, train_accuracy, 'o--', color='grey', label='train_accuracy')
    # plt.legend()
    # plt.show()


features = torch.tensor(features)#.to(torch.device("cuda"))#加cuda
labels = torch.tensor(labels)#.to(torch.device("cuda"))#加cuda
m1=k_fold1(5, features, labels, 50)
m2=k_fold2(5, features, labels, 100)
m3=k_fold3(10, features, labels, 100)
m4=k_fold4(10, features, labels,50)

plt.xlabel("fold")
plt.ylabel("performance")
fold = range(1, 6)
plt.plot(fold, m1, 'o-', color='lightgrey', label='C1')
# plt.plot(fold, test_accuracy, 'o-', color='lightgrey', label='test_accuracy')
plt.plot(fold, m2, 'o--', color='grey', label='C2')
plt.legend()
plt.show()

plt.xlabel("fold")
plt.ylabel("performance")
fold2 = range(1, 11)
plt.plot(fold2, m3, '*-', color='lightgrey', label='C3')
# plt.plot(fold, test_accuracy, 'o-', color='lightgrey', label='test_accuracy')
plt.plot(fold2, m4, '*--', color='grey', label='C4')
plt.legend()
plt.show()






