import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import csv
import numpy as np
import logging
import pathlib
import os
import time

from models import LSTMClassifier
from datasets import JSONDataset
from utilities import plot_save, create_csv

# 超参数
input_dim = 768  # 根据实际任务调整
hidden_dim = 768
output_dim = 42  # 根据实际任务调整
num_layers = 12
learning_rate = 5e-5
batch_size = 64
num_epochs = 15

train_data_path = pathlib.Path.cwd() / "Kaggle_news_train"
val_data_path = pathlib.Path.cwd() / "Kaggle_news_test"

# computing set up
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # GPU assigned
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# logging set up
parent_path = pathlib.Path.cwd()
model_save_path = parent_path / ("LSTM_train_32_" + "epoch" + str(num_epochs) + "_lr" + str(learning_rate)  + str(time.strftime("_%m_%d_%H_%M", time.localtime())))
model_save_path.mkdir(exist_ok=True)  # all outputs of this running will be saved in this path
log_path = model_save_path / ("LSTM_train_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')

Loss_list = []
Accuracy_train_list = []
Accuracy_valid_list = []

def train(net, train_iter, valid_iter, criterion, optimizer, num_epochs):
    '''
    training loop, model saving and inference of test data will be implemented after each epoch.
    Args:
        net: network to be trained
        train_iter: training dataloder
        test_iter: test dataloder
        criterion: loss function
        optimizer: torch.optim
        num_epochs: number of training epoch

    Returns:

    '''
    net = net.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)
    whole_batch_count = 0
    # training loop
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # trainning mode
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for batch in train_iter:
            embeddings = batch["embeddings"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (outputs.argmax(dim=1) == labels).sum().cpu().item()
            n += outputs.shape[0]
            whole_batch_count += 1
            batch_count += 1
            temp_loss = train_loss_sum / whole_batch_count
            Loss_list.append(loss.item())
            logging.info('-epoch %d, batch_count %d, sample nums %d, loss temp %.4f, train acc %.3f, time %.1f sec,'
                  % (epoch + 1, batch_count, n, loss.item(), train_acc_sum / n, time.time() - start))
            print('-epoch %d, batch_count %d, sample nums %d, loss temp %.4f, train acc %.3f, time %.1f sec'
                  % (epoch + 1, batch_count, n, loss.item(), train_acc_sum / n, time.time() - start))

        # test dataset inference will be done after each epoch
        with torch.no_grad():
            net.eval()  # evaluate mode
            test_acc_sum, n2 = 0.0, 0
            test_result_list = []
            for batch in valid_iter:
                embeddings = batch["embeddings"].to(device)
                labels = batch["label"].to(device)
                outputs = model(embeddings)
                test_acc_sum += (outputs.argmax(dim=1) == labels.to(device)).float().sum().cpu().item()
                temp = torch.stack(
                    (outputs.argmax(dim=1).int(), labels.to(device).int(), outputs.argmax(dim=1) == labels.to(device)),
                    1).tolist()
                test_result_list.extend(temp)
                n2 += labels.shape[0]

        temp_acc_test = test_acc_sum / n2
        Accuracy_valid_list.append(temp_acc_test)
        logging.info('---epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec---'
                     % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_test, time.time() - start))
        print('---epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec---'
              % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_test, time.time() - start))

        result_path = model_save_path / ("epoch_" + str(epoch) + "_lr_" + str(learning_rate) +"_test_result.csv")
        create_csv(result_path, test_result_list)

        torch.save(net.state_dict(),
                   model_save_path / ("epoch_" + str(epoch) + "_lr_" + str(learning_rate) + "_" + str(
                       time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))


# 数据预处理，加载数据集
# 请根据实际任务加载并预处理数据
train_dataset = JSONDataset(train_data_path)
val_dataset = JSONDataset(val_data_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建模型、损失函数和优化器
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

plot_save(Loss_list, Accuracy_valid_list, model_save_path)