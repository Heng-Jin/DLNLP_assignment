import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

import csv
import numpy as np
import logging
import pathlib
import os
import time

from datasets import CSVDataset
from utilities import plot_save, create_csv

# hyperparemeter
# input_dim = 768  
# hidden_dim = 768
# output_dim = 42  
# num_layers = 2
learning_rate = 2e-4
batch_size = 64
num_epochs = 15
pretrain = False

train_data_path = pathlib.Path.cwd()/ "Kaggle_news_train.csv"
val_data_path = pathlib.Path.cwd()/ "Kaggle_news_test.csv"

# computing set up
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU assigned
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# logging set up
parent_path = pathlib.Path.cwd()
if pretrain == False:
    model_save_path = parent_path / ("BERT_train_" + "epoch" + str(num_epochs) + "_lr" + str(learning_rate) + str(
        time.strftime("_%m_%d_%H_%M", time.localtime())))
else:
    model_save_path = parent_path / ("BERT_train_" + "epoch" + str(num_epochs) + "_lr" + str(learning_rate) + str(
        time.strftime("_%m_%d_%H_%M", time.localtime())))
model_save_path.mkdir(exist_ok=True)  # all outputs of this running will be saved in this path
log_path = model_save_path / ("BERT_train_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # loss = criterion(outputs, labels)
            preds = torch.argmax(outputs.logits, dim=1)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (preds == labels).sum().cpu().item()
            n += outputs.logits.shape[0]
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(outputs.logits, dim=1)
                test_acc_sum += (preds == labels.to(device)).float().sum().cpu().item()
                temp = torch.stack(
                    (preds, labels.to(device).int(), preds == labels.to(device)),
                    1).tolist()
                test_result_list.extend(temp)
                n2 += outputs.logits.shape[0]

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

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# load dataset
train_dataset = CSVDataset(train_data_path, tokenizer)
val_dataset = CSVDataset(val_data_path, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

if pretrain == False:
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=42)
    config.init_weights = True
    model = BertForSequenceClassification(config)
else:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=42)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

plot_save(Loss_list, Accuracy_valid_list, model_save_path)
