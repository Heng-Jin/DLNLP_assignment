import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output


class BertLSTMClassifier(nn.Module):
    def __init__(self, num_labels, hidden_size=768, lstm_hidden_size=256, num_layers=2, bidirectional=False, dropout=0.0):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * (2 if bidirectional else 1), num_labels)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state
        lstm_output, _ = self.lstm(last_hidden_state)
        cls_output = lstm_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class DistilBertLSTMClassifier(nn.Module):
    def __init__(self, num_labels, hidden_size=768, lstm_hidden_size=256, num_layers=1, bidirectional=False, dropout=0.3):
        super(DistilBertLSTMClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * (2 if bidirectional else 1), num_labels)

    def forward(self, input_ids, attention_mask):
        distilbert_outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        last_hidden_state = distilbert_outputs.last_hidden_state
        lstm_output, _ = self.lstm(last_hidden_state)
        cls_output = lstm_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits