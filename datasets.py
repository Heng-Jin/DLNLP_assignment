import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer


class JSONDataset(Dataset):
    def __init__(self, folder_path, seq_length=32):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.seq_length = seq_length+1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        with open(file_path, "r") as f:
            data = json.load(f)

        embeddings = torch.tensor(data["embeddings"])
        label = torch.tensor(data["label"])

        # Padding or truncating the embeddings to the fixed sequence length
        if embeddings.size(0) < self.seq_length:
            pad_size = self.seq_length - embeddings.size(0)
            embeddings = torch.cat([embeddings[1:], torch.zeros(pad_size, embeddings.size(1))], dim=0)
        else:
            embeddings = embeddings[1:self.seq_length]

        return {"embeddings": embeddings, "label": label}


class CSVDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=32):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'input_data_cleaned']
        label = self.df.loc[idx, 'labels']

        # Tokenize the text and truncate/pad it to the desired length
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # Convert to Tensor
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

class CSVJSONDataset(Dataset):
    def __init__(self, csv_file, folder_path, seq_length=32):
        self.df = pd.read_csv(csv_file)
        self.seq_length = seq_length + 1
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # text = self.df.loc[idx, 'input_data_cleaned']
        label = self.df.loc[idx, 'labels']
        index = self.df.loc[idx, 'Unnamed: 0']

        file_path = os.path.join(self.folder_path, (str(index)+'.json'))
        with open(file_path, "r") as f:
            data = json.load(f)

        embeddings = torch.tensor(data["embeddings"])
        label = torch.tensor(label, dtype=torch.long)
        # print(type(embeddings), index)
        # Padding or truncating the embeddings to the fixed sequence length
        if embeddings.size(0) < self.seq_length:
            pad_size = self.seq_length - embeddings.size(0)
            embeddings = torch.cat([embeddings[1:], torch.zeros(pad_size, embeddings.size(1))], dim=0)
        else:
            embeddings = embeddings[1:self.seq_length]

        return {"embeddings": embeddings, "label": label}

