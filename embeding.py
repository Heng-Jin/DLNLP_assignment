import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. read CSV
csv_file = 'Kaggle_news_train.csv'
df = pd.read_csv(csv_file)

# 2. load BERT tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {'index_id': row['Unnamed: 0'],
                'cleaned_text': row['input_data_cleaned'],
                'label': row['labels']}

dataset = TextDataset(df)

def get_token_embeddings(input_ids):
    input_ids = input_ids.to(device)
    with torch.no_grad():
        token_embeddings = model.embeddings.word_embeddings(input_ids)
    return token_embeddings

batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 4. save token embeddings as JSON file
output_folder = "Kaggle_news_train/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for batch in data_loader:
    index_ids = batch['index_id']
    cleaned_texts = batch['cleaned_text']
    labels = batch['label']

    # convert to token IDs
    token_ids_batch = \
    tokenizer.batch_encode_plus(cleaned_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)[
        'input_ids']

    embeddings_batch = get_token_embeddings(token_ids_batch)

    for index_id, embeddings, label in zip(index_ids, embeddings_batch, labels):
        embeddings_list = embeddings.tolist()

        if isinstance(label, torch.Tensor):
            label = label.cpu().item()

        print(index_id)
        data = {
            'embeddings': embeddings_list,
            'label': label
        }

        # save as JSON file
        output_file = f"{output_folder}{index_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f)


