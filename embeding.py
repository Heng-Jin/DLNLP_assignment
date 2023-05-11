import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 读取 CSV 文件
csv_file = 'Kaggle_news_train.csv'
df = pd.read_csv(csv_file)

# 2. 加载预训练的 BERT tokenizer 和 BERT 模型
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

# # 3. 获取 embeddings.word_embeddings 层的 token embeddings
# def get_token_embeddings(text):
#     input_ids = tokenizer.encode(text, return_tensors='pt')
#     input_ids = input_ids.to(device)
#     with torch.no_grad():
#         token_embeddings = model.embeddings.word_embeddings(input_ids)
#     return token_embeddings

def get_token_embeddings(input_ids):
    input_ids = input_ids.to(device)
    with torch.no_grad():
        token_embeddings = model.embeddings.word_embeddings(input_ids)
    return token_embeddings

batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 4. 保存 token embeddings 为 JSON 文件
output_folder = "Kaggle_news_train/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for batch in data_loader:
    index_ids = batch['index_id']
    cleaned_texts = batch['cleaned_text']
    labels = batch['label']

    # 将清理好的文本转换为 token IDs
    token_ids_batch = \
    tokenizer.batch_encode_plus(cleaned_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)[
        'input_ids']

    embeddings_batch = get_token_embeddings(token_ids_batch)

    for index_id, embeddings, label in zip(index_ids, embeddings_batch, labels):
        embeddings_list = embeddings.tolist()

        if isinstance(label, torch.Tensor):
            label = label.cpu().item()

        # 将 token embeddings 和 label 保存到字典中
        print(index_id)
        data = {
            'embeddings': embeddings_list,
            'label': label
        }

        # 保存为 JSON 文件
        output_file = f"{output_folder}{index_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f)

# for index, row in df.iterrows():
#     index_id = row['Unnamed: 0']
#     cleaned_text = row['input_data_cleaned']
#     label = row['labels']
#
#     embeddings = get_token_embeddings(cleaned_text)
#
#     embeddings_list = embeddings.squeeze(0).tolist()
#
#     # 将 token embeddings 和 label 保存到字典中
#     data = {
#         'embeddings': embeddings_list,
#         'label': label
#     }
#
#     # 保存为 JSON 文件
#     output_file = f"{output_folder}{index_id}.json"
#     with open(output_file, "w") as f:
#         json.dump(data, f)
