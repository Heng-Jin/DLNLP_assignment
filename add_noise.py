import pandas as pd
import numpy as np

# read original dataset
data = pd.read_csv("Kaggle_news_train.csv")

# random select data samples
sampled_data = data.sample(frac=0.60, random_state=42)

# mix label
for index, row in sampled_data.iterrows():
    current_label = row['labels']
    all_labels = list(range(0, 42))
    all_labels.remove(current_label)
    new_label = np.random.choice(all_labels, 1)[0] # exclude true label
    sampled_data.at[index, 'labels'] = new_label

data.update(sampled_data)

# save new CSV file
data.to_csv("Kaggle_news_train_noisy60.csv", index=False)

# read both datasets
original_data = pd.read_csv("Kaggle_news_train.csv")
new_data = pd.read_csv("Kaggle_news_train_noisy60.csv")

# check labels
inconsistent_labels = (original_data['labels'] != new_data['labels']).sum()

print(f"Number of unmatch label samples: {inconsistent_labels}")