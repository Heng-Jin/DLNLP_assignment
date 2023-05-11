import pandas as pd
import numpy as np

# 读取数据集
data = pd.read_csv("Kaggle_news_train.csv")

# 选择 5% 的数据
sampled_data = data.sample(frac=0.60, random_state=42)

# 遍历 5% 的数据并对其标签进行打乱
for index, row in sampled_data.iterrows():
    current_label = row['labels']
    all_labels = list(range(0, 42)) # 假设总共有 42 个标签
    all_labels.remove(current_label)
    new_label = np.random.choice(all_labels, 1)[0] # 随机选择一个新的标签，不包括当前标签
    sampled_data.at[index, 'labels'] = new_label

# 更新原始数据集中打乱后的标签
data.update(sampled_data)

# 保存新的 CSV 文件
data.to_csv("Kaggle_news_train_noisy60.csv", index=False)

# 读取原始数据集和新生成的数据集
original_data = pd.read_csv("Kaggle_news_train.csv")
new_data = pd.read_csv("Kaggle_news_train_noisy60.csv")

# 计算不一致的标签数量
inconsistent_labels = (original_data['labels'] != new_data['labels']).sum()

print(f"不一致的标签数量: {inconsistent_labels}")