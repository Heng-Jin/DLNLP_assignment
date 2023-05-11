import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import font_manager
import json
import csv

# 读取数据
# df = pd.read_csv('Kaggle_news_train_5class.csv')
# category_counts = df['category'].value_counts()
#
# # 设置字体
# font_path = r'D:\UCL_codes\0141\assignment\OpenSans-Bold.ttf' # 修改为你的字体文件路径
# my_font = font_manager.FontProperties(fname=font_path, size=16)
#
# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.bar(category_counts.index, category_counts.values, width=0.6, color=['#E24A33', '#348ABD', '#988ED5', '#8EBA42', '#FFB5B8'])
# ax.set_xticklabels(category_counts.index, fontproperties=my_font, rotation='vertical')
# ax.set_xlabel('Category', fontproperties=my_font)
# ax.set_ylabel('Count', fontproperties=my_font)
# ax.set_title('Category Counts of \'News Category Dataset\' Training Data', fontproperties=my_font, fontsize=18)
# ax.tick_params(axis='both', labelsize=14)
# # plt.show()
# plt.savefig('bar_chart_2.png', bbox_inches='tight')
#

# df = pd.read_csv('Kaggle_news_train.csv')
# plt.title("Category Counts of \'News Category Dataset\' Training Data")
# D:\UCL_codes\0141\assignment\OpenSans-Bold.ttf


df = pd.read_csv('Kaggle_news_train.csv')
groundtruth= dict(zip(df['Unnamed: 0'], df['labels']))

df = pd.read_csv('Kaggle_news_train_noisy60.csv')
labels= dict(zip(df['Unnamed: 0'], df['labels']))
title= ['Confusion Matrix between groundtruth labels and noisy labels', "Shuffled Labels %", "Groundtruth Labels %"]
name = 'label_confusion.png'
#
df = pd.read_csv('Kaggle_news_categories.csv')
categories= dict(zip(df['Unnamed: 0'], df['category']))
print(categories)
#
def compare_groundtruth_and_shuffled(groundtruth_dict, shuffled_dict, label_name_dict, title, name):
    # 获取所有唯一标签
    unique_labels = sorted(set(groundtruth_dict.values()))

    # 计算标签数量
    num_labels = len(unique_labels)

    # 初始化矩阵
    matrix = np.zeros((num_labels, num_labels), dtype=int)

    # 创建标签到索引的映射
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    unique_label_names = [label_name_dict[label] for label in unique_labels]

    # 填充矩阵
    for sample_id, gt_label in groundtruth_dict.items():
        shuffled_label = shuffled_dict[sample_id]
        matrix[label_to_idx[gt_label], label_to_idx[shuffled_label]] += 1

    # 计算百分比
    matrix_percentage = matrix / matrix.sum(axis=1, keepdims=True) * 100

    # 可视化矩阵
    plt.figure(figsize=(num_labels, num_labels))
    sns.set(font_scale=2)
    sns.heatmap(matrix_percentage, annot=True, fmt=".1f", cmap="coolwarm", xticklabels=unique_label_names,
                yticklabels=unique_label_names)
    plt.title(title[0])
    plt.xlabel(title[1])
    plt.ylabel(title[2])
    plt.savefig(name, bbox_inches='tight')
#
compare_groundtruth_and_shuffled(groundtruth, labels, categories, title, name)
#
#
# 读取csv文件，第一列为模型预测标签，第二列为groundtruth标签
df = pd.read_csv(r'D:\UCL_codes\0141\assignment\nlp_main\BERT_train_noisy60_epoch15_lr2e-05_05_10_02_36\epoch_1_lr_2e-05_test_result.csv', header=None)
df = list(df[0][1:])
y_pred = dict()
y_true = dict()
for i,item in enumerate(df):
    temp = item.strip('[]').split(',')
    print(temp)
    y_pred[i] = int(temp[0].strip())
    y_true[i] = int(temp[1].strip())

title= ['Confusion Matrix of BERT prediction results', "Prediction Labels %", "Groundtruth Labels %"]
name = 'BERT_confusion_noisy.png'

compare_groundtruth_and_shuffled(y_true, y_pred, categories, title, name)
#
# # 通过字典将标签编号映射为类别名称
# label_dict = categories
# labels = [label_dict[i] for i in range(len(label_dict))]
#
# # 生成混淆矩阵
# # 计算混淆矩阵
# num_labels = len(label_dict)
# matrix = confusion_matrix(y_true, y_pred, labels=range(num_labels))
#
# # 可视化混淆矩阵
# fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
# im = ax.imshow(matrix, cmap='Blues')
#
# # 显示颜色条
# cbar = ax.figure.colorbar(im, ax=ax)
#
# # 显示矩阵中的数值
# for i in range(num_labels):
#     for j in range(num_labels):
#         text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="b")
#
# # 替换xticklabels和yticklabels为类别名称
# ax.set_xticks(range(num_labels))
# ax.set_yticks(range(num_labels))
# ax.set_xticklabels([label_dict[i] for i in range(num_labels)], rotation=90)
# ax.set_yticklabels([label_dict[i] for i in range(num_labels)])
# ax.set_xlabel('Prediction Result')
# ax.set_ylabel('Groundtruth Label')
#
# # 添加图像标题
# ax.set_title('Confusion Matrix')
# plt.rcParams.update({'font.size': 7})
# # 调整图像边框和显示
# fig.tight_layout()
# plt.show()
#
#
# import matplotlib.pyplot as plt
#
# # results = {
# #     "LSTM": [0.4980, 0.4999, 0.4883, 0.4817, 0.4693, 0.4130],
# #     "BERT": [0.6896, 0.6841, 0.6816, 0.6755, 0.6640, 0.6436],
# #     "BERT+LSTM": [0.6781, 0.6766, 0.6749, 0.6680, 0.6427, 0.6405]
# # }
#
# results = {
#     "LSTM": [0.5451, 0.5395, 0.5344, 0.4817, 0.5093, 0.4130],
#     "BERT": [0.6896, 0.6841, 0.6816, 0.6757, 0.6640, 0.6436],
# }
#
# noise_levels = [0, 0.05, 0.1, 0.2, 0.4, 0.6]
#
# colors = ["red", "green", "blue"]
# markers = ["o", "^", "s"]
#
# for (model, performance), color, marker in zip(results.items(), colors, markers):
#     plt.plot(noise_levels, performance, label=model, color=color, marker=marker)
#     base_performance = performance[0]
#     for noise, perf in zip(noise_levels[1:], performance[1:]):
#         drop_percentage = (base_performance - perf) / base_performance * 100
#         plt.annotate(f'-{drop_percentage:.1f}%', (noise, perf), textcoords="offset points", xytext=(0,10), ha='center')
#
# plt.xlabel('Noise level')
# plt.ylabel('Performance')
# plt.title('Performance degradation under different noise levels (best acc in all LR)')
# plt.legend(loc='lower left')
# plt.grid(True)  # Optional, add grid
# # plt.show()
# plt.savefig('line_chart_1.png', bbox_inches='tight', dpi=300)
