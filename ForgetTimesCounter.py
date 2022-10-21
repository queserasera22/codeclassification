"""
根据results.csv计算训练过程中样本的遗忘次数
第一行为labels
接下来每行为每次epoch下模型的预测结果
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Forget times counter')
# parser.add_argument('--predicts_path', type=str, default="models/codeberta-small_POJ104/predicts.csv",
#                     help='the path of predicts [default: models/codeberta-small_POJ104/predicts.csv]')
parser.add_argument('--dataset', type=str, default="POJ104",
                    help='the name of dataset [default: POJ104]')


def forgetTimesCount(result_path):
    df = pd.read_csv(result_path, header=None)

    counts = []
    for col_idx in df.columns:  # 每一列是对某一数据的label以及不同epoch下的预测结果
        column = df[col_idx]
        label = column[0]
        predicts = column[1:]

        prev_is_right = False  # 前面是否预测正确
        all_is_wrong = True  # 是否从未预测正确
        count = 0
        for item in predicts:
            is_right = (item == label)  # 当前是否预测正确
            if is_right:
                all_is_wrong = False
            if prev_is_right & ~is_right:  # 前面正确，当前错误，说明遗忘
                count = count + 1
            prev_is_right = is_right

        if all_is_wrong:  # 从未预测正确标记-1
            counts.append(-1)
        else:
            counts.append(count)
    return counts


def plt_bar(x_data, y_data, title):
    plt.bar(x_data, y_data)

    for i, j in zip(x_data, y_data):
        plt.text(i, j, "%.2f" % j, ha="center", va="bottom", fontsize=7)

    plt.title(title)
    plt.xlabel("Forget Times")
    plt.ylabel("Percent")
    plt.savefig("./fig/{}.png".format(title))
    plt.clf()


def get_distribution(path):
    counts = forgetTimesCount(path)
    print("unforget: ", counts.count(0))
    print("unforget: ", counts.count(0)/len(counts))
    print("mean forget times ", (np.sum(counts) + counts.count(-1)) / len(counts))
    index = range(-1, 6)
    times = [counts.count(i) * 100 / len(counts) for i in index]
    return times


args = parser.parse_args()
a_path = "models/codebert_{}/predicts.csv".format(args.dataset)
b_path = "models/codeberta_{}/predicts.csv".format(args.dataset)
# c_path = "cnn_text_classfication/models/CNN_{}/predicts.csv".format(args.dataset)

# index = range(-1, 8)
index = [str(i) for i in range(-1, 6)]
a = get_distribution(a_path)
b = get_distribution(b_path)
# c = get_distribution(c_path)

# plt_bar(index, a, "Codeberta-{}".format(args.dataset))
# plt_bar(index, b, "Codebert-{}".format(args.dataset))
# plt_bar(index, c, "CNN-{}".format(args.dataset))
x = np.arange(len(index))
# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.9, 2
# 每种类型的柱状图宽度
width = total_width / n

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
print(x)

fig = plt.figure(figsize=(9,4))

# 画柱状图
plt.bar(x, a, width=width, label="CodeBERT")
plt.bar(x + width, b, width=width, label="CodeBERTa")
# plt.bar(x + 2*width, c, width=width, label="CNN")

# 功能2
for i, j in zip(x, a):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
for i, j in zip(x + width, b):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
# for i, j in zip(x + 2 * width, c):
#     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)

plt.xticks(x + 0.5*width, index)

plt.title("Forget Events Distribution")
plt.xlabel("Forget Events",fontsize=14)
plt.ylabel("Percent", fontsize=14)
# 显示图例
plt.legend()

fig.tight_layout()

# 显示柱状图
plt.savefig("./fig/nfe/NFE_{}.png".format(args.dataset))
