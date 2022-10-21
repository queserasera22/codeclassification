import argparse
import logging
import os
import warnings
import random

import numpy as np
import pandas as pd
import torch
from jsonlines import jsonlines
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from model import Classifier
from dataset import CodeDataset
import csv

# MODEL_NAME = "microsoft/codebert-base"
# save_path = "models1/codebert_POJ104"

# MODEL_NAME = "huggingface/CodeBERTa-small-v1"
# save_path = "models1/codeberta-small_Java250"

# train_data_path = '../dataset/POJ104/train.jsonl'
# val_data_path = '../dataset/POJ104/valid.jsonl'
# test_data_path = '../dataset/POJ104/test.jsonl'
# train_data_path = '../dataset/CodeNet_Java250/train.jsonl'
# val_data_path = '../dataset/CodeNet_Java250/valid.jsonl'
# test_data_path = '../dataset/CodeNet_Java250/test.jsonl'
# train_data_path = '../dataset/CodeNet_Python800/train.jsonl'
# val_data_path = '../dataset/CodeNet_Python800/valid.jsonl'
# test_data_path = '../dataset/CodeNet_Python800/test.jsonl'


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def getLabels(path):
    labels = []
    with open(path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):  # 每一行读取后都是一个json，可以按照key去取对应的值
            labels.append(int(item['label']))
    return labels


def train(args):
    # 加载预训练模型 codebert,codeberta
    model = Classifier(num_labels=args.num_labels, model_name=args.model_name)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_data = CodeDataset(args.train_data_file, args.model_name)
    val_data = CodeDataset(args.eval_data_file, args.model_name)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    model.to(args.device)
    criterion.to(args.device)

    os.makedirs(args.save_path, exist_ok=True)
    header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']

    with open(args.save_path + "/results.csv", 'w', encoding='UTF8', newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(header)
        with open(args.save_path + "/predicts.csv", 'w', encoding='UTF8', newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow(getLabels(args.train_data_file))

            best_accuracy = 0  # 记录模型最好的accuracy，用于保存最好的模型

            for epoch_num in tqdm(range(args.epoch)):
                # 定义两个变量，用于存储训练集的准确率和损失
                total_acc_train = 0
                total_loss_train = 0

                predicts = []
                model.train()
                # 进度条函数tqdm
                for train_input, train_label in tqdm(train_dataloader):
                    train_label = train_label.to(args.device)
                    # mask = train_input['attention_mask'].to(device)
                    input_id = train_input.squeeze(1).to(args.device)

                    # 通过模型得到输出
                    output = model(input_id)

                    predicts.extend([item.item() for item in output.argmax(dim=1)])

                    # 计算损失
                    batch_loss = criterion(output, train_label)
                    if args.n_gpu > 1:
                        batch_loss = batch_loss.mean()
                    total_loss_train += batch_loss.item()
                    # 计算精度
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc
                    # 模型更新
                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                writer2.writerow(list(predicts))

                # ------ 验证模型 -----------
                # 定义两个变量，用于存储验证集的准确率和损失
                total_acc_val = 0
                total_loss_val = 0
                # 不需要计算梯度
                with torch.no_grad():
                    model.eval()
                    # 循环获取数据集，并用训练好的模型进行验证, 保存最好的模型
                    for val_input, val_label in val_dataloader:
                        # 如果有GPU，则使用GPU，接下来的操作同训练
                        val_label = val_label.to(args.device)
                        input_id = val_input.squeeze(1).to(args.device)

                        output = model(input_id)

                        batch_loss = criterion(output, val_label)
                        if args.n_gpu > 1:
                            batch_loss = batch_loss.mean()
                        total_loss_val += batch_loss.item()

                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc

                if best_accuracy < (total_acc_val / len(val_data)):
                    best_accuracy = total_acc_val / len(val_data)
                    torch.save(model.state_dict(), args.save_path + "/model.pt")

                print(
                    f'''Epochs: {epoch_num + 1} 
                              | Train Loss: {total_loss_train / len(train_data): .3f} 
                              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
                              | Val Loss: {total_loss_val / len(val_data): .3f} 
                              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
                result = [epoch_num + 1, total_loss_train / len(train_data), total_acc_train / len(train_data),
                          total_loss_val / len(val_data), total_acc_val / len(val_data)]
                writer1.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str, required=True,
                        help="The model name or path.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--save_path", default="../", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a json file).")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="The number of classes for classification task.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch", type=int, default=20,
                        help="random seed for initialization")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    set_seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    train(args)


if __name__ == "__main__":
    main()
