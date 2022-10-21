import argparse
import time
import os
import warnings
import random
import logging


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from ptflops import get_model_complexity_info

import torch

from model import Classifier
from dataset import CodeDataset
from thop import profile



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(123456)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# 加载预训练模型 codebert,codeberta
# MODEL_NAME = "microsoft/codebert-base"
# SAVE_PATH = "models1/codebert_POJ104/model.pt"

# MODEL_NAME = "huggingface/CodeBERTa-small-v1"
# SAVE_PATH = "models/codeberta_POJ104/model.pt"

# test_data_path = '../dataset/CodeNet_Java250/test.jsonl'
# test_data_path = '../dataset/POJ104/test.jsonl'
# test_data_path = 'test.jsonl'

def input_constructor(input_res):
    batch = torch.randint(1, 1024, (1, *input_res)).cuda()
    return {"input_id": batch}


def evaluate(model_name, model_path, test_data_path, num_class, batch_size=32):
    model = Classifier(num_class, model_name=model_name)
    model.load_state_dict(torch.load(model_path))

    test_data = CodeDataset(test_data_path, model_name)
    test_dataloader = DataLoader(test_data, batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        model = model.cuda()

    model.eval()

    total_acc_test = 0
    predicts = []
    labels = []
    start_time = time.time()  # 程序开始时间
    with torch.no_grad():
        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            input_id = test_input.squeeze(1).to(device)
            output = model(input_id)

            predicts.append(output.argmax(dim=1).cpu().numpy())
            labels.append(test_label.cpu().numpy())

            # outputs.append(output.cpu().numpy())
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    end_time = time.time()  # 程序结束时间

    predicts = np.concatenate(predicts, 0)
    labels = np.concatenate(labels, 0)

    acc = accuracy_score(labels, predicts)
    precision = precision_score(labels, predicts, average='macro')
    recall = recall_score(labels, predicts, average='macro')
    f1 = f1_score(labels, predicts, average='macro')

    print(f'Test Accuracy: {acc : .5f}')
    print(f'Test Precision: {precision : .5f}')
    print(f'Test Recall: {recall : .5f}')
    print(f'Test F1: {f1 : .5f}')

    # print(f'Test Accuracy: {total_acc_test / len(test_data) * 100 : .3f}%')
    print('Test time: %.3fms' % ((end_time - start_time) / len(test_data) * 1000))

    # 效率评估 time + FLOPs
    # input = torch.randint(1, 1024, (1, 512)).to(device)
    # flops, params = profile(model, inputs=(input,))
    # print('flops is %.3fM' % (flops / 1e6))  # 打印计算量
    # print('params is %.3fM' % (params / 1e6))  # 打印参数

    macs, params = get_model_complexity_info(model, (512,), as_strings=False,
                                             print_per_layer_stat=False, verbose=True,
                                             input_constructor=input_constructor)

    print('flops is %.3fM' % (macs / 1e6))  # 打印计算量
    print('params is %.3fM' % (params / 1e6))  # 打印参数


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str, required=True,
                        help="The model name or path.")
    parser.add_argument("--model_path", default="models/codebert_POJ104/model.pt", type=str,
                        help="The path to the fine-tuned model")
    parser.add_argument("--test_data_file", default=None, type=str, required=True,
                        help="The input testing data file (a json file).")
    parser.add_argument("--num_labels", default=104, type=int,
                        help="The number of classes for classification task.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training.")


    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    evaluate(args.model_name, args.model_path, args.test_data_file, args.num_labels, args.batch_size)


if __name__ == "__main__":
    main()


# models = ["codebert", "codeberta"]
# model_names = {
#     "codebert": "microsoft/codebert-base",
#     "codeberta": "huggingface/CodeBERTa-small-v1"
# }
#
# datasets = ["POJ104", "Java250", "Python800"]
# test_data_paths = {
#     "POJ104": '../dataset/POJ104/test.jsonl',
#     "Java250": '../dataset/CodeNet_Java250/test.jsonl',
#     "Python800": '../dataset/CodeNet_Python800/test.jsonl',
# }
# class_nums = {
#     'POJ104': 104,
#     'Java250': 250,
#     'Python800': 800
# }
#
#
# # dirs = ["models", "models1", "models2", "models3", "models4"]
# dirs = ["models4"]
#
# for dir in dirs:
#     for model in models:
#         for dataset in datasets:
#             print("-------------------------------------------------")
#             print(dir, dataset, model)
#             MODEL_NAME = model_names[model]
#             SAVE_PATH = "{}/{}_{}/model.pt".format(dir, model, dataset)
#             test_data_path = test_data_paths[dataset]
#             class_num = class_nums[dataset]
#             evaluate(MODEL_NAME, SAVE_PATH, test_data_path, class_num)








