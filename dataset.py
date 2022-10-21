from jsonlines import jsonlines
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np


def readJsonl(path):
    codes = []
    labels = []
    with open(path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):  # 每一行读取后都是一个json，可以按照key去取对应的值
            codes.append(item['code'])
            labels.append(item['label'])
    return codes, labels


class CodeDataset(Dataset):
    def __init__(self, data_path, model_name):
        codes, labels = readJsonl(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.codes = [self.tokenizer(code, padding='max_length', max_length=512, truncation=True,
                                     return_tensors='pt')['input_ids'] for code in codes]
        self.labels = [int(label) for label in labels]  # 字符转int 注意这里的label应该从0开始
        self.len = len(self.labels)

    def classes(self):
        return self.labels

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_codes(self, idx):
        return np.array(self.codes[idx])

    def __getitem__(self, idx):
        batch_codes = self.get_batch_codes(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_codes, batch_labels

    def __len__(self):
        return self.len
