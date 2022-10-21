import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_cka import CKA
import torch
# import sys
# sys.path.append("..")
from model import Classifier
from dataset import CodeDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# DataLoader每次循环产生data,label，将data的shape由[batch_size,1,dim] 改为[batch_size,dim]
def my_collate(batch):
    items = default_collate(batch)
    return [items[0].squeeze(1), items[1]]


def nameToModel(name, dataset, isFinetuned=True):
    assert name in ['CodeBERT', 'CodeBERTa']
    assert dataset in ['POJ104', 'Java250', 'Python800']

    dataset_name = {
        'POJ104': 'POJ104',
        'Java250': 'CodeNet_Java250',
        'Python800': 'CodeNet_Python800'
    }
    class_num = {
        'POJ104': 104,
        'Java250': 250,
        'Python800': 800
    }

    test_data_path = '../dataset/{}/test.jsonl'.format(dataset_name[dataset])
    batch_size = 8

    if name == 'CodeBERT':
        model_name = "microsoft/codebert-base"
        model_path = "models/codebert_{}/model.pt".format(dataset)
        layers = ['model.embeddings']
        for i in range(12):
            layer = 'model.encoder.layer.{}.output'.format(i)
            layers.append(layer)

    else:
        model_name = "huggingface/CodeBERTa-small-v1"
        model_path = "models/codeberta_{}/model.pt".format(dataset)
        layers = ['model.embeddings']
        for i in range(6):
            layer = 'model.encoder.layer.{}.output'.format(i)
            layers.append(layer)

    model = Classifier(class_num[dataset], model_name=model_name)
    if isFinetuned:
        model.load_state_dict(torch.load(model_path))

    test_data = CodeDataset(test_data_path, model_name)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=lambda x: my_collate(x))


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    return model, test_dataloader, layers


def compare(name1, name2, dataset):
    model1, dataloader1, layers1 = nameToModel(name1, dataset, isFinetuned=False)
    model2, dataloader2, layers2 = nameToModel(name2, dataset, isFinetuned=False)
    cka = CKA(model1, model2,
              model1_name=name1,  # good idea to provide names to avoid confusion
              model2_name=name2,
              model1_layers=layers1,
              model2_layers=layers2,
              device='cuda')
    cka.compare(dataloader1, dataloader2)  # secondary dataloader is optional
    if not os.path.exists(f"./fig/cka_v/{dataset}/"):
        os.makedirs(f"./fig/cka_v/{dataset}/")
    cka.plot_results(save_path=f"./fig/cka_v/{dataset}/cka_{dataset}_{name1}_Pretrained.png")
    results = cka.export()  # returns a dict that contains model names, layer names and the CKA matrix
    return results


re = compare('CodeBERT', 'CodeBERT', 'Java250')
for key, value in re.items():
    print(key, value)
