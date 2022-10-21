import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_cka import CKA
import torch
# import sys
# sys.path.append("..")
from cnn_text_classfication.dataset import CNNDataset
from model import Classifier
from dataset import CodeDataset
from cnn_text_classfication.model import CNN_Text
from cnn_text_classfication.data_loader import DataLoader as myDataLoader

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

        # model = Classifier(class_num[dataset], model_name=model_name)
        # model.load_state_dict(torch.load(model_path))
        #
        # test_data = CodeDataset(test_data_path, model_name)
        # test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=lambda x: my_collate(x))

        # layers.append('linear')
        # layers.append('softmax')

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


        # layers.append('linear')
        # layers.append('softmax')

    # else:
    #     save_path = './cnn_text_classfication/outputs/CNN_{}/model.pt'.format(dataset)
    #     data_path = './cnn_text_classfication/dataset/{}.pt'.format(dataset)
    #
    #     data = torch.load(data_path)
    #     model_source = torch.load(save_path)
    #     args = model_source["settings"]
    #
    #     model = CNN_Text(args)
    #     model.load_state_dict(model_source["model"])
    #
    #     model.eval()
    #
    #     test_data = CNNDataset(
    #         data['test']['src'],
    #         data['test']['label'],
    #         args.max_len,
    #         cuda=True,
    #     )
    #     test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    #     # layers = ['lookup_table', 'encoder_0', 'encoder_1', 'encoder_2', 'dropout']
    #     layers = ['lookup_table', 'encoder_0', 'encoder_1', 'encoder_2', 'dropout', 'logistic', 'softmax']

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


# def compareFinetuningBeforeAfter(name, dataset_name):
#     assert name in ['CodeBERT', 'CodeBERTa', 'cnn']
#     assert dataset_name in ['POJ104', 'Java250', 'Python800']
#
#     class_num = {
#         'POJ104': 104,
#         'Java250': 250,
#         'Python800': 800
#     }
#
#     if name == 'CodeBERT':
#         model_name = "microsoft/codebert-base"
#         model = Classifier(class_num[dataset_name], model_name=model_name)
#     elif name == 'CodeBERTa':
#         model_name = "huggingface/CodeBERTa-small-v1"
#         model = Classifier(class_num[dataset_name], model_name=model_name)
#     else:
#         save_path = './cnn_text_classfication/outputs/CNN_{}.pt'.format(dataset_name)
#         model_source = torch.load(save_path)
#         args = model_source["settings"]
#         model = CNN_Text(args)
#
#     model1, dataloader, layers = nameToModel(name, dataset_name)
#     cka = CKA(model, model1,
#               model1_name="{}-Pretrained".format(name),  # good idea to provide names to avoid confusion
#               model2_name="{}-Finetuned".format(name),
#               model1_layers=layers,
#               model2_layers=layers,
#               device='cuda')
#     cka.compare(dataloader)  # secondary dataloader is optional
#     if not os.path.exists(f"./fig/cka_v/{dataset_name}/"):
#         os.makedirs(f"./fig/cka_v/{dataset_name}/")
#     cka.plot_results(save_path=f"./fig/cka_v/{dataset_name}/cka_{dataset_name}_{name}_Pretrained_Finetuned.png")
#     results = cka.export()  # returns a dict that contains model names, layer names and the CKA matrix
#     return results


re = compare('CodeBERT', 'CodeBERT', 'Java250')
for key, value in re.items():
    print(key, value)

# re = compareFinetuningBeforeAfter('CodeBERTa', 'Python800')
# for key, value in re.items():
#     print(key, value)
