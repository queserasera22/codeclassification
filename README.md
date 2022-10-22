# Code Classification
This repository contains the code and data for the paper "On Differences Between Pre-trained Code Models with Different Sizes: Performance, Learning Dynamics, Representation Similarity, and Geometric Properties"

## Environment configuration
To reproduce our experiments, machines with GPUs and NVIDIA CUDA toolkit are required.

The environment dependencies are listed in the file "requirements.txt". You can create conda environment to install required dependencies:

```
conda create --name <env> --file requirements.txt
```

## Datasets
We use three different datasets in our code classification experiments, namely POJ104, Java250 and Python800. The datasets can be downloaded from the following sources:

* POJ104: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104
* Java250 and Python800: https://github.com/IBM/Project_CodeNet


The partition of datasets follows [CodeNet paper](https://arxiv.org/abs/2105.12655). For each dataset, 20\% of the data is used as a testing set, while the rest is divided in 4:1 for training and validation.


## Usage

### Fine-tuning
We focus on two wildly used pre-trained code models with different sizes, [CodeBERT](https://github.com/microsoft/CodeBERT) and [CodeBERTa](https://huggingface.co/huggingface/CodeBERTa-small-v1).

"main.py" is the training script of the CodeBERT/CodeBERTa pre- model. You can run  as following 
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name=microsoft/codebert-base \
    --train_data_file=../dataset/POJ104/train.jsonl \
    --eval_data_file=../dataset/POJ104/valid.jsonl \
    --save_path=models/codebert_POJ104 \
    --num_labels 104 \
    --batch_size 32 \
    --seed 456 2>&1| tee models/codebert_POJ104/train.log
```
For CodeBERTa, you should change the model_name into `huggingface/CodeBERTa-small-v1`, and also choose a new save_path you like of course.

### Evaluation
Run evaluate.py for Evaluation.
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name=microsoft/codebert-base \
    --model_path=models/codebert_POJ104 \
    --test_data_file=../dataset/POJ104/test.jsonl \
    --num_labels 104 \
    --batch_size 32 \
```

### Learning Dynamics
We conduct additional study investigating the learning dynamics of the two models from the perspective of forgetting.
"ForgetTimesCounter.py" is the script to count the number of forgetting events for every training examples. The counter works based on the file "results.csv", which records the prediction results of training examples for each epoch.

### Centered Kernel Alignment
We investigate the representations similarity within and across models using CKA.

Run "modelCompareCKA.py" for comparison between representations within and across models.


## Waiting to update...
