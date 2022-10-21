# Code Classification
This repository contains the code and data for the paper "On Differences Between Pre-trained Code Models with Different Sizes: Performance, Learning Dynamics, Representation Similarity, and Geometric Properties"

## Environment configuration
To reproduce our experiments, machines with GPUs and NVIDIA CUDA toolkit are required.

The environment requirements are listed in the file "requirements.txt". You can create conda environment to install required dependencies:

```
conda create --name codebert python=3
conda activate codebert
pip install -r requirements.txt
```



## Datasets
We use three different datasets in our code classification experiments, namely POJ104, Java250 and Python800. The datasets can be downloaded from the following sources:

* POJ104: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104
* Java250 and Python800: https://github.com/IBM/Project_CodeNet


The partition of datasets follows [CodeNet paper](https://arxiv.org/abs/2105.12655). For each dataset, 20\% of the data is used as a testing set, while the rest is divided in 4:1 for training and validation.


## Run

Waiting to update...
