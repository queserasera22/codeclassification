from torch import nn
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, num_labels, dropout=0.5, model_name="microsoft/codebert-base"):
        super(Classifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_id):
        cls_embedding = self.model(input_ids=input_id, return_dict=False)[1]

        # dropout_output = self.dropout(cls_embedding)
        # linear_output = self.linear(dropout_output)

        linear_output = self.linear(cls_embedding)
        output = self.softmax(linear_output)
        return output
        # return cls_embedding
