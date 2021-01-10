import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, vision_transformer, input_dim, dense_1_dim, dense_2_dim,
                 output_dim, dropout):
        super(Classifier, self).__init__()
        self.vision_transformer = vision_transformer
        self.dense1 = nn.Linear(input_dim, dense_1_dim)
        self.dense2 = nn.Linear(dense_1_dim, dense_2_dim)
        self.dense3 = nn.Linear(dense_2_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.vision_transformer(x, [], [], [])[:, 0, :]
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        output = F.log_softmax(x, dim=1)
        return output