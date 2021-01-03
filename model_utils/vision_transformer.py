import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.embeddingLayer = nn.Conv2d(3, 512, 16, 16)
        self.positionalEncoding = PositionalEncoding(512, max_len=4)
        self.transformerEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, 512, activation="gelu"), 6)
        cls_tensor = torch.randn(1, 1, 512).repeat(10, 1, 1)
        self.cls = nn.Parameter(cls_tensor)

    def forward(self, x):
        x = self.embeddingLayer(x)

        n, c, w, h = x.shape
        x = torch.reshape(x, [n, h * w, c])

        x = self.positionalEncoding(x)

        x = torch.cat((self.cls, x), 1)

        x = self.transformerEncoder(x)
        return x
