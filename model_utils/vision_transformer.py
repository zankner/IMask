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
    def __init__(self, hidden_size, patch_size, model_dim, num_heads,
                 num_layers, img_size):
        super(VisionTransformer, self).__init__()
        self.embeddingLayer = nn.Conv2d(3, hidden_size, patch_size, patch_size)
        self.positionalEncoding = PositionalEncoding(hidden_size,
                                                     max_len=(img_size //
                                                              patch_size)**2)
        self.transformerEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim,
                                       num_heads,
                                       model_dim,
                                       activation="gelu"), num_layers)
        cls_tensor = torch.randn(1, 1, hidden_size)
        self.cls = nn.Parameter(cls_tensor)
        mask_token = torch.randn(1, 1, hidden_size)
        self.mask_token = nn.Parameter(mask_token)

    def forward(self, x, unmasked, masked, swapped):
        x = self.embeddingLayer(x)

        n, c, w, h = x.shape
        x = torch.reshape(x, [n, h * w, c])

        swapped_embedding_ids = unmasked[:len(swapped)]
        swapped_embeddings = x[:, swapped_embedding_ids, :].detach()
        x[:, swapped_embedding_ids, :] = swapped_embeddings

        x[:, masked, :] = self.mask_token

        x = self.positionalEncoding(x)

        x = torch.cat((self.cls.repeat(n, 1, 1), x), 1)

        x = self.transformerEncoder(x)
        return x
