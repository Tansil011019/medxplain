import torch
import torch.nn as nn
from torch.nn import Module

import math

from transformer.config.positional_encoding import PositionalEncodingConfig

class PositionalEncoding(Module):
    def __init__(
        self,
        config: PositionalEncodingConfig
    ):
        super().__init__()
        self.dropout_p = config.dropout
        self.embed_dim = config.embed_dim
        self.max_length = config.max_length

        self.dropout = nn.Dropout(self.dropout_p)

        # TODO: study this part (its kinda new)
        pos_enc = torch.zeros(self.max_length, self.embed_dim)
        position = torch.arange(0,self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))

    def forward(self, x):
        print(f"Positional: {self.pos_enc[:, :x.size(1), :]}")
        x = x + self.pos_enc[:, :x.size(1), :]
        print(f"X Before Dropout: {x}")
        return self.dropout(x)