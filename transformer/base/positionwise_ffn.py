import torch.nn as nn
from torch.nn import Module

from transformer.config.positionwise_ffn import PositionWiseFeedForwardConfig

class PositionWiseFeedForward(Module):
    def __init__(
            self,
            config: PositionWiseFeedForwardConfig
        ):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        self.linear1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x