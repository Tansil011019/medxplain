import torch.nn as nn
from torch.nn import Module

from .multihead_attention import MultiHeadAttention
from .positionwise_ffn import PositionWiseFeedForward

from transformer.config.encoder import EncoderLayerConfig
from transformer.config.multihead_attention import MultiHeadAttentionConfig
from transformer.config.positionwise_ffn import PositionWiseFeedForwardConfig

class EncoderLayer(Module):
    def __init__(
        self,
        config: EncoderLayerConfig
    ):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_head = config.num_head
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        self.multihead_attention_config = MultiHeadAttentionConfig(
            embed_dim=self.embed_dim,
            num_head=self.num_head,
            dropout=self.dropout
        )
        self.multi_attn = MultiHeadAttention(self.multihead_attention_config)

        self.positionwise_ffn_config = PositionWiseFeedForwardConfig(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.ffn = PositionWiseFeedForward(self.positionwise_ffn_config)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        # Normalized (Pre-LN)
        norm1_x = self.norm1(x)
        attn_out = self.multi_attn(norm1_x, norm1_x, norm1_x, mask)
        # Residual Connection
        x = x + self.dropout1(attn_out)

        norm2_x = self.norm2(x)
        ff_out = self.ffn(norm2_x)
        x = x + self.dropout2(ff_out)

        return x