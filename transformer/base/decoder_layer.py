import torch
import torch.nn as nn
from torch.nn import Module

from .multihead_attention import MultiHeadAttention
from .positionwise_ffn import PositionWiseFeedForward

from transformer.config.decoder import DecoderLayerConfig
from transformer.config.multihead_attention import MultiHeadAttentionConfig
from transformer.config.positionwise_ffn import PositionWiseFeedForwardConfig

class DecoderLayer(Module):
    def __init__(
        self,
        config: DecoderLayerConfig
    ):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        multi_attn_config = MultiHeadAttentionConfig(
            embed_dim=self.embed_dim,
            num_head=self.num_heads,
            dropout=self.dropout
        )
        self.multi_attn = MultiHeadAttention(multi_attn_config)

        cross_attn_config = MultiHeadAttentionConfig(
            embed_dim=self.embed_dim,
            num_head=self.num_heads,
            dropout=self.dropout
        )
        self.cross_attn = MultiHeadAttention(cross_attn_config)

        ffn_config = PositionWiseFeedForwardConfig(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.ffn = PositionWiseFeedForward(ffn_config)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        norm1_x = self.norm1(x)
        attn_out = self.multi_attn(norm1_x, norm1_x, norm1_x, tgt_mask)
        x = x + self.dropout1(attn_out)

        norm2_x = self.norm2(x)
        cross_attn_out = self.cross_attn(norm2_x, enc_out, enc_out, memory_mask)
        x = x + self.dropout2(cross_attn_out)

        norm3_x = self.norm3(x)
        ff_out = self.ffn(norm3_x)
        x = x + ff_out

        return x