import torch
import torch.nn as nn
from torch.nn import Module

import math

from transformer.config.multihead_attention import MultiHeadAttentionConfig

class   MultiHeadAttention(Module):
    def __init__(
        self,
        config: MultiHeadAttentionConfig
    ):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_head = config.num_head
        self.dropout = config.dropout

        # Ensure that each attention head gets an equal share of the input vector (embed dim)
        assert self.embed_dim % self.num_head == 0, "Embedded dimension must be divisible by number of head"

        self.head_dim = self.embed_dim // self.num_head

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.softmax_similarity = nn.Softmax(dim=-1) # key similarity

    def split_head(self, proj):
        batch_size, seq_length, _= proj.size()
        return proj.view(batch_size, seq_length, self.num_head, self.head_dim).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # print("=" * 50)
        # print(f"Scaled Dot Products")
        # print(f"K transpose: {K.transpose(-2, -1)}")
        # print(f"Head dimension: {self.head_dim}")
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # print(f"Attention score: {attn_scores}")
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        # print(f"Attention Scores After Masking: {attn_scores}")
        attn_prob = self.softmax_similarity(attn_scores)
        # print(f"Attention Probability: {attn_prob}")
        attn_prob = self.attn_dropout(attn_prob)
        # print(f"Attention Probability After Dropout: {attn_prob}")
        output = torch.matmul(attn_prob, V)
        # print(f"Output: {output}")
        return output

    def combine_head(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

    def forward(self, q_input, k_input, v_input, mask=None):
        # Linear Projection
        Q = self.q_proj(q_input) # (batch size, token, embed_dim)
        K = self.k_proj(k_input)
        V = self.v_proj(v_input)

        # print("=" * 50)
        # print(f"Linear Projection")
        # print(f"Query: {Q}\nQuery Shape: {Q.shape}")
        # print(f"Key: {K}\nKey Shape: {K.shape}")
        # print(f"Value: {V}\nValue Shape: {V.shape}")

        # Split Num Head
        Q = self.split_head(Q) # (batch size, head, token, head_dim)
        K = self.split_head(K)
        V = self.split_head(V)
        
        # print("=" * 50)
        # print(f"Split Num Head")
        # print(f"Query: {Q}\nQuery Shape: {Q.shape}")
        # print(f"Key: {K}\nKey Shape: {K.shape}")
        # print(f"Value: {V}\nValue Shape: {V.shape}")

        # Dot Product
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Combine Head
        attn_output = self.combine_head(attn_output)
        # print("=" * 50)
        # print("Combining Head")
        # print(f"Attention Output: {attn_output}")
        # print(f"Attention Output Shape: {attn_output.shape}")

        # Output
        output = self.out_proj(attn_output)
        # print("=" * 50)
        # print(f"Output: {output}")
        # print(f"Output Shape: {output.shape}")
        return output



