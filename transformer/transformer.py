import torch
import torch.nn as nn
from torch.nn import Module

from transformer.core.encoder import Encoder
from transformer.core.decoder import Decoder

from transformer.config.encoder import EncoderConfig
from transformer.config.decoder import DecoderConfig

class Transformer(Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_enc_layers, 
        num_dec_layers,
        max_length,
        dropout_p,
        src_pad_idx,
        tgt_pad_idx
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.max_length = max_length
        self.dropout_p = dropout_p
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        enc_config = EncoderConfig(
            num_layers=self.num_enc_layers,
            embed_dim=self.embed_dim,
            num_heads = self.num_heads,
            hidden_dim = self.hidden_dim,
            vocab_size=self.src_vocab_size,
            max_seq_length=self.max_length,
            dropout=self.dropout_p
        )
        self.encoder = Encoder(enc_config)

        dec_config = DecoderConfig(
            num_layers=self.num_dec_layers,
            embed_dim=self.embed_dim,
            num_heads = self.num_heads,
            hidden_dim = self.hidden_dim,
            vocab_size=self.tgt_vocab_size,
            max_seq_length=self.max_length,
            dropout=self.dropout_p
        )
        self.decoder = Decoder(dec_config)

    # the masking is kinda sus
    def generate_mask(self, src, tgt):

        src_pad_mask = (src == self.src_pad_idx)
        src_mha_mask = src_pad_mask.unsqueeze(1).unsqueeze(2) 

        _, tgt_len = tgt.shape
        tgt_pad_mask = (tgt == self.tgt_pad_idx)

        causal_mask = ~torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))

        decoder_self_attn_mha_mask = torch.logical_or(
            tgt_pad_mask.unsqueeze(1).unsqueeze(2),
            causal_mask.unsqueeze(0).unsqueeze(0)  
        )

        return src_mha_mask, decoder_self_attn_mha_mask

    def forward(self, src, tgt):
        print("=" * 50)
        print("Input")
        print(f"Source: {src}")
        print(f"Source Shape: {src.shape}")
        print(f"Target: {tgt}")
        print(f"Target shape: {tgt.shape}")
        print("=" * 50)
        print("Masking")
        src_mha_mask, decoder_self_attn_mha_mask = self.generate_mask(src, tgt)
        print(f"Source Padding Mask: {src_mha_mask}")
        print(f"Source Padding Mask Shape: {src_mha_mask.shape}")
        print(f"Target Padding Mask: {decoder_self_attn_mha_mask}")
        print(f"Target Padding Mask Shape: {decoder_self_attn_mha_mask.shape}")

        print("=" * 50)
        print(f"Encoder")
        memory = self.encoder(src, src_mha_mask)

        print("=" * 50)
        print(f"Decoder")
        decoder_output = self.decoder(tgt, memory, tgt_mask=decoder_self_attn_mha_mask, memory_mask=src_mha_mask)
        return decoder_output