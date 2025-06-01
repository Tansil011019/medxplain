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
        self.src_pad_idx = src_pad_idx,
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
            num_layers=self.num_enc_layers,
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

        _, tgt_len = tgt.shape
        tgt_pad_mask = (tgt == self.tgt_pad_idx)

        causal_mask = ~torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))

        return src_pad_mask, tgt_pad_mask, causal_mask



    def forward(self, src, tgt):
        src_pad_mask, tgt_pad_mask, causal_mask = self.generate_mask(src, tgt)

        memory = self.encoder(src, src_pad_mask)

        decoder_output = self.decoder(tgt, memory, tgt_mask=tgt_pad_mask, memory_mask=memory)
        return decoder_output