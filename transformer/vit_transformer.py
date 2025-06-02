import torch
import torch.nn as nn
from torch.nn import Module

from transformer.core.vit_encoder import ViTEncoder
from transformer.core.decoder import Decoder

from transformer.config.vit_encoder import ViTEncoderConfig
from transformer.config.decoder import DecoderConfig

class VisionTransformer(Module):
    def __init__(
        self,
        width,
        height,
        patch_size,
        in_channel,
        num_enc_layers,
        num_dec_layers,
        num_heads,
        hidden_dim,
        dropout_p,
        embed_dim,
        tgt_pad_idx,
        tgt_vocab_size,
        max_seq_length,
    ): 
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.embed_dim = embed_dim
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_vocab_size = tgt_vocab_size
        self.max_length = max_seq_length

        vit_enc_config = ViTEncoderConfig(
            height=self.height,
            width=self.width,
            patch_size=self.patch_size,
            in_channel=self.in_channel,
            embed_dim=self.embed_dim,
            num_layers=self.num_enc_layers,
            num_heads = self.num_heads,
            hidden_dim = self.hidden_dim,
            dropout_p=self.dropout_p
        )
        self.encoder = ViTEncoder(vit_enc_config)

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

    def generate_mask(self, tgt):

        _, tgt_len = tgt.shape
        tgt_pad_mask = (tgt == self.tgt_pad_idx)

        causal_mask = ~torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))

        decoder_self_attn_mha_mask = torch.logical_or(
            tgt_pad_mask.unsqueeze(1).unsqueeze(2),
            causal_mask.unsqueeze(0).unsqueeze(0)  
        )

        return decoder_self_attn_mha_mask

    def forward(self, img_src, tgt):
        image_memory = self.encoder(img_src, None)

        decoder_self_attn_mha_mask = self.generate_mask(tgt)
        decoder_output = self.decoder(tgt, image_memory, tgt_mask=decoder_self_attn_mha_mask, memory_mask=None)
        return decoder_output