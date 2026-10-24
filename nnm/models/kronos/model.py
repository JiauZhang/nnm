import torch
from torch import nn

from nnm.models.kronos.module import (
    BSQuantizer,
    DualHead,
    DependencyAwareLayer,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)
from nnm.models.pretrained import PretrainedModel


class KronosTokenizer(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.s1_bits = config.s1_bits
        self.s2_bits = config.s2_bits
        self.codebook_dim = config.s1_bits + config.s2_bits

        self.embed = nn.Linear(config.d_in, config.d_model)
        self.head = nn.Linear(config.d_model, config.d_in)
        self.encoder = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads, config.ff_dim) for _ in range(config.n_enc_layers - 1)]
        )
        self.decoder = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads, config.ff_dim) for _ in range(config.n_dec_layers - 1)]
        )
        self.quant_embed = nn.Linear(config.d_model, self.codebook_dim)
        self.post_quant_embed = nn.Linear(self.codebook_dim, config.d_model)
        self.tokenizer = BSQuantizer(config.s1_bits, config.s2_bits)

    def encode(self, x, half=False):
        hidden = self.embed(x)
        for layer in self.encoder:
            hidden = layer(hidden)
        return self.tokenizer(self.quant_embed(hidden), half=half)

    def decode(self, indices, half=False):
        hidden = self.post_quant_embed(self.indices_to_bits(indices, half))
        for layer in self.decoder:
            hidden = layer(hidden)
        return self.head(hidden)

    def indices_to_bits(self, indices, half=False):
        if half:
            first, second = indices
            mask = 2 ** torch.arange(self.codebook_dim // 2, dtype=torch.long, device=first.device)
            indices = torch.cat([(first.unsqueeze(-1) & mask) != 0, (second.unsqueeze(-1) & mask) != 0], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, dtype=torch.long, device=indices.device)
            indices = (indices.unsqueeze(-1) & mask) != 0
        return (indices.float() * 2 - 1) / (self.codebook_dim**0.5)


class Kronos(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.s1_bits = config.s1_bits
        self.s2_bits = config.s2_bits
        self.s1_vocab_size = 2**config.s1_bits

        self.embedding = HierarchicalEmbedding(config.s1_bits, config.s2_bits, config.d_model)
        self.time_emb = TemporalEmbedding(config.d_model, config.learn_te)
        self.transformer = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads, config.ff_dim) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.dep_layer = DependencyAwareLayer(config.d_model)
        self.head = DualHead(config.s1_bits, config.s2_bits, config.d_model)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, s1_targets=None):
        context = self.encode_context(s1_ids, s2_ids, stamp, padding_mask)
        sibling = self.embedding.emb_s1(s1_ids if s1_targets is None else s1_targets)
        return self.head(context), self.head.cond_forward(self.dep_layer(context, sibling, padding_mask))

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        context = self.encode_context(s1_ids, s2_ids, stamp, padding_mask)
        return self.head(context), context

    def decode_s2(self, context, s1_ids, padding_mask=None):
        sibling = self.embedding.emb_s1(s1_ids)
        return self.head.cond_forward(self.dep_layer(context, sibling, padding_mask))

    def encode_context(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        hidden = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            hidden = hidden + self.time_emb(stamp)
        for layer in self.transformer:
            hidden = layer(hidden, key_padding_mask=padding_mask)
        return self.norm(hidden)
