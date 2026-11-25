import torch
from functools import partial
from torch import nn

from nnm.layers.attention import InterleavedRotarySelfAttention
from nnm.layers.mlp import GatedMLP
from nnm.layers.norm import RMSNorm
from nnm.layers.transformer import TransformerBlock
from nnm.models.falcon_h1 import FalconH1Model
from nnm.pretrained import PretrainedModel


class Audio8TTS(PretrainedModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.slow = FalconH1Model(config)
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.hidden_size)
        self.fast_project_in = (
            nn.Identity() if config.fast_dim == config.hidden_size else nn.Linear(config.hidden_size, config.fast_dim)
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.fast_dim,
                    InterleavedRotarySelfAttention(
                        config.fast_dim,
                        config.fast_n_head,
                        config.fast_n_local_heads,
                        config.fast_head_dim,
                        qkv_bias=config.fast_attention_qkv_bias,
                        o_bias=config.fast_attention_o_bias,
                    ),
                    GatedMLP(config.fast_dim, config.fast_intermediate_size),
                    norm_layer=partial(RMSNorm, eps=config.norm_eps),
                )
                for _ in range(config.n_fast_layer)
            ]
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        self.semantic_output = nn.Linear(config.hidden_size, config.codebook_size + 1, bias=False)
        self.semantic_begin_id = config.semantic_begin_id
        self.semantic_end_id = config.semantic_end_id
        self.eos_token_id = config.eos_token_id
        self.codebook_size = config.codebook_size
        self.num_codebooks = config.num_codebooks

    def _embed(self, input_ids):
        codebook_embeds = []
        for index in range(self.num_codebooks):
            codebook_embeds.append(self.codebook_embeddings(input_ids[:, index + 1] + index * self.codebook_size))
        codebook_sum = torch.stack(codebook_embeds, dim=1).sum(dim=1)
        semantic = (input_ids[:, 0] >= self.semantic_begin_id) & (input_ids[:, 0] <= self.semantic_end_id)
        codebook_sum = torch.where(semantic.unsqueeze(-1), codebook_sum, 0.0)
        return self.slow.embed_tokens(input_ids[:, 0]) + codebook_sum

    def forward(self, input_ids, attn_mask=None):
        hidden = self._embed(input_ids)
        normalized, _, _ = self.slow(hidden, attn_mask=attn_mask)
        logits = self.semantic_output(normalized)
        return logits, normalized
