import torch
from torch import nn

from nnm.layers.attention import RotarySelfAttention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.mlp import GatedMLP
from nnm.layers.rope import RoPE, rotary_inv_freq


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.get("norm_bias", False))

    def forward(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        hidden_size = config.hidden_size
        norm_bias = config.get("norm_bias", False)
        bias = config.get("attention_bias", False)
        self.attn_norm = nn.Identity() if index == 0 else nn.LayerNorm(hidden_size, eps=config.norm_eps, bias=norm_bias)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=config.norm_eps, bias=norm_bias)
        self.attn = RotarySelfAttention(
            hidden_size,
            config.num_attention_heads,
            head_dim=hidden_size // config.num_attention_heads,
            qkv_bias=bias,
            o_bias=bias,
            causal=False,
        )
        self.mlp = GatedMLP(
            hidden_size,
            config.intermediate_size,
            bias=config.get("mlp_bias", False),
            activation=config.get("hidden_activation", "gelu"),
        )

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), rotary_embeddings=(cos, sin), attn_mask=mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_types = list(config.layer_types)
        self.windows = {
            layer_type: config.local_attention // 2 if layer_type == "sliding_attention" else None
            for layer_type in set(self.layer_types)
        }
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList([ModernBertDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.get("norm_bias", False))
        self.rope = nn.ModuleDict(
            {
                layer_type: RoPE(
                    rotary_inv_freq(
                        config.hidden_size // config.num_attention_heads, config.rope_parameters[layer_type]["rope_theta"]
                    )
                )
                for layer_type in self.windows
            }
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden = self.embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]

        length = input_ids.shape[1]
        ropes = {layer_type: self.rope[layer_type](position_ids) for layer_type in self.windows}
        masks = {
            layer_type: make_causal_mask(
                length,
                length,
                0,
                input_ids.device,
                torch.float32,
                attn_mask=attention_mask,
                sliding_window=window,
                causal=False,
            )
            for layer_type, window in self.windows.items()
        }
        for layer, layer_type in zip(self.layers, self.layer_types):
            cos, sin = ropes[layer_type]
            hidden = layer(hidden, cos, sin, masks[layer_type])
        return self.final_norm(hidden)
