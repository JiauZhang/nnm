from torch import nn

from nnm.layers.drop import DropPath
from nnm.layers.vit import LayerScale


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, self_attn, mlp, *, norm_layer):
        super().__init__()
        self.input_layernorm = norm_layer(hidden_size)
        self.post_attention_layernorm = norm_layer(hidden_size)
        self.self_attn = self_attn
        self.mlp = mlp

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x), **kwargs)
        return x + self.mlp(self.post_attention_layernorm(x))


class CrossTransformerBlock(nn.Module):
    def __init__(self, hidden_size, self_attn, cross_attn, mlp, *, norm_layer):
        super().__init__()
        self.self_attn_layer_norm = norm_layer(hidden_size)
        self.self_attn = self_attn
        self.encoder_attn_layer_norm = norm_layer(hidden_size)
        self.cross_attn = cross_attn
        self.final_layer_norm = norm_layer(hidden_size)
        self.mlp = mlp

    def forward(self, x, memory, attn_mask=None, cross_attn_mask=None, memory_pos=None, self_pos=None):
        x = x + self.self_attn(self.self_attn_layer_norm(x), attn_mask=attn_mask, qk_pos=(self_pos, None))
        x = x + self.cross_attn(
            self.encoder_attn_layer_norm(x),
            memory,
            qk_pos=(None, memory_pos),
            attn_mask=cross_attn_mask,
        )
        return x + self.mlp(self.final_layer_norm(x))


class VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        attn,
        mlp,
        *,
        norm_layer,
        layerscale_init=0.0,
        drop_path_rate=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_size)
        self.attn = attn
        self.ls1 = LayerScale(hidden_size, layerscale_init) if layerscale_init else nn.Identity()
        self.norm2 = norm_layer(hidden_size)
        self.mlp = mlp
        self.ls2 = LayerScale(hidden_size, layerscale_init) if layerscale_init else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, qk_pos=None, attn_mask=None, lengths=None):
        x = x + self.dropout(
            self.drop_path(self.ls1(self.attn(self.norm1(x), qk_pos=qk_pos, attn_mask=attn_mask, lengths=lengths)))
        )
        return x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))


class RotaryVisionBlock(VisionBlock):
    def forward(self, x, rotary_embeddings=None, attn_mask=None, lengths=None):
        x = x + self.dropout(
            self.drop_path(
                self.ls1(
                    self.attn(
                        self.norm1(x),
                        rotary_embeddings=rotary_embeddings,
                        attn_mask=attn_mask,
                        lengths=lengths,
                    )
                )
            )
        )
        return x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
