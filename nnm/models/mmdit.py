import torch
from functools import partial
from torch import nn
from torch.nn import functional as F

from nnm.layers.attention import SelfAttention
from nnm.layers.dit import AdaLNModulation, FinalLayer, TimestepEmbedder, modulate
from nnm.layers.norm import RMSNorm
from nnm.layers.mlp import MLP
from nnm.layers.vit import PatchEmbed2D
from nnm.models.pretrained import PretrainedModel


class JointAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, *, qkv_bias=True, o_bias=True, qk_norm=False, norm_eps=1e-6):
        super().__init__()
        self.attn = SelfAttention(
            hidden_size,
            num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            causal=False,
            qk_norm=qk_norm,
            norm_layer=partial(RMSNorm, eps=norm_eps),
        )
        self.num_heads = self.attn.num_heads
        self.head_dim = self.attn.head_dim

    def project(self, hidden):
        batch, length, _ = hidden.shape
        query, key, value = self.attn.project(hidden)
        query = query.view(batch, length, self.num_heads, self.head_dim)
        key = key.view(batch, length, self.num_heads, self.head_dim)
        value = value.view(batch, length, self.num_heads, self.head_dim)
        if self.attn.q_norm is not None:
            query = self.attn.q_norm(query)
            key = self.attn.k_norm(key)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


class MMDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        context_dim,
        mlp_ratio=4.0,
        *,
        qkv_bias=True,
        o_bias=True,
        qk_norm=False,
        norm_eps=1e-6,
        context_pre_only=False,
        mlp_act="gelu",
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.img_norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.img_attn = JointAttention(
            hidden_size, num_heads, qkv_bias=qkv_bias, o_bias=o_bias, qk_norm=qk_norm, norm_eps=norm_eps
        )
        self.img_norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.img_mlp = MLP(hidden_size, int(hidden_size * mlp_ratio), mlp_act, bias=True)
        self.txt_norm1 = nn.LayerNorm(context_dim, eps=norm_eps)
        self.txt_attn = JointAttention(
            context_dim, num_heads, qkv_bias=qkv_bias, o_bias=o_bias, qk_norm=qk_norm, norm_eps=norm_eps
        )
        self.txt_norm2 = nn.LayerNorm(context_dim, eps=norm_eps) if not context_pre_only else None
        self.txt_mlp = MLP(context_dim, int(context_dim * mlp_ratio), mlp_act, bias=True) if not context_pre_only else None
        self.img_mod = AdaLNModulation(hidden_size, 6)
        self.txt_mod = AdaLNModulation(hidden_size, 6)

    def forward(self, hidden, context, condition):
        img_mod = self.img_mod(condition).chunk(6, dim=1)
        txt_mod = self.txt_mod(condition).chunk(6, dim=1)
        img_shift_attn, img_scale_attn, img_gate_attn, img_shift_mlp, img_scale_mlp, img_gate_mlp = img_mod
        txt_shift_attn, txt_scale_attn, txt_gate_attn, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = txt_mod

        img_norm = modulate(self.img_norm1(hidden), img_shift_attn, img_scale_attn)
        txt_norm = modulate(self.txt_norm1(context), txt_shift_attn, txt_scale_attn)
        img_q, img_k, img_v = self.img_attn.project(img_norm)
        txt_q, txt_k, txt_v = self.txt_attn.project(txt_norm)

        query = torch.cat([img_q, txt_q], dim=2)
        key = torch.cat([img_k, txt_k], dim=2)
        value = torch.cat([img_v, txt_v], dim=2)
        output = F.scaled_dot_product_attention(query, key, value, scale=self.img_attn.head_dim**-0.5)
        output = output.transpose(1, 2).reshape(query.shape[0], -1, query.shape[1] * query.shape[3])
        img_output, txt_output = output[:, : hidden.shape[1]], output[:, hidden.shape[1] :]

        hidden = hidden + img_gate_attn.unsqueeze(1) * self.img_attn.attn.o_proj(img_output)
        context = context + txt_gate_attn.unsqueeze(1) * self.txt_attn.attn.o_proj(txt_output)

        hidden = hidden + img_gate_mlp.unsqueeze(1) * self.img_mlp(
            modulate(self.img_norm2(hidden), img_shift_mlp, img_scale_mlp)
        )
        if not self.context_pre_only:
            context = context + txt_gate_mlp.unsqueeze(1) * self.txt_mlp(
                modulate(self.txt_norm2(context), txt_shift_mlp, txt_scale_mlp)
            )
        return hidden, context


class MMDiT(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        patch_size = config.get("patch_size", 2)
        in_channels = config.get("in_channels", 4)
        context_dim = config.get("context_dim", config.get("text_hidden_size", hidden_size))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        mlp_ratio = config.get("mlp_ratio", 4.0)
        qkv_bias = config.get("qkv_bias", True)
        o_bias = config.get("o_bias", True)
        qk_norm = config.get("qk_norm", False)
        norm_eps = config.get("norm_eps", config.get("layer_norm_eps", 1e-6))
        mlp_act = config.get("mlp_act", config.get("hidden_act", "gelu"))
        learn_sigma = config.get("learn_sigma", False)
        num_classes = config.get("num_classes", 0)
        image_size = config.get("image_size", 64)

        self.x_embedder = PatchEmbed2D(
            kernel_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True, norm=False
        )
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.context_embedder = nn.Linear(context_dim, context_dim, bias=True)
        self.y_embedder = nn.Embedding(num_classes, hidden_size) if num_classes else None
        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    hidden_size,
                    num_heads,
                    context_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    o_bias=o_bias,
                    qk_norm=qk_norm,
                    norm_eps=norm_eps,
                    context_pre_only=index == num_layers - 1,
                    mlp_act=mlp_act,
                )
                for index in range(num_layers)
            ]
        )
        out_channels = in_channels * 2 if learn_sigma else in_channels
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(self, hidden_states, timestep, context, class_labels=None):
        hidden, (height, width) = self.x_embedder(hidden_states)
        hidden = hidden + self.pos_embed[:, : hidden.shape[1]]
        condition = self.t_embedder(timestep)
        if self.y_embedder is not None and class_labels is not None:
            condition = condition + self.y_embedder(class_labels)
        context = self.context_embedder(context)
        for block in self.blocks:
            hidden, context = block(hidden, context, condition)
        hidden = self.final_layer(hidden, condition)
        batch = hidden.shape[0]
        hidden = hidden.reshape(batch, height, width, self.patch_size, self.patch_size, self.out_channels)
        hidden = hidden.permute(0, 5, 1, 3, 2, 4)
        return hidden.reshape(batch, self.out_channels, height * self.patch_size, width * self.patch_size)
