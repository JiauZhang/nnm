import math

import torch
from conippets.config import Config
from torch import nn
from torch.nn import functional as F

from nnm.models.dit import DiT


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    arguments = timesteps.float()[:, None] * frequencies[None]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)


class ReferenceDiT(nn.Module):
    def __init__(self, hidden=32, patch=2, depth=2, heads=4, image=32, classes=10):
        super().__init__()
        self.patch = patch
        self.x_embedder = nn.Conv2d(3, hidden, patch, patch)
        self.t_mlp = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.y_embedding = nn.Embedding(classes, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, (image // patch) ** 2, hidden))
        self.blocks = nn.ModuleList([self._block(hidden, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.final_linear = nn.Linear(hidden, patch * patch * 3)

    @staticmethod
    def _block(hidden, heads):
        block = nn.Module()
        block.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        block.qkv = nn.Linear(hidden, 3 * hidden)
        block.proj = nn.Linear(hidden, hidden)
        block.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        block.fc1 = nn.Linear(hidden, hidden * 4)
        block.fc2 = nn.Linear(hidden * 4, hidden)
        block.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        block.heads = heads
        return block

    def forward(self, images, timestep, labels):
        hidden = self.x_embedder(images).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embed
        condition = self.t_mlp(timestep_embedding(timestep, 256)) + self.y_embedding(labels)
        for block in self.blocks:
            shift1, scale1, gate1, shift2, scale2, gate2 = block.modulation(condition).chunk(6, dim=1)
            normed = block.norm1(hidden) * (1 + scale1[:, None]) + shift1[:, None]
            batch, length, _ = normed.shape
            qkv = block.qkv(normed).reshape(batch, length, 3, block.heads, -1).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            attn = F.scaled_dot_product_attention(query, key, value)
            attn = attn.transpose(1, 2).reshape(batch, length, -1)
            hidden = hidden + gate1[:, None] * block.proj(attn)
            normed = block.norm2(hidden) * (1 + scale2[:, None]) + shift2[:, None]
            hidden = hidden + gate2[:, None] * block.fc2(F.gelu(block.fc1(normed)))
        shift, scale = self.final_modulation(condition).chunk(2, dim=1)
        hidden = self.final_linear(self.final_norm(hidden) * (1 + scale[:, None]) + shift[:, None])
        grid = int(hidden.shape[1] ** 0.5)
        hidden = hidden.reshape(-1, grid, grid, self.patch, self.patch, 3).permute(0, 5, 1, 3, 2, 4)
        return hidden.reshape(-1, 3, grid * self.patch, grid * self.patch)


def copy_weights(model, reference):
    state = {}
    state["x_embedder.proj.weight"] = reference.x_embedder.weight
    state["x_embedder.proj.bias"] = reference.x_embedder.bias
    state["t_embedder.mlp.0.weight"] = reference.t_mlp[0].weight
    state["t_embedder.mlp.0.bias"] = reference.t_mlp[0].bias
    state["t_embedder.mlp.2.weight"] = reference.t_mlp[2].weight
    state["t_embedder.mlp.2.bias"] = reference.t_mlp[2].bias
    state["y_embedder.weight"] = reference.y_embedding.weight
    state["pos_embed"] = reference.pos_embed
    for index, block in enumerate(reference.blocks):
        prefix = f"blocks.{index}"
        qkv = block.qkv.weight.reshape(3, 32, 32)
        state[f"{prefix}.attn.q_proj.weight"] = qkv[0]
        state[f"{prefix}.attn.k_proj.weight"] = qkv[1]
        state[f"{prefix}.attn.v_proj.weight"] = qkv[2]
        bias = block.qkv.bias.reshape(3, 32)
        state[f"{prefix}.attn.q_proj.bias"] = bias[0]
        state[f"{prefix}.attn.k_proj.bias"] = bias[1]
        state[f"{prefix}.attn.v_proj.bias"] = bias[2]
        state[f"{prefix}.attn.o_proj.weight"] = block.proj.weight
        state[f"{prefix}.attn.o_proj.bias"] = block.proj.bias
        state[f"{prefix}.mlp.fc1.weight"] = block.fc1.weight
        state[f"{prefix}.mlp.fc1.bias"] = block.fc1.bias
        state[f"{prefix}.mlp.fc2.weight"] = block.fc2.weight
        state[f"{prefix}.mlp.fc2.bias"] = block.fc2.bias
        state[f"{prefix}.adaLN_modulation.proj.1.weight"] = block.modulation[1].weight
        state[f"{prefix}.adaLN_modulation.proj.1.bias"] = block.modulation[1].bias
    state["final_layer.adaLN_modulation.proj.1.weight"] = reference.final_modulation[1].weight
    state["final_layer.adaLN_modulation.proj.1.bias"] = reference.final_modulation[1].bias
    state["final_layer.linear.weight"] = reference.final_linear.weight
    state["final_layer.linear.bias"] = reference.final_linear.bias
    model.load_state_dict(state, strict=True)


def test_dit_matches_reference():
    torch.manual_seed(0)
    config = Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=32,
        patch_size=2,
        in_channels=3,
        num_classes=10,
        learn_sigma=False,
    )
    nnm_model = DiT(config).eval()
    reference = ReferenceDiT().eval()
    copy_weights(nnm_model, reference)
    images = torch.randn(2, 3, 32, 32)
    timestep = torch.tensor([3.0, 7.0])
    labels = torch.tensor([1, 9])
    with torch.no_grad():
        assert torch.allclose(nnm_model(images, timestep, labels), reference(images, timestep, labels), atol=1e-5)
