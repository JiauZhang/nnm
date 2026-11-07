import torch
from torch import nn


class Engram(nn.Module):
    def __init__(self, d_model, tables, slots, sub_dim, num_layers, dilation, orders, heads):
        super().__init__()
        self.d_model, self.tables, self.slots, self.sub_dim = d_model, tables, slots, sub_dim
        self.dilation, self.orders, self.heads = dilation, orders, heads
        self.embedding = nn.Parameter(torch.zeros(tables, slots, sub_dim))
        self.key_proj = nn.Linear(tables * sub_dim, d_model, bias=False)
        self.value_proj = nn.Linear(tables * sub_dim, d_model, bias=False)
        self.taps = nn.Parameter(torch.zeros(4, d_model))

    def forward(self, tokens):
        length = tokens.shape[-1]
        positions = torch.arange(length, device=tokens.device)
        tables = len(self.orders) * self.heads
        indices = engram_indices(tokens, self.orders, self.heads, self.slots)
        offsets = torch.arange(tables, device=tokens.device) * self.slots
        flat = (indices + offsets).reshape(-1)
        fetched = self.embedding.reshape(-1, self.sub_dim)[flat]
        fetched = fetched.reshape(*tokens.shape, tables, self.sub_dim)
        order_mask = torch.stack([positions >= order - 1 for order in self.orders for _ in range(self.heads)], -1)
        fetched = (fetched * order_mask.to(fetched.dtype)[None, :, :, None]).reshape(*tokens.shape, -1)
        key = self.key_proj(fetched)
        value = self.value_proj(fetched)
        tapped = self.taps[0] * value
        for tap in range(1, self.taps.shape[0]):
            offset = tap * self.dilation
            shifted = (
                torch.cat([torch.zeros_like(value[:, :offset]), value[:, :-offset]], dim=1)
                if offset < length
                else torch.zeros_like(value)
            )
            tapped = tapped + self.taps[tap] * shifted * (positions >= offset).to(value.dtype)[None, :, None]
        return key, tapped


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    u = tokens.long() & 0xFFFFFFFF
    stride = seed_heads or heads
    out = []
    for order_index, order in enumerate(orders):
        for head in range(heads):
            seed = (0x9E3779B9 * (order_index * stride + head + 1)) & 0xFFFFFFFF
            acc = torch.full_like(u, seed)
            for j in range(order):
                shifted = torch.cat([torch.zeros_like(u[:, :j]), u[:, :-j]], dim=1) if j else u
                acc = ((acc ^ shifted) * 0x01000193) & 0xFFFFFFFF
            acc = acc ^ (acc >> 15)
            out.append(acc % slots)
    return torch.stack(out, -1)
