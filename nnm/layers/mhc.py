import torch
from torch import nn

from nnm.layers.norm import rms_unit

SINKHORN_ITERS = 20


def sinkhorn(logits, iters=SINKHORN_ITERS):
    for _ in range(iters):
        logits = logits - torch.logsumexp(logits, -1, keepdim=True)
        logits = logits - torch.logsumexp(logits, -2, keepdim=True)
    return logits.exp()


class MHCStack(nn.Module):
    def __init__(self, d_model, num_layers, lanes):
        super().__init__()
        self.d_model, self.num_layers, self.lanes = d_model, num_layers, lanes
        n_lane = lanes * d_model
        self.a_pre = nn.Parameter(torch.zeros(num_layers))
        self.a_post = nn.Parameter(torch.zeros(num_layers))
        self.a_res = nn.Parameter(torch.zeros(num_layers))
        self.b_pre = nn.Parameter(torch.zeros(num_layers, lanes))
        self.b_post = nn.Parameter(torch.zeros(num_layers, lanes))
        self.b_res = nn.Parameter(torch.zeros(num_layers, lanes, lanes))
        self.phi_pre = nn.Parameter(torch.zeros(num_layers, lanes, n_lane))
        self.phi_post = nn.Parameter(torch.zeros(num_layers, lanes, n_lane))
        self.phi_res = nn.Parameter(torch.zeros(num_layers, lanes * lanes, n_lane))
        self.register_buffer("lane", torch.eye(lanes), persistent=False)
        self.register_buffer("lane_index", torch.arange(num_layers) % lanes, persistent=False)

    def layer_params(self, index):
        lane = self.lane[self.lane_index[index]]
        return dict(
            pre_off=8 * lane - 4,
            post_off=-4 * (1 - lane),
            phi_pre=self.phi_pre[index].T,
            phi_post=self.phi_post[index].T,
            phi_res=self.phi_res[index].T,
            a_pre=self.a_pre[index],
            a_post=self.a_post[index],
            a_res=self.a_res[index],
            b_pre=self.b_pre[index],
            b_post=self.b_post[index],
            b_res=self.b_res[index],
        )

    def advance(self, stream, block, params):
        batch, length, lanes, channels = stream.shape
        wide = rms_unit(stream.reshape(batch, length, lanes * channels))
        h_pre = torch.sigmoid(params["a_pre"] * (wide @ params["phi_pre"]) + params["b_pre"] + params["pre_off"])
        mixed = torch.einsum("btn,btnc->btc", h_pre, stream)
        delta = block(mixed) - mixed
        h_post = 2 * torch.sigmoid(params["a_post"] * (wide @ params["phi_post"]) + params["b_post"] + params["post_off"])
        h_res = sinkhorn(params["a_res"] * (wide @ params["phi_res"]).reshape(batch, length, lanes, lanes) + params["b_res"])
        return torch.einsum("btij,btjc->btic", h_res, stream) + h_post[..., None] * delta[:, :, None, :]
