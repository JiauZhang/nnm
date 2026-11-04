import torch
from torch import nn
from torch.nn import functional as F

from nnm.layers.mlp import GateMLP


class MoEGate(nn.Module):
    def __init__(
        self, *, hidden_size, n_routed_experts, num_experts_per_tok, scoring_func, norm_topk_prob, routed_scaling_factor
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, hidden_states.shape[-1]).float(), self.weight.float())
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise ValueError(f"unsupported scoring_func: {self.scoring_func}")
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


class MoE(nn.Module):
    def __init__(
        self,
        *,
        hidden_size,
        moe_intermediate_size,
        n_routed_experts,
        n_shared_experts,
        num_experts_per_tok,
        scoring_func,
        norm_topk_prob,
        routed_scaling_factor,
        act=nn.SiLU,
    ):
        super().__init__()
        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            scoring_func=scoring_func,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.experts = nn.ModuleList([GateMLP(hidden_size, moe_intermediate_size, act=act) for _ in range(n_routed_experts)])
        self.shared_experts = GateMLP(hidden_size, moe_intermediate_size * n_shared_experts, act=act)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        topk_idx, topk_weight = self.gate(x)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_index, slot_index = (topk_idx == expert_index).nonzero(as_tuple=True)
            if token_index.numel() == 0:
                continue
            expert_out = expert(flat[token_index]) * topk_weight[token_index, slot_index].unsqueeze(-1)
            output = output.index_add(0, token_index, expert_out.to(output.dtype))
        return output.reshape(x.shape) + self.shared_experts(x)
