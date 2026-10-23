import os

import torch
from torch import nn

from conippets.config import Config

from nnm.models.laya.encoder import ModernBert
from nnm.models.pretrained import PretrainedModel


class Laya(PretrainedModel):
    num_types = 3
    head_width = 256
    head_dropout = 0.1
    weight_pattern = 'model.safetensors'

    @classmethod
    def load_config(cls, directory):
        config = super().load_config(directory)
        config.encoder = Config.from_json(os.path.join(directory, 'encoder', 'config.json'))
        agent = Config.from_json(os.path.join(directory, 'rl_agent_config.json'))
        config.head_layers = agent.head_layers
        config.num_actions = len(agent.act_costs) + 1
        return config

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ModernBert(config.encoder)
        hidden = config.encoder.hidden_size
        self.head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, hidden // 64, 4 * hidden, self.head_dropout, batch_first=True, norm_first=True),
            config.head_layers,
            enable_nested_tensor=False,
        )
        self.type_emb = nn.Embedding(self.num_types, hidden)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.act_head = nn.Sequential(
            nn.Linear(hidden + 4, self.head_width), nn.GELU(), nn.Linear(self.head_width, config.num_actions)
        )
        self.register_buffer("temperature", torch.ones(self.num_types))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, position_ids=None):
        hidden = self.encoder(input_ids, attention_mask, position_ids)
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        padding = ~attention_mask.bool()
        for layer in self.head.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)

        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        logits = self.scorer(torch.gather(hidden, 1, index)).squeeze(-1)
        logits = logits.masked_fill(~marker_mask.bool(), -1e4)
        return logits, self.action_logits(hidden, logits, marker_mask)

    def action_logits(self, hidden, logits, marker_mask):
        probs = torch.softmax(logits.detach().float(), -1)
        count = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(-1) / torch.log(count)
        top = probs.topk(min(2, probs.size(-1)), -1).values
        if top.size(-1) < 2:
            top = torch.cat([top, torch.zeros_like(top)], dim=-1)
        features = torch.stack([top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255.0], -1)
        return self.act_head(torch.cat([hidden[:, 0].float(), features], -1))
