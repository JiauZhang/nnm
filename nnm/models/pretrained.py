import os

import torch
from torch import nn

from conippets.config import Config


class PretrainedModel(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        config_path = os.path.join(pretrained_path, 'config.json')
        model_path = os.path.join(pretrained_path, 'model.pth')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model weights not found: {model_path}')

        config = Config.from_json(config_path)
        model = cls(config, model_path=model_path, **kwargs)
        return model

    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.load_state_dict(state_dict, strict=True)
