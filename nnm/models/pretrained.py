import os
import re

import torch
from torch import nn

from conippets.config import Config


class PretrainedModel(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        config_path = os.path.join(pretrained_path, 'config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')

        config = Config.from_json(config_path)

        weight_path = getattr(config, '_nnm_weight_path', 'model.pth')
        model_path = os.path.join(pretrained_path, weight_path)

        if not os.path.exists(model_path):
            repo = getattr(config, '_nnm_weight_repo', None)
            if repo:
                repo_path = os.path.expanduser(repo)
                model_path = os.path.join(repo_path, weight_path)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f'Weight file not found: {model_path}. '
                        f'Please download weights from HuggingFace repo: {repo}'
                    )
            else:
                raise FileNotFoundError(f'Weight file not found: {model_path}')

        model = cls(config, **kwargs)
        model._load_weights(model_path)
        return model

    def _load_weights(self, path, weights_only=True):
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=weights_only)
            state_dict = checkpoint.get('state_dict', checkpoint)

        key_mapping = getattr(self.config, '_nnm_key_mapping', None)
        if key_mapping:
            state_dict = self._apply_key_mapping(state_dict, key_mapping)

        self.load_state_dict(state_dict, strict=True)

    def _apply_key_mapping(self, state_dict, key_mapping):
        mapped = {}
        for key, tensor in state_dict.items():
            new_key = key
            for pattern, replacement in key_mapping:
                new_key = re.sub(pattern, replacement, new_key)
            mapped[new_key] = tensor
        return mapped
