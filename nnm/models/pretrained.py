import os
import re
import subprocess

import torch
from torch import nn

from conippets.config import Config

_DEFAULT_HF_ROOT = os.path.expanduser('~/.nnm/huggingface')


class PretrainedModel(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        download_root = (
            kwargs.pop('download_root', None)
            or os.environ.get('NNM_HF_ROOT')
            or _DEFAULT_HF_ROOT
        )

        if not os.path.exists(pretrained_path):
            local_path = os.path.join(download_root, pretrained_path)
            if not os.path.exists(local_path):
                cls._download_from_hf(pretrained_path, local_path)
            pretrained_path = local_path

        config_path = os.path.join(pretrained_path, 'config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')

        config = Config.from_json(config_path)

        weight_names = getattr(config, '_nnm_weight_path', 'model.pth')
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        elif not isinstance(weight_names, list):
            raise TypeError(f'_nnm_weight_path must be a string or list of strings, got {type(weight_names).__name__}')

        model_paths = [os.path.join(pretrained_path, w) for w in weight_names]

        missing = [p for p in model_paths if not os.path.exists(p)]
        if missing:
            repo = getattr(config, '_nnm_weight_repo', None)
            if repo:
                repo_path = os.path.expanduser(repo)
                model_paths = [os.path.join(repo_path, w) for w in weight_names]
                missing = [p for p in model_paths if not os.path.exists(p)]
                if missing:
                    raise FileNotFoundError(
                        f'Weight file not found: {missing[-1]}. '
                        f'Please download weights from HuggingFace repo: {repo}'
                    )
            else:
                raise FileNotFoundError(f'Weight file not found: {missing[0]}')

        model = cls(config, **kwargs)
        model._load_weights(model_paths)
        return model

    @classmethod
    def _download_from_hf(cls, repo_id, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            subprocess.run(
                ['hf', 'download', repo_id, '--local-dir', local_path],
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError('hf command not found. Please install hf CLI first (e.g., pip install huggingface-hub)')

    def _load_weights(self, paths, weights_only=True):
        if isinstance(paths, str):
            paths = [paths]

        state_dict = {}
        for path in paths:
            if path.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(path)
            else:
                checkpoint = torch.load(path, map_location='cpu', weights_only=weights_only)
                sd = checkpoint.get('state_dict', checkpoint)
            state_dict.update(sd)

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
