import glob
import os
import re
import subprocess

import torch
from torch import nn

from conippets.config import Config

_DEFAULT_HF_ROOT = os.path.expanduser('~/.nnm/huggingface')


class PretrainedModel(nn.Module):
    weight_pattern = None

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        download_root = kwargs.pop('download_root', None)
        directory = cls._resolve_pretrained_path(pretrained_path, download_root)

        config = cls.load_config(directory)
        model = cls(config, **kwargs)
        model._load_weights(cls._weight_paths(directory, config, download_root))
        model._post_load(directory)
        return model

    def _post_load(self, directory):
        pass

    @classmethod
    def load_config(cls, directory):
        config_path = os.path.join(directory, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')
        return Config.from_json(config_path)

    @classmethod
    def _resolve_pretrained_path(cls, pretrained_path, download_root=None):
        if os.path.exists(pretrained_path) or os.path.isabs(pretrained_path):
            return pretrained_path

        if download_root is None:
            download_root = os.environ.get('NNM_HF_ROOT') or _DEFAULT_HF_ROOT

        parts = pretrained_path.split('/')
        repo_id = '/'.join(parts[:2])
        local_path = os.path.join(download_root, repo_id)
        if not os.path.exists(local_path):
            cls._download_from_hf(repo_id, local_path)
        return os.path.join(local_path, *parts[2:])

    @classmethod
    def _weight_paths(cls, directory, config, download_root=None):
        pattern = getattr(config, 'weight_pattern', cls.weight_pattern)
        if not pattern or not isinstance(pattern, str):
            raise KeyError(f"'weight_pattern' must be a string in config.json, got {pattern!r}")

        paths = sorted(glob.glob(os.path.join(directory, pattern)))
        if paths:
            return paths

        repo = getattr(config, '_nnm_weight_repo', None)
        if not repo:
            raise FileNotFoundError(f'No weight files matching "{pattern}" in {directory}')

        repo_directory = cls._resolve_pretrained_path(repo, download_root)
        paths = sorted(glob.glob(os.path.join(repo_directory, pattern)))
        if not paths:
            raise FileNotFoundError(f'No weight files matching "{pattern}" in weight repo: {repo}')
        return paths

    @classmethod
    def _download_from_hf(cls, repo_id, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            subprocess.run(['hf', 'download', repo_id, '--local-dir', local_path], check=True)
        except FileNotFoundError:
            raise RuntimeError('hf command not found. Please install hf CLI first (e.g., pip install huggingface-hub)')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Failed to download {repo_id}: hf command exited with code {e.returncode}') from e

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

        drop = getattr(self.config, '_nnm_drop', None)
        if drop:
            state_dict = {key: tensor for key, tensor in state_dict.items() if not any(re.search(p, key) for p in drop)}

        transpose = getattr(self.config, '_nnm_transpose', None)
        if transpose:
            state_dict = {
                key: tensor.T.contiguous() if any(re.search(p, key) for p in transpose) else tensor
                for key, tensor in state_dict.items()
            }

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
