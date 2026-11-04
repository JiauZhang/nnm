import glob
import importlib.util
import json
import os
from pathlib import Path

import torch
from conippets.config import Config
from torch import nn

MANIFEST_NAME = "nnm_weights.json"
DEFAULT_PATTERNS = ("model.safetensors", "model-*.safetensors", "*.safetensors")
PROVIDERS = ("huggingface", "modelscope")
PROVIDER_ALIASES = {"hf": "huggingface", "ms": "modelscope"}


class PretrainedModel(nn.Module):
    weight_pattern = None
    module_name = None

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        cache_dir = kwargs.pop("download_root", None)
        directory = cls._resolve_pretrained_path(pretrained_path, cache_dir)
        manifest = cls._load_manifest(directory)
        entry = cls._resolve_entry(manifest, kwargs)
        config = cls.load_config(directory, entry)
        model = cls(config, **kwargs)
        state_dict = cls._load_state_dict(directory, config, entry, cache_dir)
        model.load_state_dict(state_dict, strict=True)
        model._post_load(directory)
        return model

    def _post_load(self, directory):
        pass

    @classmethod
    def load_config(cls, directory, manifest=None):
        relative = manifest.get("config", "config.json") if manifest else "config.json"
        config_path = os.path.join(directory, relative)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return Config.from_json(config_path)

    @classmethod
    def _load_manifest(cls, directory):
        path = os.path.join(directory, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        manifest = json.loads(Path(path).read_text())
        if manifest.get("version") != 1:
            raise ValueError(f"Unsupported {MANIFEST_NAME} version: {manifest.get('version')!r}")
        return manifest

    @classmethod
    def manifest_key(cls, **kwargs):
        return cls.module_name

    @classmethod
    def _resolve_entry(cls, manifest, kwargs):
        if manifest is None or cls.module_name is None:
            return manifest
        modules = manifest.get("modules", {})
        key = cls.manifest_key(**kwargs)
        if key not in modules:
            raise KeyError(f"{MANIFEST_NAME} has no module {key!r}")
        return {**manifest, **modules[key]}

    @classmethod
    def _resolve_pretrained_path(cls, pretrained_path, cache_dir=None):
        path = os.path.expanduser(str(pretrained_path))
        if os.path.exists(path):
            return path
        if os.path.isabs(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        provider, repo, subdirectory = cls._split_repo(pretrained_path)
        snapshot = cls._download(repo, provider, cache_dir=cache_dir)
        return os.path.join(snapshot, subdirectory) if subdirectory else snapshot

    @classmethod
    def _split_repo(cls, repo_id):
        value = str(repo_id)
        provider = "huggingface"
        if ":" in value:
            prefix, value = value.split(":", 1)
            provider = PROVIDER_ALIASES.get(prefix, prefix)
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider {provider!r}; expected one of {PROVIDERS}")

        parts = value.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid repo id {repo_id!r}; expected owner/name")
        repo = "/".join(parts[:2])
        subdirectory = os.path.join(*parts[2:]) if len(parts) > 2 else None
        return provider, repo, subdirectory

    @classmethod
    def _download(cls, repo, provider, cache_dir=None, pattern=None):
        kwargs = {"repo_id": repo} if provider == "huggingface" else {"model_id": repo}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if pattern:
            kwargs["allow_patterns"] = [pattern]

        if provider == "huggingface":
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        else:
            from modelscope import snapshot_download  # noqa: PLC0415
        return snapshot_download(**kwargs)

    @classmethod
    def _load_state_dict(cls, directory, config, manifest, cache_dir=None):
        paths = []
        mapping = None
        if manifest:
            for source in manifest.get("sources", []):
                provider = source.get("provider", "huggingface")
                pattern = source.get("pattern")
                snapshot = cls._download(source["repo"], provider, cache_dir=cache_dir, pattern=pattern)
                paths.extend(cls._matching_paths(snapshot, pattern))
            paths.extend(os.path.join(directory, name) for name in manifest.get("weights", []))
            mapping = manifest.get("mapping")
        else:
            pattern = getattr(config, "weight_pattern", cls.weight_pattern)
            paths = cls._matching_paths(directory, pattern) if pattern else cls._auto_paths(directory)

        if not paths:
            raise FileNotFoundError(f"No weight files found for {directory}")

        state_dict = {}
        for path in paths:
            state_dict.update(cls._read_file(path))

        if mapping:
            state_dict = cls._apply_mapping(state_dict, directory, mapping, config)
        return state_dict

    @classmethod
    def _matching_paths(cls, directory, pattern):
        if not pattern:
            return cls._auto_paths(directory)
        return sorted(glob.glob(os.path.join(directory, pattern)))

    @classmethod
    def _auto_paths(cls, directory):
        for pattern in DEFAULT_PATTERNS:
            paths = sorted(glob.glob(os.path.join(directory, pattern)))
            if paths:
                return paths
        return []

    @classmethod
    def _read_file(cls, path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file  # noqa: PLC0415

            return load_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return checkpoint.get("state_dict", checkpoint)

    @classmethod
    def _apply_mapping(cls, state_dict, directory, mapping, config):
        path = os.path.join(directory, mapping)
        spec = importlib.util.spec_from_file_location("nnm_weight_mapping", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.map_state_dict(state_dict, config)
